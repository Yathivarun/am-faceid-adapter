"""Build the training set: faces -> 512 px JPG + AuraFace embedding + caption.

    python prep/build_dataset.py --parquet "data/raw/ffhq512/*.parquet" --images data/raw/celebahq \
        --out data/faces [--limit 200] [--gpu]

AuraFace comes from --aura, else from the config's paths.aura (--config,
default configs/poc.yaml; on the GPU box pass --config configs/gpu.yaml).

Inputs: parquet shards with an `image` column (bytes or {"bytes": ...}) and
an optional `text` caption column (FFHQ-512-caption), and/or image folders
(CelebA-HQ JPGs, official FFHQ 1024 PNGs; every image is resized to --size).

Keeps an image only if AuraFace finds exactly one clear face (det score and
face size above thresholds). Writes:
    <out>/images/<id>.jpg     square, --size px
    <out>/embeds.npy          float16 (N, 512), row i = meta line i
    <out>/meta.jsonl          {"id", "src", "caption", "gender", "age", "det", "split"}
Split: ~2% of images go to "val" (held out, by a hash of the id).
Resumable: re-running skips ids already in meta.jsonl.
"""

import argparse
import glob
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceid.aura import Aura  # noqa: E402

_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _parquet_items(pattern):
    import pyarrow.parquet as pq
    for shard in sorted(glob.glob(pattern)):
        table = pq.read_table(shard)
        cols = table.column_names
        images = table.column("image").to_pylist()
        texts = table.column("text").to_pylist() if "text" in cols else [None] * len(images)
        stem = Path(shard).stem.split("-of-")[0]
        for i, (img, text) in enumerate(zip(images, texts)):
            data = img["bytes"] if isinstance(img, dict) else img
            yield f"ffhq_{stem}_{i:05d}", shard, data, text


def _folder_items(folder, prefix):
    root = Path(folder)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in _EXT:
            # id from the path inside the folder: same file names in different
            # sub-folders must not collide
            rel = p.relative_to(root).with_suffix("").as_posix().replace("/", "__")
            yield f"{prefix}_{rel}", str(p), p.read_bytes(), None


def _square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.LANCZOS) if s != size else img


def _caption(text, gender, age):
    if text:
        return text.strip()
    who = {1: "man", 0: "woman"}.get(gender, "person")
    return f"a photo of a {who}" + (f", around {int(age)} years old" if age else "")


def _split(item_id, val_pct):
    return "val" if int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 100 < val_pct else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", help="glob of parquet shards (image [+ text] columns)")
    ap.add_argument("--images", action="append", default=[], help="image folder (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/poc.yaml", help="source of paths.aura (the default)")
    ap.add_argument("--aura", help="AuraFace-v1 pack directory (overrides the config)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--min-det", type=float, default=0.6)
    ap.add_argument("--min-face-frac", type=float, default=0.15, help="face box side / image side")
    ap.add_argument("--val-pct", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept images (smoke tests)")
    ap.add_argument("--gpu", action="store_true", help="AuraFace on CUDA (onnxruntime-gpu)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path, emb_path = out / "meta.jsonl", out / "embeds.npy"
    done, embeds = set(), []
    if meta_path.exists():
        lines = []
        for line in meta_path.read_text().splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                break                           # a line cut off by a kill
        embeds = list(np.load(emb_path)) if emb_path.exists() else []
        n = min(len(lines), len(embeds))
        if len(lines) != len(embeds):
            # Killed between checkpoints: meta.jsonl and embeds.npy are saved at
            # different moments. Keep the part both agree on; the rest is redone.
            print(f"resume: meta has {len(lines)}, embeds {len(embeds)}; keeping the first {n}")
            lines, embeds = lines[:n], embeds[:n]
            meta_path.write_text("".join(json.dumps(m) + "\n" for m in lines))
            np.save(emb_path, np.stack(embeds) if embeds else np.zeros((0, 512), np.float16))
        done = {m["id"] for m in lines}
        print(f"resume: {len(done)} faces already built")

    if not args.aura:
        import yaml
        root = Path(__file__).resolve().parent.parent
        cfg_path = Path(args.config) if Path(args.config).is_absolute() else root / args.config
        aura_path = Path(yaml.safe_load(cfg_path.read_text())["paths"]["aura"])
        args.aura = str(aura_path if aura_path.is_absolute() else (root / aura_path).resolve())
    print(f"AuraFace: {args.aura}")
    aura = Aura(args.aura, gpu=args.gpu)
    sources = []
    if args.parquet:
        sources.append(_parquet_items(args.parquet))
    for i, folder in enumerate(args.images):
        sources.append(_folder_items(folder, f"img{i}"))

    kept = rejected = 0
    with meta_path.open("a") as meta:
        for source in sources:
            for item_id, src, data, text in source:
                if item_id in done:
                    continue
                try:
                    img = _square(Image.open(io.BytesIO(data)).convert("RGB"), args.size)
                except Exception:  # noqa: BLE001 - unreadable file
                    rejected += 1
                    continue
                bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                faces = aura.faces(bgr)
                clear = [f for f in faces if f.det_score >= args.min_det
                         and (f.bbox[2] - f.bbox[0]) >= args.min_face_frac * args.size]
                if len(clear) != 1:
                    rejected += 1
                    continue
                f = clear[0]
                img.save(out / "images" / f"{item_id}.jpg", quality=95)
                embeds.append(f.normed_embedding.astype(np.float16))
                gender = int(f.gender) if getattr(f, "gender", None) is not None else None
                age = float(f.age) if getattr(f, "age", None) is not None else None
                meta.write(json.dumps({"id": item_id, "src": src, "caption": _caption(text, gender, age),
                                       "gender": gender, "age": age, "det": round(float(f.det_score), 3),
                                       "split": _split(item_id, args.val_pct)}) + "\n")
                done.add(item_id)
                kept += 1
                if kept % 500 == 0:
                    np.save(emb_path, np.stack(embeds))
                    meta.flush()
                    print(f"kept {kept} (rejected {rejected})", flush=True)
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break
    np.save(emb_path, np.stack(embeds) if embeds else np.zeros((0, 512), np.float16))
    print(f"done: kept {kept} new, rejected {rejected}, total {len(embeds)} -> {out}")


if __name__ == "__main__":
    main()
