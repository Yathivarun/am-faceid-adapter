"""Likeness evaluation: our AuraFace adapter vs h94 full-face (production) and
plus-face, on the same reference faces, seeds and prompts.

    python eval.py --config configs/poc.yaml --adapter checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors
    python eval.py ... --refs ../am-ml-fullbody-sketch/dev/face_inpaint/mul_face/p1 --modes ours full-face

References: held-out "val" faces from the dataset (default) and/or image folders
(--refs, repeatable). Score = AuraFace cosine between each generated face and
its reference. Writes <out>/eval/scores.csv, summary.json and one sheet per
prompt (rows = references; columns = reference | each mode).

Vector alignment (--align, mode "ours" only): an adapter trained on synthetic
faces sees real faces' vectors from outside its training cloud. --align maps
each reference vector from the --align-src distribution (real faces: a built
dataset folder or a folder of photos) onto the --align-tgt one (the training
set) before it enters the adapter:
    mean   v' = unit(v - mean_src + mean_tgt)
    coral  v' = unit((v - mean_src) C_src^-1/2 C_tgt^1/2 + mean_tgt)
           (covariances shrunk toward the identity when the source is small)
Scores are still taken against the ORIGINAL vectors.

    python eval.py --config configs/commercial.yaml --paths-from configs/gpu.yaml --data data/faces_sfhq \
        --adapter checkpoints/commercial/step-0017500/faceid_aura_sd15.safetensors --modes ours \
        --refs <visitors> --align coral --align-src <real photos or dataset> --align-tgt data/faces_sfhq
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faceid.aura import Aura, cosine  # noqa: E402
from train import build_models, load_config  # noqa: E402

H94 = {"full-face": "ip-adapter-full-face_sd15.safetensors", "plus-face": "ip-adapter-plus-face_sd15.safetensors"}
PROMPTS = ["a photo of a person, high quality", "a pencil sketch portrait of a person, monochrome"]


def references(cfg, refs, n_val):
    out = []
    data = Path(cfg["paths"]["data"])
    if n_val and (data / "meta.jsonl").exists():
        lines = [json.loads(line) for line in (data / "meta.jsonl").read_text().splitlines() if line]
        out += [data / "images" / f"{m['id']}.jpg" for m in lines if m["split"] == "val"][:n_val]
    for folder in refs:
        out += sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return out


_IMG = {".jpg", ".jpeg", ".png", ".webp"}


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def vector_set(path, aura, limit=20000, seed=0):
    """Unit AuraFace vectors of a built dataset folder (embeds.npy) or of every
    image under a folder of photos."""
    p = Path(path)
    if (p / "embeds.npy").exists():
        emb = np.load(p / "embeds.npy").astype(np.float32)
        if len(emb) > limit:
            emb = emb[np.random.default_rng(seed).choice(len(emb), size=limit, replace=False)]
        return _unit(emb)
    vecs = [e for f in sorted(p.rglob("*")) if f.suffix.lower() in _IMG
            for e in [aura.embed_path(f)[0]] if e is not None]
    if not vecs:
        raise SystemExit(f"no faces found under {p}")
    return _unit(np.stack(vecs))


def _sqrtm(c, inverse=False):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-8, None)
    return (v * (w ** (-0.5 if inverse else 0.5))) @ v.T


def _cov(x):
    """Covariance shrunk toward a scaled identity (a few hundred real photos
    cannot pin down a 512x512 covariance on their own)."""
    n, d = x.shape
    s = np.cov(x, rowvar=False)
    a = d / (n + d)
    return (1 - a) * s + a * (np.trace(s) / d) * np.eye(d)


def make_aligner(kind, src, tgt):
    """v -> aligned unit vector, mapping the src distribution onto tgt."""
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    if kind == "mean":
        return lambda v: _unit(v - mu_s + mu_t)
    m = _sqrtm(_cov(src), inverse=True) @ _sqrtm(_cov(tgt))
    return lambda v: _unit((v - mu_s) @ m + mu_t)


def pipeline(cfg, mode, adapter, dtype, device):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    tok, text, vae, unet, _ = build_models(cfg["paths"]["sd15"], dtype, device)
    sched = DPMSolverMultistepScheduler.from_pretrained(cfg["paths"]["sd15"], subfolder="scheduler",
                                                        local_files_only=True)
    enc = fe = None
    if mode != "ours":
        enc = CLIPVisionModelWithProjection.from_pretrained(cfg["paths"]["image_encoder"], torch_dtype=dtype,
                                                            local_files_only=True).to(device)
        fe = CLIPImageProcessor()
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tok, unet=unet, scheduler=sched,
                                   safety_checker=None, feature_extractor=fe, image_encoder=enc,
                                   requires_safety_checker=False)
    if mode == "ours":
        a = Path(adapter)
        pipe.load_ip_adapter(str(a.parent), subfolder="", weight_name=a.name, image_encoder_folder=None,
                             local_files_only=True)
    else:
        pipe.load_ip_adapter(cfg["paths"]["ip_adapter_dir"], subfolder="", weight_name=H94[mode],
                             image_encoder_folder=None, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--paths-from", help="take the paths section from this config (e.g. configs/gpu.yaml)")
    ap.add_argument("--adapter", help="our adapter .safetensors (needed for mode 'ours')")
    ap.add_argument("--modes", nargs="+", default=["ours", "full-face", "plus-face"],
                    choices=["ours", "full-face", "plus-face"])
    ap.add_argument("--refs", action="append", default=[], help="folder of reference faces (repeatable)")
    ap.add_argument("--n-val", type=int, default=16, help="held-out dataset faces to include")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.8, help="IP-Adapter scale for every mode")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--data", help="dataset folder (overrides paths.data)")
    ap.add_argument("--out", help="output folder (overrides paths.out)")
    ap.add_argument("--align", choices=["none", "mean", "coral"], default="none",
                    help="map reference vectors onto the training distribution (mode 'ours')")
    ap.add_argument("--align-src", help="real faces: dataset folder or photo folder (source statistics)")
    ap.add_argument("--align-tgt", help="training set folder (target statistics), e.g. data/faces_sfhq")
    args = ap.parse_args()
    if args.align != "none" and not (args.align_src and args.align_tgt):
        raise SystemExit("--align needs --align-src and --align-tgt")

    cfg = load_config(args.config, {}, {"data": args.data, "out": args.out}, args.paths_from)
    cfg["paths"].setdefault("ip_adapter_dir", str((ROOT / "../am-ml-fullbody-sketch/dev/models/ip_adapter").resolve()))
    cfg["paths"].setdefault("image_encoder", str((ROOT / "../am-ml-fullbody-sketch/dev/models/image_encoder").resolve()))
    if "ours" in args.modes and not args.adapter:
        raise SystemExit("--adapter is required for mode 'ours'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    refs = references(cfg, args.refs, args.n_val)
    if not refs:
        raise SystemExit("no reference faces (build the dataset or pass --refs)")

    aura = Aura(cfg["paths"]["aura"])
    ref_emb = {}
    for r in refs:
        e, _ = aura.embed_path(r)
        if e is not None:
            ref_emb[r] = e
    refs = list(ref_emb)
    align = lambda v: v  # noqa: E731
    if args.align != "none":
        src, tgt = vector_set(args.align_src, aura), vector_set(args.align_tgt, aura)
        align = make_aligner(args.align, src, tgt)
        moved = np.mean([float(np.dot(align(ref_emb[r]), ref_emb[r])) for r in refs])
        print(f"align {args.align}: src {len(src)} vectors, tgt {len(tgt)}; "
              f"mean cosine(aligned, original) over references {moved:.3f}")
    out = Path(cfg["paths"]["out"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rows, images = [], {}

    for mode in args.modes:
        pipe = pipeline(cfg, mode, args.adapter, dtype, device)
        pipe.set_ip_adapter_scale(args.scale)
        for r in refs:
            if mode == "ours":
                pos = torch.from_numpy(align(ref_emb[r]).astype(np.float32)).to(device, dtype)[None, None]
                kw = {"ip_adapter_image_embeds": [torch.cat([torch.zeros_like(pos), pos])]}
            else:
                kw = {"ip_adapter_image": Image.open(r).convert("RGB")}
            for pi, prompt in enumerate(PROMPTS):
                for seed in range(args.seeds):
                    img = pipe(prompt, negative_prompt="blurry, lowres, deformed face",
                               num_inference_steps=args.steps, guidance_scale=args.guidance,
                               height=512, width=512, generator=torch.Generator(device).manual_seed(seed),
                               **kw).images[0]
                    e, _ = aura.embed(np.asarray(img)[:, :, ::-1].copy())
                    score = cosine(e, ref_emb[r]) if e is not None else None
                    rows.append({"mode": mode, "ref": str(r), "prompt": pi, "seed": seed,
                                 "score": None if score is None else round(score, 4)})
                    if seed == 0:
                        images[(mode, r, pi)] = img
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for mode in args.modes:
        s = [r["score"] for r in rows if r["mode"] == mode and r["score"] is not None]
        n = sum(1 for r in rows if r["mode"] == mode)
        summary[mode] = {"mean_cosine": round(float(np.mean(s)), 4) if s else None,
                         "face_found": f"{len(s)}/{n}"}
    summary["_settings"] = {"adapter": args.adapter, "align": args.align, "align_src": args.align_src,
                            "align_tgt": args.align_tgt, "scale": args.scale}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for pi in range(len(PROMPTS)):
        t = 192
        sheet = Image.new("RGB", (t * (1 + len(args.modes)), t * len(refs)), "white")
        for ri, r in enumerate(refs):
            sheet.paste(Image.open(r).convert("RGB").resize((t, t)), (0, ri * t))
            for mi, mode in enumerate(args.modes):
                sheet.paste(images[(mode, r, pi)].resize((t, t)), ((mi + 1) * t, ri * t))
        sheet.save(out / f"sheet_prompt{pi}.jpg", quality=90)
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
