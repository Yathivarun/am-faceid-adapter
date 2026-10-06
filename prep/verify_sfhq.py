"""Verify the SFHQ-T2I Kaggle zip before building from it.

    python prep/verify_sfhq.py --zip data/raw/sfhq_t2i/download/archive.zip \
        --report data/raw/sfhq_t2i/verify_report.json [--sample 300] [--no-crc]

Checks: every zip entry's CRC; the CSV <-> images/ match; per-model counts
against the published numbers; a random sample decoded (size, mode); and an
AuraFace pass on a sample of the licence-clean subset (FLUX1_schnell + SDXL),
reporting how many would pass build_dataset.py's one-clear-face filter.
"""

import argparse
import collections
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PUBLISHED = {"FLUX1_schnell": 58034, "SDXL": 53087, "FLUX1_dev": 7273, "FLUX1_pro": 3209, "DALLE3": 1123}
COMMERCIAL = ("FLUX1_schnell", "SDXL")   # Apache-2.0 / OpenRAIL++ outputs (see data/raw/sfhq_t2i/SOURCE.md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sample", type=int, default=300, help="images decoded / run through AuraFace")
    ap.add_argument("--config", default="configs/poc.yaml", help="source of paths.aura")
    ap.add_argument("--no-crc", action="store_true")
    args = ap.parse_args()
    rep = {"zip": args.zip, "zip_bytes": Path(args.zip).stat().st_size}

    z = zipfile.ZipFile(args.zip)
    infos = z.infolist()
    images = {PurePosixPath(i.filename).name: i.filename for i in infos
              if i.filename.startswith("images/images/") and i.filename.lower().endswith(".jpg")}
    csv_name = next(i.filename for i in infos if i.filename.endswith("SFHQ_T2I_dataset.csv"))
    rows = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(z.read(csv_name)), encoding="utf-8")))
    names = [r["image_filename"] for r in rows]
    per_model = collections.Counter(r["model_used"] for r in rows)
    rep["entries"] = len(infos)
    rep["images"] = len(images)
    rep["csv_rows"] = len(rows)
    rep["csv_duplicates"] = len(names) - len(set(names))
    rep["csv_missing_image"] = sorted(set(names) - set(images))[:20]
    rep["images_not_in_csv"] = sorted(set(images) - set(names))[:20]
    rep["per_model"] = dict(per_model)
    rep["per_model_matches_published"] = dict(per_model) == PUBLISHED
    rep["commercial_subset"] = sum(per_model[m] for m in COMMERCIAL)
    rep["empty_prompts"] = sum(1 for r in rows if not r["text_prompt"].strip())
    print(json.dumps({k: v for k, v in rep.items()}, indent=1), flush=True)

    rng = random.Random(0)
    clean = [r for r in rows if r["model_used"] in COMMERCIAL]
    sample = rng.sample(clean, min(args.sample, len(clean)))
    sizes, modes, bad = collections.Counter(), collections.Counter(), []
    decoded = []
    for r in sample:
        try:
            img = Image.open(io.BytesIO(z.read(images[r["image_filename"]])))
            img.load()
            sizes[img.size] += 1
            modes[img.mode] += 1
            decoded.append((r, img.convert("RGB")))
        except Exception as e:  # noqa: BLE001
            bad.append(f"{r['image_filename']}: {e}")
    rep["sample_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.items()}
    rep["sample_modes"] = dict(modes)
    rep["sample_decode_errors"] = bad

    # AuraFace: same filter as build_dataset.py (512 px, det >= 0.6, face side >= 0.15 of image, exactly one)
    import yaml
    from faceid.aura import Aura
    aura_path = Path(yaml.safe_load((ROOT / args.config).read_text())["paths"]["aura"])
    aura = Aura(aura_path if aura_path.is_absolute() else (ROOT / aura_path).resolve())
    keep = collections.Counter()
    why = collections.Counter()
    for r, img in decoded:
        small = img.resize((512, 512), Image.LANCZOS)
        faces = aura.faces(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        clear = [f for f in faces if f.det_score >= 0.6 and (f.bbox[2] - f.bbox[0]) >= 0.15 * 512]
        ok = len(clear) == 1
        keep[(r["model_used"], ok)] += 1
        if not ok:
            why["no face" if not faces else ("small/low-score" if not clear else "several faces")] += 1
    rep["aura_sample_keep"] = {m: f"{keep[(m, True)]}/{keep[(m, True)] + keep[(m, False)]}" for m in COMMERCIAL}
    rep["aura_reject_reasons"] = dict(why)
    print(json.dumps({k: rep[k] for k in ("sample_sizes", "sample_modes", "sample_decode_errors",
                                          "aura_sample_keep", "aura_reject_reasons")}, indent=1), flush=True)

    if not args.no_crc:
        t = time.time()
        bad_crc = []
        for n, i in enumerate(infos, 1):
            if i.is_dir():
                continue
            try:
                with z.open(i) as f:
                    while f.read(1 << 20):
                        pass
            except Exception as e:  # noqa: BLE001 - BadZipFile on CRC mismatch
                bad_crc.append(f"{i.filename}: {e}")
            if n % 25000 == 0:
                print(f"crc {n}/{len(infos)} ({time.time() - t:.0f}s, {len(bad_crc)} bad)", flush=True)
        rep["crc_checked"] = len(infos)
        rep["crc_errors"] = bad_crc[:50]
        rep["crc_error_count"] = len(bad_crc)

    Path(args.report).write_text(json.dumps(rep, indent=2))
    ok = (rep["csv_rows"] == rep["images"] == 122726 and rep["per_model_matches_published"]
          and not rep["csv_missing_image"] and not bad and not rep.get("crc_error_count"))
    print(("VERIFIED" if ok else "PROBLEMS FOUND") + f" -> {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
