"""Side-by-side sheets for visual checks: each person's registration photo and
FRU captures next to what each mode generated (eval2.py's img/ folders).

    python prep/compare_sheet.py --evalset ~/WPU/testings/3-faces/evalset.json \
        --run checkpoints/ev2/dev_reg2fru --modes full-face "ours[105k]" "ours[50k]+full-face@0.8,0.5" \
        --out compare_reg2fru

One JPEG per prompt (photo, sketch). Row = person; columns = registration |
FRU 1-3 | each mode (seed 0). Only people with saved images appear (eval2
keeps the first --keep-images inputs). Small enough to copy to a laptop.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

TILE = 256


def tile(path_or_img):
    img = Image.open(path_or_img) if not isinstance(path_or_img, Image.Image) else path_or_img
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.pad(img, (TILE, TILE), color="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--run", required=True, help="eval2 --out folder (has img/<mode>/)")
    ap.add_argument("--modes", nargs="+", required=True, help="mode names as in rows.jsonl / summary.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    people = json.loads(Path(args.evalset).expanduser().read_text())["people"]
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {m: run / "img" / m.replace("@", "_at_").replace(",", "_") for m in args.modes}
    for m, d in dirs.items():
        if not d.is_dir():
            raise SystemExit(f"no images for mode {m!r} ({d})")
    for prompt in ("photo", "sketch"):
        rows = []
        for pid in sorted(people):
            p = people[pid]
            gens = []
            for m in args.modes:
                hits = sorted(dirs[m].glob(f"{pid[:8]}_*_{prompt}.jpg"))
                gens.append(hits[0] if hits else None)
            if not any(gens):
                continue
            reg = p["usable"]["registration"][:1]
            fru = p["usable"]["fru"][:3]
            rows.append((pid, reg + [None] * (1 - len(reg)), fru + [None] * (3 - len(fru)), gens))
        if not rows:
            print(f"{prompt}: nothing to show")
            continue
        heads = ["registration", "FRU 1", "FRU 2", "FRU 3", *args.modes]
        sheet = Image.new("RGB", (TILE * len(heads), 24 + TILE * len(rows)), "white")
        d = ImageDraw.Draw(sheet)
        for c, h in enumerate(heads):
            d.text((c * TILE + 4, 6), h[:40], fill="black")
        for r, (pid, reg, fru, gens) in enumerate(rows):
            for c, item in enumerate(reg + fru + gens):
                if item is not None:
                    sheet.paste(tile(item), (c * TILE, 24 + r * TILE))
            d.text((4, 24 + r * TILE + 4), pid[:8], fill="red")
        path = out / f"compare_{prompt}.jpg"
        sheet.save(path, quality=88)
        print(f"{prompt}: {len(rows)} people -> {path}")


if __name__ == "__main__":
    main()
