"""Identity diversity of a built dataset, and an optional near-duplicate filter.

    python prep/diversity.py data/faces_sfhq data/faces            # compare datasets
    python prep/diversity.py data/faces_sfhq --dedupe 0.6 --write-exclude

Synthetic faces can cluster ("same-face syndrome": a generator drawing similar
people over and over). Two numbers show it, on a random sample of AuraFace
vectors per dataset:
  - pairwise cosine between different images (lower mean = more distinct identities)
  - the share of images that have a near-twin above --dedupe (likely the same person)

--write-exclude goes over the WHOLE dataset (chunked, CPU) and writes
<data>/exclude.txt: every image whose identity is within --dedupe of an earlier
one. FaceDataset skips those ids in training. The images stay on disk.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(root: Path):
    ids = [json.loads(line)["id"] for line in (root / "meta.jsonl").read_text().splitlines() if line]
    emb = np.load(root / "embeds.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return ids, emb


def stats(emb: np.ndarray, sample: int, dedupe: float, rng) -> dict:
    idx = rng.choice(len(emb), size=min(sample, len(emb)), replace=False)
    s = emb[idx]
    sim = s @ s.T
    off = sim[~np.eye(len(s), dtype=bool)]
    np.fill_diagonal(sim, -1.0)
    twins = (sim.max(axis=1) >= dedupe).mean()
    return {"images": int(len(emb)), "sample": int(len(s)), "mean_cosine": round(float(off.mean()), 4),
            "p50": round(float(np.percentile(off, 50)), 4), "p95": round(float(np.percentile(off, 95)), 4),
            "p99": round(float(np.percentile(off, 99)), 4),
            f"share_with_twin>={dedupe}": round(float(twins), 4)}


def near_duplicates(emb: np.ndarray, threshold: float, chunk: int = 2048) -> np.ndarray:
    """Indices whose vector is within `threshold` of any EARLIER vector, so the
    first image of each cluster is kept. ~1 GB of RAM per chunk at 111k images."""
    drop = np.zeros(len(emb), dtype=bool)
    for start in range(0, len(emb), chunk):
        block = emb[start:start + chunk]
        sim = block @ emb[: start + len(block)].T                    # vs everything up to this block
        rows = start + np.arange(len(block))
        sim[np.arange(sim.shape[1])[None, :] >= rows[:, None]] = -1.0  # only earlier images count
        drop[start:start + len(block)] = sim.max(axis=1) >= threshold
    return np.nonzero(drop)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="built dataset folders (meta.jsonl + embeds.npy)")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--dedupe", type=float, default=0.6, help="cosine above which two images count as one person")
    ap.add_argument("--write-exclude", action="store_true", help="write <data>/exclude.txt for the FIRST dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    report = {}
    for d in map(Path, args.data):
        ids, emb = load(d)
        report[str(d)] = stats(emb, args.sample, args.dedupe, rng)
        print(f"{d}: {report[str(d)]}", flush=True)

    if args.write_exclude:
        d = Path(args.data[0])
        ids, emb = load(d)
        drop = near_duplicates(emb, args.dedupe)
        (d / "exclude.txt").write_text("".join(f"{ids[i]}\n" for i in drop))
        report[str(d)]["excluded"] = int(len(drop))
        print(f"{d}: {len(drop)} near-duplicates (cosine >= {args.dedupe}) -> {d / 'exclude.txt'}")
    out = Path(args.data[0]) / "diversity.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
