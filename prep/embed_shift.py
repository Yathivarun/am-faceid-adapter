"""Where do real visitors' AuraFace vectors sit relative to the training data?

    python prep/embed_shift.py --data data/faces_sfhq --data data/faces \
        --refs ../fullbody-lab/dev/face_inpaint/mul_face --config configs/gpu.yaml

The commercial adapter matched the research one on SFHQ's own held-out faces
but fell from 0.53 to 0.31 on real visitors: the classic synthetic-to-real gap.
This measures it, read-only, on the vectors the adapter actually receives:

  spread      how tightly a set clusters: |mean vector| and mean pairwise cosine
  centres     cosine between the sets' mean directions
  domain      nearest-centre classification accuracy between two sets (0.5 =
              indistinguishable, ~1.0 = two separate clouds), raw and after
              centring each set on its own mean
  reach       for every reference face: cosine to its nearest training vectors
              (top-1 / mean top-10) in each set, and its projection on each
              set's mean direction ("does it look like this set at all?")

A large raw domain accuracy that drops to ~0.5 after centring means the gap is
mostly a shared offset, and mean-centring the adapter input would remove it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_set(root: Path, split=None, limit=20000, rng=None):
    lines = [json.loads(line) for line in (root / "meta.jsonl").read_text().splitlines() if line]
    emb = np.load(root / "embeds.npy").astype(np.float32)
    exclude = set((root / "exclude.txt").read_text().split()) if (root / "exclude.txt").exists() else set()
    idx = [i for i, m in enumerate(lines) if m["id"] not in exclude and (split is None or m["split"] == split)]
    if rng is not None and len(idx) > limit:
        idx = sorted(rng.choice(idx, size=limit, replace=False))
    return unit(emb[idx])


def embed_refs(folders, aura_path):
    from faceid.aura import Aura
    aura = Aura(aura_path)
    names, vecs = [], []
    for folder in folders:
        for p in sorted(Path(folder).rglob("*")):
            if p.suffix.lower() in _EXT:
                e, _ = aura.embed_path(p)
                if e is not None:
                    names.append(str(p.relative_to(folder)))
                    vecs.append(e)
    return names, unit(np.stack(vecs)) if vecs else np.zeros((0, 512), np.float32)


def spread(x, rng, n=3000):
    s = x[rng.choice(len(x), size=min(n, len(x)), replace=False)]
    sim = s @ s.T
    return {"n": int(len(x)), "mean_vec_norm": round(float(np.linalg.norm(x.mean(0))), 4),
            "mean_pairwise_cos": round(float(sim[~np.eye(len(s), dtype=bool)].mean()), 4)}


def domain_acc(a, b, rng, center=False):
    """Nearest-centre accuracy, trained on 70% and scored on 30% of each set."""
    def halves(x):
        p = rng.permutation(len(x))
        k = int(0.7 * len(x))
        return x[p[:k]], x[p[k:]]
    if center:
        a, b = unit(a - a.mean(0)), unit(b - b.mean(0))
    (a_tr, a_te), (b_tr, b_te) = halves(a), halves(b)
    ca, cb = unit(a_tr.mean(0)), unit(b_tr.mean(0))
    hits = np.concatenate([(a_te @ ca) > (a_te @ cb), (b_te @ cb) > (b_te @ ca)])
    return round(float(hits.mean()), 4)


def reach(refs, x, skip_self=False):
    sim = refs @ x.T
    top = np.sort(sim, axis=1)[:, ::-1]
    if skip_self:                                  # refs are members of x: drop the self-match
        top = top[:, 1:]
    return top[:, 0], top[:, :10].mean(1), refs @ unit(x.mean(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="built dataset folder (repeatable)")
    ap.add_argument("--refs", action="append", default=[], help="folder of real reference photos (repeatable)")
    ap.add_argument("--config", default="configs/poc.yaml", help="source of paths.aura")
    ap.add_argument("--aura", help="AuraFace-v1 pack (overrides the config)")
    ap.add_argument("--limit", type=int, default=20000, help="vectors sampled per dataset")
    ap.add_argument("--out", help="write the report as JSON here")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    sets = {Path(d).name: load_set(Path(d), limit=args.limit, rng=rng) for d in args.data}
    report = {"spread": {k: spread(v, rng) for k, v in sets.items()}}
    keys = list(sets)
    report["centres_cosine"] = {f"{a} vs {b}": round(float(unit(sets[a].mean(0)) @ unit(sets[b].mean(0))), 4)
                                for i, a in enumerate(keys) for b in keys[i + 1:]}
    report["domain_accuracy"] = {f"{a} vs {b}": {"raw": domain_acc(sets[a], sets[b], rng),
                                                 "centred": domain_acc(sets[a], sets[b], rng, center=True)}
                                 for i, a in enumerate(keys) for b in keys[i + 1:]}

    if args.refs:
        if not args.aura:
            import yaml
            p = Path(yaml.safe_load((ROOT / args.config).read_text())["paths"]["aura"])
            args.aura = str(p if p.is_absolute() else (ROOT / p).resolve())
        names, refs = embed_refs(args.refs, args.aura)
        report["refs"] = {"n": len(names), "spread": spread(refs, rng, n=len(refs)) if len(refs) > 1 else None}
        per = {}
        for k, x in sets.items():
            t1, t10, proj = reach(refs, x)
            members = x[rng.choice(len(x), size=min(200, len(x)), replace=False)]
            own_t1, own_t10, own_proj = reach(members, x, skip_self=True)
            per[k] = {"refs_top1": round(float(t1.mean()), 4), "refs_top10": round(float(t10.mean()), 4),
                      "refs_on_mean_dir": round(float(proj.mean()), 4),
                      # the same for the set's own members: what "in distribution" looks like
                      "own_top1": round(float(own_t1.mean()), 4),
                      "own_top10": round(float(own_t10.mean()), 4),
                      "own_on_mean_dir": round(float(own_proj.mean()), 4)}
        report["reach"] = per

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
