"""Build the real-people evaluation set (evaluation only, never training).

    python prep/build_evalset.py --root ~/WPU/testings/3-faces/fru-images \
        --dev ~/WPU/testings/3-faces/dev60.txt --aura <AuraFace-v1 pack> --out ~/WPU/testings/3-faces/evalset.json

Layout: <root>/<person id>/{registration,fru,sau}/ images (jpg/png/bmp; videos
are skipped). For every registration and FRU image: AuraFace detection (face
count, detection score, box), the unit identity vector, and a sharpness score
(variance of the Laplacian on the face crop). Flags per image:
    no_face        nothing detected
    multi_face     a second face at least 30% the area of the largest
    low_det        detection score < 0.6
    blurry         face-crop sharpness < 40
Per person: `reg_fru_cos` (registration vs mean FRU vector) and the usable
images (no flags). People whose own images disagree (reg_fru_cos < 0.25: wrong
photo filed under the id) are dropped. Ids that look like the same person
(mean-vector cosine >= 0.6: registered more than once) are grouped; each group
keeps ONE id (a dev id if the group has one, else the id with the most usable
images) and the others are marked duplicate_of it, so no impostor is secretly
the same person and nobody is lost.

Splits: the ids in --dev (the dev set already used for the 2026-10-17 baseline)
are "dev"; every other usable person is "test". Protocols in eval2.py:
    reg2fru   input = the registration photo, scored against the FRU images
    fru2fru   input = each FRU image, scored against the person's other FRU images
Vectors are written next to the JSON (<out>.npz, keyed by image path) so the
evaluation doesn't recompute them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from faceid.aura import Aura  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_DET, MIN_SHARP, MULTI_AREA = 0.6, 40.0, 0.3
MISMATCH_COS, DUPLICATE_COS = 0.25, 0.6


def sharpness(bgr, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = bgr.shape[:2]
    crop = bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def inspect(aura, path):
    """(record, vector or None) for one image."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return {"path": str(path), "flags": ["unreadable"]}, None
    found = aura.faces(bgr)
    rec = {"path": str(path), "size": [bgr.shape[1], bgr.shape[0]], "n_faces": len(found), "flags": []}
    if not found:
        rec["flags"].append("no_face")
        return rec, None
    f = found[0]
    area = lambda g: (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1])  # noqa: E731
    rec["det"] = round(float(f.det_score), 3)
    rec["box"] = [round(float(v), 1) for v in f.bbox]
    rec["kps"] = [[round(float(a), 1), round(float(b), 1)] for a, b in f.kps] if f.kps is not None else None
    rec["sharp"] = round(sharpness(bgr, f.bbox), 1)
    if len(found) > 1 and area(found[1]) >= MULTI_AREA * area(f):
        rec["flags"].append("multi_face")
    if rec["det"] < MIN_DET:
        rec["flags"].append("low_det")
    if rec["sharp"] < MIN_SHARP:
        rec["flags"].append("blurry")
    return rec, f.normed_embedding.astype(np.float32)


def images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG) if folder.is_dir() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dev", required=True, help="text file of dev-set person ids")
    ap.add_argument("--aura", required=True, help="AuraFace-v1 pack folder")
    ap.add_argument("--out", required=True, help="evalset .json (vectors go to the same name .npz)")
    ap.add_argument("--gpu", action="store_true", help="AuraFace on CUDA (fails loudly if it falls back to CPU)")
    ap.add_argument("--limit", type=int, help="first N people only (smoke test)")
    args = ap.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
    dev = set(Path(args.dev).expanduser().read_text().split())
    aura = Aura(args.aura, gpu=args.gpu)
    people_dirs = sorted(p for p in root.iterdir() if p.is_dir())[: args.limit]
    people, vecs, t0 = {}, {}, time.time()
    for n, d in enumerate(people_dirs, 1):
        person = {"registration": [], "fru": [], "flags": []}
        for kind in ("registration", "fru"):
            for p in images(d / kind):
                rec, v = inspect(aura, p)
                person[kind].append(rec)
                if v is not None:
                    vecs[str(p)] = v
        good = {k: [r["path"] for r in person[k] if not r["flags"] and r["path"] in vecs]
                for k in ("registration", "fru")}
        person["usable"] = good
        fru_v = [vecs[p] for p in good["fru"]]
        if good["registration"] and fru_v:
            m = np.mean(fru_v, 0)
            person["reg_fru_cos"] = round(float(vecs[good["registration"][0]] @ (m / np.linalg.norm(m))), 4)
            if person["reg_fru_cos"] < MISMATCH_COS:
                person["flags"].append("mismatch")
        if len(good["fru"]) >= 2:
            fv = np.stack(fru_v)
            sims = fv @ fv.T
            person["fru_fru_cos"] = round(float(sims[np.triu_indices(len(fv), 1)].mean()), 4)
        person["protocols"] = {"reg2fru": bool(good["registration"] and good["fru"]),
                               "fru2fru": len(good["fru"]) >= 2}
        people[d.name] = person
        if n % 25 == 0:
            print(f"{n}/{len(people_dirs)} people  {time.time() - t0:.0f}s", flush=True)

    # Same person under two ids: mean identity vector per id, pairwise cosine
    ids = [i for i, p in people.items() if p["usable"]["fru"] or p["usable"]["registration"]]
    means = []
    for i in ids:
        v = np.mean([vecs[p] for p in people[i]["usable"]["registration"] + people[i]["usable"]["fru"]], 0)
        means.append(v / np.linalg.norm(v))
    dups, parent = [], {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(means) > 1:
        sims = np.stack(means) @ np.stack(means).T
        for a, b in zip(*np.where(np.triu(sims, 1) >= DUPLICATE_COS)):
            dups.append([ids[a], ids[b], round(float(sims[a, b]), 4)])
            parent[find(ids[a])] = find(ids[b])
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    groups = [sorted(g) for g in groups.values() if len(g) > 1]
    for g in groups:
        n_usable = lambda i: len(people[i]["usable"]["registration"]) + len(people[i]["usable"]["fru"])  # noqa: E731
        keep = max(g, key=lambda i: (i in dev and not people[i]["flags"], not people[i]["flags"], n_usable(i), i))
        for i in g:
            if i != keep:
                people[i]["flags"].append("duplicate")
                people[i]["duplicate_of"] = keep

    for i, p in people.items():
        ok = not p["flags"] and (p["protocols"]["reg2fru"] or p["protocols"]["fru2fru"])
        p["split"] = ("dev" if i in dev else "test") if ok else "excluded"

    img_flags = {}
    for p in people.values():
        for kind in ("registration", "fru"):
            for r in p[kind]:
                for f in r["flags"]:
                    img_flags[f] = img_flags.get(f, 0) + 1
    split = lambda s: [i for i, p in people.items() if p["split"] == s]  # noqa: E731
    stats = {
        "people": len(people),
        "dev": len(split("dev")), "test": len(split("test")), "excluded": len(split("excluded")),
        "dev_ids_missing": sorted(dev - set(people)),
        "dev_ids_excluded": sorted(i for i in dev if people.get(i, {}).get("split") == "excluded"),
        "reg2fru": {s: sum(people[i]["protocols"]["reg2fru"] for i in split(s)) for s in ("dev", "test")},
        "fru2fru": {s: sum(people[i]["protocols"]["fru2fru"] for i in split(s)) for s in ("dev", "test")},
        "image_flags": img_flags,
        "person_flags": {f: sum(f in p["flags"] for p in people.values())
                         for f in ("mismatch", "duplicate")},
        "duplicate_groups": {"groups": len(groups), "ids": sum(map(len, groups)),
                             "largest": max(map(len, groups)) if groups else 0},
        "duplicates": dups,
        "reg_fru_cos_mean": round(float(np.mean([p["reg_fru_cos"] for p in people.values()
                                                 if "reg_fru_cos" in p])), 4) if people else None,
        "fru_fru_cos_mean": round(float(np.mean([p["fru_fru_cos"] for p in people.values()
                                                 if "fru_fru_cos" in p])), 4) if people else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(root), "dev_list": str(args.dev),
           "thresholds": {"min_det": MIN_DET, "min_sharp": MIN_SHARP, "multi_area": MULTI_AREA,
                          "mismatch_cos": MISMATCH_COS, "duplicate_cos": DUPLICATE_COS},
           "stats": stats, "people": people}
    out.write_text(json.dumps(doc, indent=1))
    np.savez(out.with_suffix(".npz"), paths=np.array(list(vecs)), vecs=np.stack(list(vecs.values()))
             if vecs else np.zeros((0, 512), np.float32))
    print(json.dumps(stats, indent=2))
    print(f"-> {out} (+ {out.with_suffix('.npz').name})")


if __name__ == "__main__":
    main()
