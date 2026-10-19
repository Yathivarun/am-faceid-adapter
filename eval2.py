"""Evaluation v2 (plan 2026-10, section 2.3): likeness on the real-people set,
with an independent scorer, per-person confidence intervals, adapter fusion,
and results written as they are produced (a crash loses nothing; a rerun
skips what is done).

    python eval2.py --config configs/poc.yaml --paths-from configs/gpu_plus.yaml \
        --evalset ~/WPU/testings/3-faces/evalset.json --split dev --protocol reg2fru \
        --adapter checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors \
        --modes ours full-face plus-face "ours+full-face@0.6,0.3" --baseline full-face \
        --out checkpoints/ev2/dev_reg2fru_poc50k

Modes: `ours`, `full-face`, `plus-face`, each optionally `@scale` (default
--scale), and fusions `ours+full-face@s_ours,s_h94` / `ours+plus-face@...`
(both adapters loaded together, diffusers multi-IP-Adapter). h94 inputs are a
square face crop (--crop-scale x the detected box, default 2.0); ours gets the
input's AuraFace vector. CLIP-image adapters copy the crop's framing (2.0 shows
hair and shoulders, so full-face draws a small face on a torso), hence the knob;
a non-default crop is part of the mode name ("full-face@0.8~crop1.4").

Protocols (from prep/build_evalset.py):
    reg2fru   input = registration photo; targets = the person's FRU images
    fru2fru   input = each FRU image;     targets = that person's OTHER FRU images
Per generation and scorer (AuraFace; FaceNet-VGGFace2 as the independent one):
    own      mean cosine to the person's targets (never the input itself)
    input    cosine to the input image (how much is copied)
    impostor mean cosine to the next person's targets (chance level)
    margin   own - impostor
Summary: per mode x prompt x scorer, means over people with a 95% bootstrap CI
(people resampled), and against --baseline the per-person paired difference
with its CI and a sign-flip permutation p-value.

--tag NAME labels modes that use our adapter as "ours[NAME]" (e.g. the step),
so several checkpoints can share one --out and be compared person by person:
    ... --adapter .../step-0075000/... --modes ours --tag 75k --baseline "ours[50k]"

Outputs in --out: rows.jsonl (one line per generation), summary.json,
settings.json, img/<mode>/ (seed-0 images of the first --keep-images inputs).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faceid.aura import Aura  # noqa: E402
from train import build_models, load_config  # noqa: E402

H94 = {"full-face": "ip-adapter-full-face_sd15.safetensors", "plus-face": "ip-adapter-plus-face_sd15.safetensors"}
PROMPTS = {"photo": "a photo of a person, high quality",
           "sketch": "a pencil sketch portrait of a person, monochrome"}
NEGATIVE = "blurry, lowres, deformed face"


def parse_mode(spec, default_scale):
    """'ours+full-face@0.6,0.3' -> (base 'ours+full-face', parts ['ours','full-face'], scales [0.6, 0.3])."""
    base, _, sc = spec.partition("@")
    parts = base.split("+")
    for p in parts:
        if p not in ("ours", *H94):
            raise SystemExit(f"unknown mode part {p!r} in {spec!r}")
    if len(parts) > 1 and parts[0] != "ours":
        raise SystemExit(f"fusion modes start with 'ours': {spec!r}")
    scales = [float(s) for s in sc.split(",")] if sc else [default_scale] * len(parts)
    if len(scales) != len(parts):
        raise SystemExit(f"{spec!r}: {len(parts)} adapters but {len(scales)} scales")
    return base, parts, scales


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def square_crop(img: Image.Image, box, scale=2.0) -> Image.Image:
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale / 2
    w, h = img.size                       # clipped at the image border (the CLIP processor resizes anyway)
    return img.crop((int(max(0, cx - s)), int(max(0, cy - s)), int(min(w, cx + s)), int(min(h, cy + s))))


class FaceNet:
    """Independent identity scorer (InceptionResnetV1, VGGFace2; facenet-pytorch).
    Evaluation only. Input: a face box from AuraFace detection, 10% margin, 160 px."""

    def __init__(self, device):
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, rgb: np.ndarray, box):
        x0, y0, x1, y1 = box
        m = 0.1 * max(x1 - x0, y1 - y0)
        h, w = rgb.shape[:2]
        crop = rgb[int(max(0, y0 - m)):int(min(h, y1 + m)), int(max(0, x0 - m)):int(min(w, x1 + m))]
        if crop.size == 0:
            return None
        x = cv2.resize(crop, (160, 160)).astype(np.float32)
        x = torch.from_numpy((x - 127.5) / 128.0).permute(2, 0, 1)[None].to(self.device)
        return _unit(self.net(x)[0].float().cpu().numpy())


def cases(evalset, split, protocol, limit=None):
    """[(person, input path, input record, target paths, impostor target paths)]."""
    people = {i: p for i, p in evalset["people"].items() if p["split"] == split and p["protocols"][protocol]}
    ids = sorted(people)[:limit]
    recs = {r["path"]: r for p in people.values() for k in ("registration", "fru") for r in p[k]}
    out = []
    for n, i in enumerate(ids):
        imp = people[ids[(n + 1) % len(ids)]]["usable"]["fru"] if len(ids) > 1 else []
        fru = people[i]["usable"]["fru"]
        if protocol == "reg2fru":
            inp = people[i]["usable"]["registration"][0]
            out.append((i, inp, recs[inp], fru, imp))
        else:
            for inp in fru:
                out.append((i, inp, recs[inp], [f for f in fru if f != inp], imp))
    return out


def build_pipe(cfg, parts, adapter, dtype, device):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    tok, text, vae, unet, _ = build_models(cfg["paths"]["sd15"], dtype, device)
    sched = DPMSolverMultistepScheduler.from_pretrained(cfg["paths"]["sd15"], subfolder="scheduler",
                                                        local_files_only=True)
    enc = fe = None
    if any(p != "ours" for p in parts):
        enc = CLIPVisionModelWithProjection.from_pretrained(cfg["paths"]["image_encoder"], torch_dtype=dtype,
                                                            local_files_only=True).to(device)
        fe = CLIPImageProcessor()
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tok, unet=unet, scheduler=sched,
                                   safety_checker=None, feature_extractor=fe, image_encoder=enc,
                                   requires_safety_checker=False)
    dirs, names = [], []
    for p in parts:
        if p == "ours":
            dirs.append(str(Path(adapter).parent))
            names.append(Path(adapter).name)
        else:
            dirs.append(cfg["paths"]["ip_adapter_dir"])
            names.append(H94[p])
    pipe.load_ip_adapter(dirs if len(parts) > 1 else dirs[0], subfolder=[""] * len(parts) if len(parts) > 1 else "",
                         weight_name=names if len(parts) > 1 else names[0], image_encoder_folder=None,
                         local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def condition(pipe, parts, vec, crop, device, dtype):
    """Pipeline kwargs for one input."""
    def ours():
        pos = torch.from_numpy(vec.astype(np.float32)).to(device, dtype)[None, None]
        return torch.cat([torch.zeros_like(pos), pos])                      # (2, 1, 512): [negative, positive]
    if parts == ["ours"]:
        return {"ip_adapter_image_embeds": [ours()]}
    if len(parts) == 1:
        return {"ip_adapter_image": crop}
    pos, neg = pipe.encode_image(crop, device, 1, output_hidden_states=True)  # (1, 257, 1280) each
    h94 = torch.cat([neg, pos])[:, None]                                     # (2, 1, 257, 1280)
    return {"ip_adapter_image_embeds": [ours(), h94]}


# ---------------------------------------------------------------- statistics
def _boot_ci(x, rng, n=2000):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return [None, None]
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _signflip_p(d, rng, n=10000):
    d = np.asarray(d, np.float64)
    if len(d) < 2 or not np.any(d):
        return None
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], (n, len(d)))
    return round(float(((np.abs((flips * d).mean(1)) >= obs).sum() + 1) / (n + 1)), 5)


def summarize(rows, baseline=None, seed=0):
    rng = np.random.default_rng(seed)
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    scorers = sorted({k.split(":")[0] for r in rows for k in r if ":" in k})
    out, per_person = {}, {}
    for mode in modes:
        out[mode] = {"generations": sum(r["mode"] == mode for r in rows)}
        for prompt in PROMPTS:
            mine = [r for r in rows if r["mode"] == mode and r["prompt"] == prompt]
            for sc in scorers:
                found = [r for r in mine if r.get(f"{sc}:own") is not None]
                by = {}
                for r in found:
                    by.setdefault(r["person"], []).append(r)
                if not by:
                    continue
                def avg(rs, k):
                    vals = [r[f"{sc}:{k}"] for r in rs if r.get(f"{sc}:{k}") is not None]
                    return float(np.mean(vals)) if vals else float("nan")
                pp = {p: {k: avg(rs, k) for k in ("own", "input", "impostor")} for p, rs in by.items()}
                for v in pp.values():
                    v["margin"] = v["own"] - v["impostor"]
                pp = {p: v for p, v in pp.items() if not np.isnan(v["margin"])}
                if not pp:
                    continue
                per_person[(mode, prompt, sc)] = pp
                ent = {"people": len(pp), "face_found": f"{len(found)}/{len(mine)}"}
                for k in ("own", "input", "impostor", "margin"):
                    vals = [v[k] for v in pp.values() if not np.isnan(v[k])]
                    ent[k] = round(float(np.mean(vals)), 4) if vals else None
                    if k == "margin":
                        ent["margin_ci95"] = _boot_ci(vals, rng)
                if ent["input"] is not None:
                    ent["copy_gap"] = round(ent["input"] - ent["own"], 4)   # input-likeness beyond identity
                out[mode].setdefault(prompt, {})[sc] = ent
    if baseline and baseline in modes:
        for mode in modes:
            if mode == baseline:
                continue
            for prompt in PROMPTS:
                for sc in scorers:
                    a, b = per_person.get((mode, prompt, sc)), per_person.get((baseline, prompt, sc))
                    if not a or not b:
                        continue
                    common = sorted(set(a) & set(b))
                    d = [a[p]["margin"] - b[p]["margin"] for p in common]
                    out[mode][prompt][sc][f"vs_{baseline}"] = {
                        "people": len(common), "margin_diff": round(float(np.mean(d)), 4),
                        "ci95": _boot_ci(d, rng), "p_signflip": _signflip_p(d, rng),
                        "wins": f"{sum(x > 0 for x in d)}/{len(d)}"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--paths-from", help="model paths from this config (e.g. configs/gpu_plus.yaml)")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--protocol", default="reg2fru", choices=["reg2fru", "fru2fru"])
    ap.add_argument("--adapter", help="our adapter .safetensors (modes containing 'ours')")
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--baseline", help="mode to compare every other mode against (paired, per person)")
    ap.add_argument("--tag", help="label for our adapter in mode names, e.g. 75k -> 'ours[75k]'")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--crop-scale", type=float, default=2.0,
                    help="h94 input crop = this x the detected face box (framing of the output)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--scorers", nargs="+", default=["aura", "facenet"], choices=["aura", "facenet"])
    ap.add_argument("--aura-gpu", action="store_true", help="AuraFace on CUDA (fails loudly on CPU fallback)")
    ap.add_argument("--limit", type=int, help="first N people only (smoke test)")
    ap.add_argument("--keep-images", type=int, default=12, help="save seed-0 images of the first N inputs per mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line] if rows_path.exists() else []

    def write_summary():
        (out / "summary.json").write_text(json.dumps(summarize(rows, args.baseline), indent=2))

    if args.summarize_only:
        write_summary()
        print((out / "summary.json").read_text())
        return

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available (check nvidia-smi); --allow-cpu only for a smoke test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cfg = load_config(args.config, {}, {}, args.paths_from)
    specs = [parse_mode(m, args.scale) for m in args.modes]
    names = [m.replace("ours", f"ours[{args.tag}]", 1) if (args.tag and "ours" in m) else m for m in args.modes]
    if args.crop_scale != 2.0:
        names = [n + f"~crop{args.crop_scale:g}" if any(p != "ours" for p in sp[1]) else n
                 for n, sp in zip(names, specs)]
    if any("ours" in parts for _, parts, _ in specs) and not args.adapter:
        raise SystemExit("--adapter is required for modes with 'ours'")

    evalset_path = Path(args.evalset).expanduser()
    evalset = json.loads(evalset_path.read_text())
    npz = np.load(evalset_path.with_suffix(".npz"))
    aura_vec = {str(p): v for p, v in zip(npz["paths"], npz["vecs"])}
    todo = cases(evalset, args.split, args.protocol, args.limit)
    if not todo:
        raise SystemExit(f"no people in split {args.split!r} usable for {args.protocol}")
    aura = Aura(cfg["paths"]["aura"], gpu=args.aura_gpu)
    facenet = FaceNet(device) if "facenet" in args.scorers else None

    # Reference vectors of every image that is scored against (targets, impostors, inputs)
    recs = {r["path"]: r for p in evalset["people"].values() for k in ("registration", "fru") for r in p[k]}
    fn_vec = {}
    if facenet:
        need = {p for _, inp, _, tg, imp in todo for p in (inp, *tg, *imp)}
        for p in sorted(need):
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            v = facenet.embed(rgb, recs[p]["box"])
            if v is not None:
                fn_vec[p] = v
    refs = {"aura": aura_vec, "facenet": fn_vec}

    settings = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv,
                "evalset": str(evalset_path), "evalset_sha256": hashlib.sha256(evalset_path.read_bytes()).hexdigest()[:16],
                "split": args.split, "protocol": args.protocol, "people": len({c[0] for c in todo}),
                "inputs": len(todo), "adapter": args.adapter, "modes": args.modes, "seeds": args.seeds,
                "steps": args.steps, "guidance": args.guidance, "prompts": PROMPTS, "negative": NEGATIVE,
                "scorers": args.scorers, "h94_input": f"square face crop, {args.crop_scale:g}x detected box"}
    settings["tag"] = args.tag
    (out / f"settings{'-' + args.tag if args.tag else ''}.json").write_text(json.dumps(settings, indent=2))
    done = {(r["mode"], r["input"], r["prompt"], r["seed"]) for r in rows}
    print(f"{args.split}/{args.protocol}: {settings['people']} people, {len(todo)} inputs, "
          f"{len(args.modes)} modes; {len(done)} generations already done", flush=True)

    pipe, loaded = None, None
    with rows_path.open("a") as fout:
        for spec, (base, parts, scales) in zip(names, specs):
            left = [(c, pr, s) for c in todo for pr in PROMPTS for s in range(args.seeds)
                    if (spec, c[1], pr, s) not in done]
            if not left:
                print(f"[{spec}] done already")
                continue
            if loaded != base:
                pipe = None
                if device == "cuda":
                    torch.cuda.empty_cache()
                pipe, loaded = build_pipe(cfg, parts, args.adapter, dtype, device), base
            pipe.set_ip_adapter_scale(scales if len(scales) > 1 else scales[0])
            keep = {c[1] for c in todo[: args.keep_images]}
            img_dir = out / "img" / spec.replace("@", "_at_").replace(",", "_")
            t0, n = time.time(), 0
            cond_cache = {}
            for (person, inp, rec, targets, imp), prompt, seed in left:
                if inp not in cond_cache:
                    crop = square_crop(Image.open(inp).convert("RGB"), rec["box"], args.crop_scale)
                    cond_cache = {inp: condition(pipe, parts, aura_vec[inp], crop, device, dtype)}
                img = pipe(PROMPTS[prompt], negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                           guidance_scale=args.guidance, height=512, width=512,
                           generator=torch.Generator(device).manual_seed(seed), **cond_cache[inp]).images[0]
                rgb = np.asarray(img)
                row = {"mode": spec, "person": person, "input": inp, "prompt": prompt, "seed": seed}
                e, face = aura.embed(rgb[:, :, ::-1].copy())
                gen = {"aura": e, "facenet": facenet.embed(rgb, face.bbox) if (facenet and face is not None) else None}
                for sc in args.scorers:
                    g = gen[sc]
                    if g is None:
                        row[f"{sc}:own"] = None
                        continue
                    tg = [refs[sc][p] for p in targets if p in refs[sc]]
                    im = [refs[sc][p] for p in imp if p in refs[sc]]
                    row[f"{sc}:own"] = round(float(np.mean([g @ t for t in tg])), 4) if tg else None
                    row[f"{sc}:input"] = round(float(g @ refs[sc][inp]), 4) if inp in refs[sc] else None
                    row[f"{sc}:impostor"] = round(float(np.mean([g @ t for t in im])), 4) if im else None
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                rows.append(row)
                n += 1
                if seed == 0 and inp in keep:
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img.save(img_dir / f"{person[:8]}_{Path(inp).stem[:24]}_{prompt}.jpg", quality=90)
                if n % 50 == 0:
                    print(f"[{spec}] {n}/{len(left)}  {(time.time() - t0) / n:.2f} s/image", flush=True)
            print(f"[{spec}] finished {n} generations, {(time.time() - t0) / max(n, 1):.2f} s/image", flush=True)
            write_summary()
    write_summary()
    print((out / "summary.json").read_text())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
