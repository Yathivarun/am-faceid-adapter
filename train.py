"""Train the AuraFace identity adapter on a frozen SD1.5.

    accelerate launch train.py --config configs/poc.yaml      # or plain: python train.py --config ...
    python train.py --config configs/poc.yaml --max-steps 20 --batch-size 1 --no-val   # smoke test

Trains only the identity projection and the to_k_ip/to_v_ip layers
(~22M params). Every `save_every` steps: adapter weights (h94 FaceID format,
loadable with `pipe.load_ip_adapter`) + full resumable state. Every
`val_every` steps: held-out faces rendered from their embeddings, saved as a
sheet, and scored with AuraFace (mean cosine to the source face).

Continued runs (plan 2026-10, Stage A):
    python train.py --config configs/real_v3.yaml --paths-from configs/gpu.yaml \
        --init-adapter checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors --keep-state-at 75000
  --init-adapter starts from saved adapter WEIGHTS (no optimiser state survives,
  hence the LR warm-up); the step counter continues from the step in its folder
  name, so checkpoints read step-0052500, ... A cool-down branch resumes a kept
  state into its own folder:
    python train.py ... --resume checkpoints/real_v3/state-0075000 --out checkpoints/real_v3_cd75 \
        --decay-start 75000 --max-steps 80000

LR schedule (warm-up, stable, decay; "WSD"): linear warm-up over lr_warmup steps
from the step this process started at (fresh or --init-adapter runs only, not
--resume), constant lr, then a cosine decay to lr_min over lr_decay_steps from
lr_decay_start (unset = no decay). Every run writes run.json (config, command,
code and init-weight hashes) and metrics.jsonl (loss, lr, speed, validation).
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faceid.data import FaceDataset  # noqa: E402
from faceid.model import attach_adapter, id_embeds, load_adapter, save_adapter  # noqa: E402

WEIGHT_NAME = "faceid_aura_sd15.safetensors"


def _read_config(path: Path) -> dict:
    """A YAML config; `base: other.yaml` (relative to it) is loaded first and
    this file's sections are merged over it, key by key."""
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    if not base:
        return cfg
    merged = _read_config(path.parent / base)
    for section, values in cfg.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


# Machine-specific model locations; the only paths --paths-from may supply.
MODEL_PATH_KEYS = ("sd15", "aura", "ip_adapter_dir", "image_encoder")


def load_config(path, overrides, paths=None, paths_from=None):
    """`paths_from`: take the MODEL paths (sd15, aura, ...) from another config,
    e.g. the GPU box's gpu.yaml, so one training recipe runs on any machine.
    The recipe's own `data` and `out` are never replaced by it: doing so once
    trained the commercial recipe on the research data."""
    cfg = _read_config(Path(path))
    if paths_from:
        other = _read_config(Path(paths_from))["paths"]
        cfg["paths"].update({k: v for k, v in other.items() if k in MODEL_PATH_KEYS})
    for key, val in overrides.items():
        if val is not None:
            cfg["train"][key] = val
    for key, val in (paths or {}).items():
        if val is not None:
            cfg["paths"][key] = val
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else (ROOT / p).resolve())
    return cfg


def lr_at(step: int, t: dict, warm_from=None) -> float:
    """Learning rate at global `step` (see the module docstring). `warm_from`:
    the step the warm-up starts at, or None for no warm-up (resumed runs)."""
    lr = t["lr"]
    warm = t.get("lr_warmup", 0)
    if warm_from is not None and warm > 0:
        lr *= min(1.0, (step - warm_from + 1) / warm)
    start = t.get("lr_decay_start")
    if start is not None and step >= start:
        frac = min(1.0, (step - start) / max(1, t.get("lr_decay_steps", 5000)))
        lo = t.get("lr_min", 0.0)
        lr = lo + (lr - lo) * 0.5 * (1 + math.cos(math.pi * frac))
    return lr


def step_of(path) -> int:
    """The training step in a checkpoint/state folder name (step-0050000, state-0075000), else 0."""
    for part in reversed(Path(path).parts):
        m = re.fullmatch(r"(?:step|state)-(\d+)", part)
        if m:
            return int(m.group(1))
    return 0


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> dict:
    """Git commit when available (the box copy may not be a checkout) + hashes of the code that trains."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    files = [ROOT / "train.py", *sorted((ROOT / "faceid").glob("*.py"))]
    return {"commit": commit, "dirty": dirty,
            "files": {str(f.relative_to(ROOT)): sha256(f)[:16] for f in files}}


def latest_state(out_dir: Path):
    """Newest resumable state folder (state-XXXXXXX), or None."""
    states = sorted(Path(out_dir).glob("state-*"))
    return states[-1] if states else None


def build_models(sd15, dtype, device):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    variant = "fp16" if (Path(sd15) / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists() else None
    kw = dict(local_files_only=True, variant=variant)
    tok = CLIPTokenizer.from_pretrained(sd15, subfolder="tokenizer", local_files_only=True)
    text = CLIPTextModel.from_pretrained(sd15, subfolder="text_encoder", **kw)
    vae = AutoencoderKL.from_pretrained(sd15, subfolder="vae", **kw)
    unet = UNet2DConditionModel.from_pretrained(sd15, subfolder="unet", **kw)
    sched = DDPMScheduler.from_pretrained(sd15, subfolder="scheduler", local_files_only=True)
    for m in (text, vae, unet):
        m.requires_grad_(False)
    text.to(device, dtype)
    vae.to(device, dtype)
    unet.to(device, dtype)
    return tok, text, vae, unet, sched


@torch.no_grad()
def validate(cfg, accel, tok, text, vae, unet, step, out_dir, dtype):
    """Render held-out faces from their embeddings and score them with AuraFace."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from PIL import Image

    from faceid.aura import Aura, cosine

    v = cfg["val"]
    data = Path(cfg["paths"]["data"])
    lines = [json.loads(line) for line in (data / "meta.jsonl").read_text().splitlines() if line]
    embeds = np.load(data / "embeds.npy", mmap_mode="r")
    val = [(m, i) for i, m in enumerate(lines) if m["split"] == "val"][: v["num_faces"]]
    if not val:
        return None
    sched = DPMSolverMultistepScheduler.from_pretrained(cfg["paths"]["sd15"], subfolder="scheduler",
                                                        local_files_only=True)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tok, unet=unet,
                                   scheduler=sched, safety_checker=None, feature_extractor=None,
                                   requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    aura = Aura(cfg["paths"]["aura"])
    # The photo prompt, plus a sketch prompt when configured: identity under a
    # drawing style is what the sketch pipelines need, and what the photo-only
    # research adapter lost.
    prompts = {"photo": v["prompt"]}
    if v.get("sketch_prompt"):
        prompts["sketch"] = v["sketch_prompt"]
    rows, scores = [], {k: [] for k in prompts}
    for m, row in val:
        pos = torch.from_numpy(np.asarray(embeds[row], np.float32)).to(accel.device, dtype)[None, None]
        embeds_cfg = [torch.cat([torch.zeros_like(pos), pos])]          # [negative, positive]
        tiles = [Image.open(data / "images" / f"{m['id']}.jpg").convert("RGB").resize((256, 256))]
        for kind, prompt in prompts.items():
            with accel.autocast():      # adapter weights are fp32, the frozen UNet fp16
                img = pipe(prompt, negative_prompt=v["negative"], num_inference_steps=v["steps"],
                           guidance_scale=v["guidance"], ip_adapter_image_embeds=embeds_cfg, height=512,
                           width=512, generator=torch.Generator(accel.device).manual_seed(0)).images[0]
            tiles.append(img.resize((256, 256)))
            gen_emb, _ = aura.embed(np.asarray(img)[:, :, ::-1].copy())
            if gen_emb is not None:
                scores[kind].append(cosine(gen_emb, np.asarray(embeds[row], np.float32)))
        rows.append(tiles)
    sheet = Image.new("RGB", (256 * len(rows), 256 * (1 + len(prompts))), "white")
    for i, tiles in enumerate(rows):                 # columns = faces; rows = source, photo, sketch
        for j, tile in enumerate(tiles):
            sheet.paste(tile, (256 * i, 256 * j))
    sheet.save(out_dir / f"val_step{step:07d}.jpg", quality=90)
    means = {k: float(np.mean(s)) if s else float("nan") for k, s in scores.items()}
    parts = "  ".join(f"{k} {means[k]:.3f} ({len(scores[k])}/{len(val)})" for k in prompts)
    accel.print(f"[val] step {step}: AuraFace cosine {parts} (sheet val_step{step:07d}.jpg)")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--paths-from", help="take the paths section from this config (e.g. configs/gpu.yaml)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--resume", help="state directory from a previous run (…/state-XXXXXXX)")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--data", help="dataset folder (overrides paths.data)")
    ap.add_argument("--out", help="output folder (overrides paths.out)")
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--generic-captions", action="store_true",
                    help="train with fixed captions instead of the dataset's (or train.generic_captions)")
    ap.add_argument("--init-adapter", help="start from these adapter weights (.safetensors); warm start")
    ap.add_argument("--init-step", type=int, help="global step of --init-adapter (default: from its folder name)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup-steps", type=int, dest="lr_warmup")
    ap.add_argument("--decay-start", type=int, dest="lr_decay_start")
    ap.add_argument("--decay-steps", type=int, dest="lr_decay_steps")
    ap.add_argument("--keep-state-at", type=int, action="append", default=[],
                    help="never prune the resumable state of this step (repeatable; for cool-down branches)")
    ap.add_argument("--extra-data", help="second dataset mixed into training (e.g. data/faces_sfhq)")
    ap.add_argument("--extra-ratio", type=float, default=0.2, help="share of samples drawn from --extra-data")
    ap.add_argument("--extra-generic-captions", action="store_true",
                    help="fixed captions for --extra-data (SFHQ's own captions describe the identity)")
    ap.add_argument("--allow-cpu", action="store_true", help="run without CUDA (laptop smoke tests only)")
    args = ap.parse_args()
    cfg = load_config(args.config, {"max_steps": args.max_steps, "batch_size": args.batch_size,
                                    "val_every": args.val_every, "save_every": args.save_every,
                                    "num_workers": args.num_workers, "seed": args.seed, "lr": args.lr,
                                    "lr_warmup": args.lr_warmup, "lr_decay_start": args.lr_decay_start,
                                    "lr_decay_steps": args.lr_decay_steps},
                      {"data": args.data, "out": args.out}, args.paths_from)
    t = cfg["train"]
    if args.resume and args.init_adapter:
        raise SystemExit("--resume and --init-adapter exclude each other")
    if not torch.cuda.is_available() and not args.allow_cpu:
        # A CPU fallback would "train" for weeks without saying so
        raise SystemExit("CUDA is not available (driver/library mismatch? check nvidia-smi); "
                         "pass --allow-cpu only for a smoke test")
    keep_at = set(args.keep_state_at) | set(t.get("keep_states_at") or [])

    accel = Accelerator(mixed_precision=t["mixed_precision"], gradient_accumulation_steps=t["grad_accum"])
    set_seed(t["seed"])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(t["mixed_precision"], torch.float32)
    out_dir = Path(cfg["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, text, vae, unet, sched = build_models(cfg["paths"]["sd15"], dtype, accel.device)
    adapter = attach_adapter(unet, num_tokens=cfg["adapter"]["num_tokens"], mult=cfg["adapter"]["mult"])
    adapter.to(accel.device, torch.float32)          # trainable weights in fp32; autocast handles the rest
    adapter.requires_grad_(True)
    if t["grad_checkpointing"]:
        unet.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in adapter.parameters())
    accel.print(f"trainable parameters: {n_params / 1e6:.1f}M")

    opt_cls = torch.optim.AdamW
    if t["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            opt_cls = bnb.optim.AdamW8bit
        except ImportError:
            accel.print("bitsandbytes not installed; using torch AdamW")
    opt = opt_cls(adapter.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    step, warm_from, init = 0, 0, None
    if args.init_adapter:
        n = load_adapter(unet, args.init_adapter)
        step = warm_from = args.init_step if args.init_step is not None else step_of(args.init_adapter)
        init = {"path": str(Path(args.init_adapter).resolve()), "sha256": sha256(args.init_adapter), "step": step}
        accel.print(f"warm start: {n} tensors from {args.init_adapter} at step {step}")

    ds = FaceDataset(cfg["paths"]["data"], tok, "train", t["resolution"], t["drop_text"], t["drop_id"],
                     t["drop_both"], p_sketch=t.get("p_sketch", 0.0),
                     generic_captions=args.generic_captions or t.get("generic_captions", False))
    data_info = {"main": {"path": cfg["paths"]["data"], "n": len(ds), "excluded": ds.excluded}}
    sampler = None
    if args.extra_data:
        extra = FaceDataset(args.extra_data, tok, "train", t["resolution"], t["drop_text"], t["drop_id"],
                            t["drop_both"], p_sketch=t.get("p_sketch", 0.0),
                            generic_captions=args.extra_generic_captions)
        r = args.extra_ratio
        weights = torch.cat([torch.full((len(ds),), (1 - r) / len(ds)), torch.full((len(extra),), r / len(extra))])
        g = torch.Generator().manual_seed(t["seed"])
        sampler = WeightedRandomSampler(weights.double(), num_samples=len(ds) + len(extra), replacement=True,
                                        generator=g)
        data_info["extra"] = {"path": str(Path(args.extra_data).resolve()), "n": len(extra),
                              "excluded": extra.excluded, "ratio": r, "generic_captions": extra.generic_captions}
        ds = ConcatDataset([ds, extra])
        accel.print(f"mixing {r:.0%} of samples from {args.extra_data} ({len(extra)} faces)")
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=sampler is None, sampler=sampler,
                    num_workers=t["num_workers"], drop_last=True, pin_memory=True,
                    persistent_workers=t["num_workers"] > 0)
    first = ds.datasets[0] if isinstance(ds, ConcatDataset) else ds
    accel.print(f"train faces: {len(ds)} (excluded {first.excluded})  batch {t['batch_size']} x accum "
                f"{t['grad_accum']}  tokens {cfg['adapter']['num_tokens']}  p_sketch {first.p_sketch}  "
                f"captions {'generic' if first.generic_captions else 'dataset'}")
    adapter, opt, dl = accel.prepare(adapter, opt, dl)

    if args.resume:
        state = latest_state(out_dir) if args.resume == "latest" else Path(args.resume)
        if state is None:
            accel.print(f"no saved state in {out_dir}; starting from scratch")
        else:
            accel.load_state(str(state))
            step = int(state.name.split("-")[-1])
            warm_from = None                       # optimiser state restored: no second warm-up
            accel.print(f"resumed from {state} at step {step}")
    if step >= t["max_steps"]:
        raise SystemExit(f"already at step {step} >= max_steps {t['max_steps']}")

    metrics = out_dir / "metrics.jsonl"

    def log(rec):
        if accel.is_main_process:
            with metrics.open("a") as f:
                f.write(json.dumps({"time": round(time.time(), 1), **rec}) + "\n")

    if accel.is_main_process:
        run = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "start_step": step,
               "warmup_from": warm_from, "init_adapter": init, "resume": args.resume, "data": data_info,
               "keep_states_at": sorted(keep_at), "code": code_version(), "config": cfg,
               "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        name = "run.json" if not (out_dir / "run.json").exists() else f"run-{step:07d}.json"
        (out_dir / name).write_text(json.dumps(run, indent=2, default=str))
        accel.print(f"run record: {out_dir / name}")

    def checkpoint(step):
        """Adapter weights (for inference) + full resumable state; prune old states."""
        if not accel.is_main_process:
            return
        ckpt = out_dir / f"step-{step:07d}"
        save_adapter(unet, ckpt / WEIGHT_NAME)
        accel.save_state(str(out_dir / f"state-{step:07d}"))
        for old in sorted(out_dir.glob("state-*"))[:-t.get("keep_states", 2)]:
            if step_of(old) not in keep_at:
                shutil.rmtree(old, ignore_errors=True)
        accel.print(f"saved step {step}: {ckpt / WEIGHT_NAME}")

    unet.train()
    t0, running = time.time(), 0.0
    try:
        while step < t["max_steps"]:
            for batch in dl:
                lr = lr_at(step, t, warm_from)
                for group in opt.param_groups:
                    group["lr"] = lr
                with accel.accumulate(adapter):
                    with torch.no_grad():
                        lat = vae.encode(batch["pixel_values"].to(dtype)).latent_dist.sample()                             * vae.config.scaling_factor
                        enc = text(batch["input_ids"])[0]
                    noise = torch.randn_like(lat)
                    ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=lat.device)
                    noisy = sched.add_noise(lat, noise, ts)
                    target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(lat, noise, ts)
                    with accel.autocast():
                        pred = unet(noisy, ts, encoder_hidden_states=enc,
                                    added_cond_kwargs={"image_embeds": id_embeds(batch["id_embed"].to(dtype))}).sample
                    loss = F.mse_loss(pred.float(), target.float())
                    accel.backward(loss)
                    if accel.sync_gradients:
                        accel.clip_grad_norm_(adapter.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if not accel.sync_gradients:
                    continue
                step += 1
                running += loss.item()
                if step % t["log_every"] == 0:
                    rate = t["log_every"] / (time.time() - t0)
                    eta = (t["max_steps"] - step) / max(rate, 1e-9) / 3600
                    accel.print(f"step {step}/{t['max_steps']}  loss {running / t['log_every']:.4f}  "
                                f"lr {lr:.2e}  {rate:.2f} it/s  eta {eta:.1f} h")
                    log({"step": step, "loss": round(running / t["log_every"], 5), "lr": lr,
                         "it_s": round(rate, 3)})
                    t0, running = time.time(), 0.0
                if step % t["save_every"] == 0 or step == t["max_steps"]:
                    checkpoint(step)
                if (not args.no_val and accel.is_main_process
                        and (step % t["val_every"] == 0 or step == t["max_steps"])):
                    unet.eval()
                    val = validate(cfg, accel, tok, text, vae, unet, step, out_dir, dtype)
                    if val:
                        log({"step": step, "val": {k: round(v, 4) for k, v in val.items()}})
                    unet.train()
                if step >= t["max_steps"]:
                    break
    except KeyboardInterrupt:
        # Ctrl+C = pause: save where we are, then exit; continue with --resume latest
        accel.print(f"interrupted at step {step}; saving before exit")
        if step and not (out_dir / f"state-{step:07d}").exists():
            checkpoint(step)
        argv, skip = [], False                 # the same command, with --resume latest
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == "--resume":
                skip = True
                continue
            argv.append(a)
        accel.print("paused. continue with: python train.py " + " ".join(argv + ["--resume", "latest"]))
        return
    accel.print("training finished")


if __name__ == "__main__":
    main()
