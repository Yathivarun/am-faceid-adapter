"""Train the AuraFace identity adapter on a frozen SD1.5.

    accelerate launch train.py --config configs/poc.yaml      # or plain: python train.py --config ...
    python train.py --config configs/poc.yaml --max-steps 20 --batch-size 1 --no-val   # smoke test

Trains only the identity projection and the to_k_ip/to_v_ip layers
(~22M params). Every `save_every` steps: adapter weights (h94 FaceID format,
loadable with `pipe.load_ip_adapter`) + full resumable state. Every
`val_every` steps: held-out faces rendered from their embeddings, saved as a
sheet, and scored with AuraFace (mean cosine to the source face).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faceid.data import FaceDataset  # noqa: E402
from faceid.model import attach_adapter, id_embeds, save_adapter  # noqa: E402

WEIGHT_NAME = "faceid_aura_sd15.safetensors"


def load_config(path, overrides, paths=None):
    cfg = yaml.safe_load(Path(path).read_text())
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
    tiles, scores = [], []
    for m, row in val:
        pos = torch.from_numpy(np.asarray(embeds[row], np.float32)).to(accel.device, dtype)[None, None]
        embeds_cfg = [torch.cat([torch.zeros_like(pos), pos])]          # [negative, positive]
        with accel.autocast():      # adapter weights are fp32, the frozen UNet fp16
            img = pipe(v["prompt"], negative_prompt=v["negative"], num_inference_steps=v["steps"],
                       guidance_scale=v["guidance"], ip_adapter_image_embeds=embeds_cfg, height=512, width=512,
                       generator=torch.Generator(accel.device).manual_seed(0)).images[0]
        src = Image.open(data / "images" / f"{m['id']}.jpg").convert("RGB").resize((256, 256))
        tiles.append((src, img.resize((256, 256))))
        gen_emb, _ = aura.embed(np.asarray(img)[:, :, ::-1].copy())
        if gen_emb is not None:
            scores.append(cosine(gen_emb, np.asarray(embeds[row], np.float32)))
    sheet = Image.new("RGB", (256 * len(tiles), 512), "white")
    for i, (src, gen) in enumerate(tiles):
        sheet.paste(src, (256 * i, 0))
        sheet.paste(gen, (256 * i, 256))
    sheet.save(out_dir / f"val_step{step:07d}.jpg", quality=90)
    score = float(np.mean(scores)) if scores else float("nan")
    accel.print(f"[val] step {step}: AuraFace cosine {score:.3f} over {len(scores)}/{len(val)} faces "
                f"(sheet val_step{step:07d}.jpg)")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--resume", help="state directory from a previous run (…/state-XXXXXXX)")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--data", help="dataset folder (overrides paths.data)")
    ap.add_argument("--out", help="output folder (overrides paths.out)")
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--num-workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config, {"max_steps": args.max_steps, "batch_size": args.batch_size,
                                    "val_every": args.val_every, "save_every": args.save_every,
                                    "num_workers": args.num_workers},
                      {"data": args.data, "out": args.out})
    t = cfg["train"]

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

    ds = FaceDataset(cfg["paths"]["data"], tok, "train", t["resolution"], t["drop_text"], t["drop_id"],
                     t["drop_both"])
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"],
                    drop_last=True, pin_memory=True, persistent_workers=t["num_workers"] > 0)
    accel.print(f"train faces: {len(ds)}  batch {t['batch_size']} x accum {t['grad_accum']}")
    adapter, opt, dl = accel.prepare(adapter, opt, dl)

    step = 0
    if args.resume:
        state = latest_state(out_dir) if args.resume == "latest" else Path(args.resume)
        if state is None:
            accel.print(f"no saved state in {out_dir}; starting from scratch")
        else:
            accel.load_state(str(state))
            step = int(state.name.split("-")[-1])
            accel.print(f"resumed from {state} at step {step}")

    def checkpoint(step):
        """Adapter weights (for inference) + full resumable state; prune old states."""
        if not accel.is_main_process:
            return
        ckpt = out_dir / f"step-{step:07d}"
        save_adapter(unet, ckpt / WEIGHT_NAME)
        accel.save_state(str(out_dir / f"state-{step:07d}"))
        for old in sorted(out_dir.glob("state-*"))[:-t.get("keep_states", 2)]:
            shutil.rmtree(old, ignore_errors=True)
        accel.print(f"saved step {step}: {ckpt / WEIGHT_NAME}")

    unet.train()
    t0, running = time.time(), 0.0
    try:
        while step < t["max_steps"]:
            for batch in dl:
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
                                f"{rate:.2f} it/s  eta {eta:.1f} h")
                    t0, running = time.time(), 0.0
                if step % t["save_every"] == 0 or step == t["max_steps"]:
                    checkpoint(step)
                if (not args.no_val and accel.is_main_process
                        and (step % t["val_every"] == 0 or step == t["max_steps"])):
                    unet.eval()
                    validate(cfg, accel, tok, text, vae, unet, step, out_dir, dtype)
                    unet.train()
                if step >= t["max_steps"]:
                    break
    except KeyboardInterrupt:
        # Ctrl+C = pause: save where we are, then exit; continue with --resume latest
        accel.print(f"interrupted at step {step}; saving before exit")
        if step and not (out_dir / f"state-{step:07d}").exists():
            checkpoint(step)
        accel.print(f"paused. continue with: python train.py --config {args.config} --resume latest")
        return
    accel.print("training finished")


if __name__ == "__main__":
    main()
