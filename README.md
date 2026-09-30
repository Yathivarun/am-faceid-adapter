# faceid-adapter

This is a research proof of concept: an **AuraFace-driven identity adapter for SD1.5**, a commercial-safe replacement for InsightFace-based FaceID/InstantID. It is meant for the FRU face-inpaint pipeline, where likeness is currently limited by IP-Adapter full-face/plus-face, which use CLIP features.

- **Identity:** AuraFace-v1 (Apache-2.0) 512-D face vector → 4 identity tokens → decoupled cross-attention in a frozen SD1.5 UNet. This is the same architecture and weight format as h94 IP-Adapter-FaceID, so diffusers loads the result natively (`load_ip_adapter(..., image_encoder_folder=None)` + `ip_adapter_image_embeds`).
- **Pose** (at inference, in the face-inpaint lab): the character's MediaPipe face-mesh ControlNet. Adapter + mesh ControlNet ≈ a light InstantID.
- **Data:** FFHQ + CelebA-HQ, **research only** (see `prep/download.md`). Before any production use, retrain on free/synthetic or licensed data.

| Path | Purpose |
|---|---|
| `prep/download.md` | Dataset links, sizes, licences, resumable download |
| `prep/build_dataset.py` | Images → AuraFace embeddings + captions → `data/faces/` |
| `faceid/aura.py` | AuraFace detection / embedding / gender |
| `faceid/model.py` | Attach the adapter to a UNet; save/load in h94 FaceID format |
| `faceid/data.py` | Training dataset (flip, caption/identity dropout for CFG) |
| `train.py` | accelerate training loop, checkpoints, validation images + AuraFace score |
| `eval.py` | Our adapter vs full-face / plus-face on held-out faces: AuraFace cosine + sheet |
| `configs/poc.yaml` | Paths and hyper-parameters |

## Environment

- **GPU box:** the offline **master venv** (`AM/linux-wheelhouse`, `bash install.sh master`): torch 2.2 cu121, diffusers 0.29.2, transformers 4.38.2, accelerate 0.27.2, insightface (AuraFace). Extra for training: `bitsandbytes` (8-bit AdamW, optional), `pyarrow` (parquet) and `pyyaml`; see `requirements.txt`.
- **Laptop (code + smoke tests only):** `.venv`, layered on the existing local environments. Not for real training.

## Run

```bash
python prep/build_dataset.py --parquet "data/raw/ffhq512/*.parquet" --images data/raw/celebahq --out data/faces
accelerate launch train.py --config configs/poc.yaml          # or: python train.py --config ...
python eval.py --config configs/poc.yaml --adapter checkpoints/poc/step-50000/faceid_aura_sd15.safetensors
python -m pytest tests -q
```

**Go/no-go:** mean AuraFace cosine on held-out faces clearly above full-face and plus-face at the same seeds and prompts, and visibly better likeness in the face-inpaint lab's pose mode.
