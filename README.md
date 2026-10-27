# faceid-adapter

This is a research proof of concept: an **AuraFace-driven identity adapter for SD1.5**, a commercial-safe replacement for InsightFace-based FaceID/InstantID. It is meant for the FRU face-inpaint pipeline, where likeness is currently limited by IP-Adapter full-face/plus-face, which use CLIP features.

- **Identity:** AuraFace-v1 (Apache-2.0) 512-D face vector → 4 identity tokens → decoupled cross-attention in a frozen SD1.5 UNet. This is the same architecture and weight format as h94 IP-Adapter-FaceID, so diffusers loads the result natively (`load_ip_adapter(..., image_encoder_folder=None)` + `ip_adapter_image_embeds`).
- **Pose** (at inference, in the face-inpaint lab): the character's MediaPipe face-mesh ControlNet. Adapter + mesh ControlNet ≈ a light InstantID.
- **Data:** FFHQ + CelebA-HQ, **research only** (see `prep/download.md`). Before any production use, retrain on free/synthetic or licensed data.

Related repos:
- [am-wpu-face-inpaint](https://github.com/Yathivarun/am-wpu-face-inpaint): the FRU lab that uses this adapter (character renders, scoring).
- [wpu-face-inpaint-research](https://github.com/Yathivarun/wpu-face-inpaint-research): results, charts, research log.

## Status (2026-10-27)

- **Adapter in use:** `checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors`. That is the Stage A run (`configs/real_v3.yaml`): warm-started from poc 50k, constant LR to 150k, then a 5k cool-down. Training has converged; more steps gave nothing measurable.
- **Best use:** fused with h94 full-face. In the FRU lab recipe (4:5 crop, face-erased lineart + eyes/lips keypoints, ours 1.0 + full-face 0.5), FaceNet identity at 512 is **0.578 vs 0.418** for full-face alone, and ours wins every matched pair.
- **Alone,** it trails full-face@0.8 on FaceNet. Its value is in the fusion, and in a vectors-only path.
- **Weights are research-only** (FFHQ / CelebA-HQ). The commercial retrain (branch `commercial`, SFHQ-T2I) showed a synthetic-to-real gap: 0.53 on SFHQ's own faces vs 0.31 on real visitors. It is not solved yet.
- The weights, datasets and checkpoints are **not in git**. They live on the GPU box (`~/WPU/testings/faceid-adapter/{data,checkpoints}`); back them up separately.

### Branches

| Branch | What |
|---|---|
| `main` | Same as `real-v3` (the current work) |
| `poc` | First 50k run: data prep, training, eval v1, Colab dataset build |
| `commercial` | Recipe v2 on SFHQ-T2I: 16 tokens, sketch-domain samples, distribution alignment |
| `real-v3` | Stage A: warm start + WSD LR to 155k, eval v2 on real people, run queues |

Each branch builds on the previous one, in a straight line: `poc` → `commercial` → `real-v3`.

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
| `configs/commercial.yaml` | Recipe v2 on SFHQ-T2I (inherits `poc.yaml`) |
| `prep/verify_sfhq.py` | Checks the SFHQ-T2I zip: CRCs, CSV ↔ images, per-model counts, AuraFace sample |
| `prep/diversity.py` | Identity spread / near-duplicates; optional `exclude.txt` |
| `configs/real_v3.yaml` | Stage A: warm start from poc 50k, warm-up-stable-decay LR, kept states for cool-down branches |
| `prep/build_evalset.py` | Real-people evaluation set (registration + FRU captures, quality flags, duplicate-person groups, dev/test split). Evaluation only |
| `eval2.py` | Eval v2: reg2fru / fru2fru protocols, AuraFace + **FaceNet** (independent) scorers, impostor margin, bootstrap CIs, paired tests, adapter fusion modes |
| `prep/compare_sheet.py` | Registration + captures next to each mode's generations, for visual review |
| `prep/embed_shift.py` | Where real visitors' AuraFace vectors sit relative to the training vectors (the synthetic-to-real gap) |
| `prep/colab_build.ipynb` | Builds the training set on a Colab T4 (AuraFace on CUDA) |
| `queue_holiday.sh`, `queue_followup.sh` | The unattended GPU-box runs behind the Stage A / G1 / G2 results (exact record) |
| `tests/` | Projection shapes, FaceID-format save/load round trip, commercial recipe, v3 schedule |

`configs/gpu.yaml` / `gpu_plus.yaml` (model paths on the GPU box, passed with `--paths-from`) are machine-specific and not in the repo. Copy `poc.yaml`'s `paths:` block and point it at the box's model folders.

## Environment

- **GPU box:** the offline **master venv** (`AM/linux-wheelhouse`, `bash install.sh master`): torch 2.2 cu121, diffusers 0.29.2, transformers 4.38.2, accelerate 0.27.2, insightface (AuraFace). Extra for training: `bitsandbytes` (8-bit AdamW, optional), `pyarrow` (parquet) and `pyyaml`; see `requirements.txt`.
- **Laptop (code + smoke tests only):** `.venv`, layered on the existing local environments. Not for real training.

## Run

```bash
python prep/build_dataset.py --parquet "data/raw/ffhq512/*.parquet" --images data/raw/celebahq --out data/faces
accelerate launch train.py --config configs/poc.yaml          # or: python train.py --config ...
python eval.py --config configs/poc.yaml --adapter checkpoints/poc/step-50000/faceid_aura_sd15.safetensors
python -m pytest tests -q

# Stage A (the current adapter): warm start from poc 50k, then a cool-down branch from a kept state
python train.py --config configs/real_v3.yaml --paths-from configs/gpu.yaml \
    --init-adapter checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors --keep-state-at 75000
# eval v2 on the real-people set (see eval2.py's docstring for modes and protocols)
python eval2.py --config configs/poc.yaml --paths-from configs/gpu_plus.yaml --evalset <evalset.json> --split dev \
    --protocol reg2fru --adapter <step-XXXXXXX/faceid_aura_sd15.safetensors> \
    --modes ours full-face "ours+full-face@0.8,0.5" --baseline full-face --out checkpoints/ev2/<name>
```

### Pause / resume

- **Ctrl+C = pause:** it saves the current step (adapter + full training state) and exits.
- `python train.py --config configs/poc.yaml --resume latest` continues from the newest saved state, with the optimiser, fp16 scaler and random-number state restored. `--resume checkpoints/poc/state-XXXXXXX` resumes from a specific state.
- It also saves automatically every `save_every` steps (2,500). Only the newest `keep_states` (2) resumable states are kept, while every `step-XXXXXXX/faceid_aura_sd15.safetensors` (~46 MB, for inference) is kept.

### Time and storage (3090 Ti, SD1.5)

| | |
|---|---|
| Dataset build (~100k faces, AuraFace on CPU) | ~4–8 h, resumable |
| Training, 50k steps at batch 8 | ~7–11 h (150k: ~21–32 h) |
| Disk: raw FFHQ-512 parquet + CelebA-HQ | ~30 GB (deletable after the build) |
| Disk: built dataset + checkpoints + 2 states | ~10–11 GB |
| **Free space to plan for** | **~60–70 GB** (with buffer); ~20–25 GB once raw data is deleted |

**Go/no-go:** mean AuraFace cosine on held-out faces clearly above full-face and plus-face at the same seeds and prompts, and visibly better likeness in the face-inpaint lab's pose mode.

## Commercial path (branch `commercial`)

**Research result (`poc`, 50k steps on FFHQ + CelebA-HQ):**
- Validation cosine reached 0.52, and photo-prompt identity was close to full-face.
- Under a **sketch prompt** identity fell to ~0.38, against full-face's 0.55. In the lab's pose mode it tied plus-face on score but was weaker by eye, and raising `--ip-scale` barely changed anything.
- Drawing quality was the best of the three.

**Recipe v2** (`configs/commercial.yaml`, which inherits `poc.yaml`):
- **Data:** SFHQ-T2I, synthetic and MIT-licensed, **FLUX.1-schnell + SDXL images only**, because the dev/pro/DALL-E 3 output terms forbid training competing models. See `data/raw/sfhq_t2i/SOURCE.md`. The generation prompts are used as captions.
- **Sketch-domain samples:** `p_sketch: 0.3`. The target is a pencil rendering of the photo with a sketch caption; the identity vector stays the photo's.
- **Identity tokens:** 16 instead of 4. diffusers infers the count from the weights.
- **Validation:** scores a sketch prompt as well as the photo prompt.

```bash
python prep/verify_sfhq.py --zip data/raw/sfhq_t2i/download/archive.zip --report data/raw/sfhq_t2i/verify_report.json
python prep/build_dataset.py --zip data/raw/sfhq_t2i/download/archive.zip \
    --keep-models FLUX1_schnell SDXL --out data/faces_sfhq [--config configs/gpu.yaml]
python prep/diversity.py data/faces_sfhq data/faces --dedupe 0.6 [--write-exclude]   # same-face check
python train.py --config configs/commercial.yaml --paths-from configs/gpu.yaml --max-steps 15000   # pilot
python train.py --config configs/commercial.yaml --paths-from configs/gpu.yaml --resume latest     # full run
```
