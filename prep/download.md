# Datasets (research proof of concept only)

Both datasets are **non-commercial / research-only**. They are used to prove that an AuraFace-driven identity adapter works. A production adapter must be retrained on free/synthetic data or on a licensed dataset, and none of these images or derived weights may ship.

| Dataset | Images | Get it from | Size | Licence |
|---|---|---|---|---|
| **FFHQ 512 + captions** (recommended) | 70,000 aligned faces, 512 px, one caption each | https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption, 54 parquet shards (`data/train-000NN-of-00054*.parquet`) | ~27 GB | Images: FFHQ terms, CC BY-NC-SA 4.0 (dataset) / per-image Flickr licences. The mirror's "MIT" tag covers the repo, **not** the images. |
| FFHQ 1024 (official, fallback) | 70,000 aligned faces, 1024 px PNG | https://github.com/NVlabs/ffhq-dataset (Google Drive, `download_ffhq.py --images`) | 89 GB | CC BY-NC-SA 4.0 |
| **CelebA-HQ 512** (via CelebAMask-HQ) | 30,000 faces, 512 px JPG | https://github.com/switchablenorms/CelebAMask-HQ, Google Drive https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv (Baidu: https://pan.baidu.com/s/1wN1E-B1bJ7mE1mrn9loj5g) | ~3 GB | Non-commercial research only |

## Downloading on a weak connection

The FFHQ parquet shards are independent (~0.5 GB each), so fetch them one at a time and resume failures. Use `curl -C -` or `huggingface-cli`:

```bash
# one shard (repeat NN = 00..53); -C - resumes a partial file
curl -L -C - -o data/raw/ffhq512/train-00000-of-00054.parquet \
  "https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption/resolve/main/data/train-00000-of-00054.parquet"

# or everything, resumable
huggingface-cli download Ryan-sjtu/ffhq512-caption --repo-type dataset --local-dir data/raw/ffhq512
```

Check the exact shard file names on the dataset's "Files" tab before scripting the loop.

For CelebAMask-HQ, download `CelebAMask-HQ.zip` from Google Drive in a browser (it resumes), unzip, and keep only `CelebA-HQ-img/` (30,000 JPGs).

## Layout expected by `prep/build_dataset.py`

```
data/raw/ffhq512/*.parquet        # columns: image (bytes), text (caption)
data/raw/celebahq/*.jpg           # CelebA-HQ-img/
```

Then, on the GPU box:

```bash
python prep/build_dataset.py --parquet "data/raw/ffhq512/*.parquet" --images data/raw/celebahq --out data/faces
```
