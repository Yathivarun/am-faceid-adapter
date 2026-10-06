"""Training dataset: 512 px face image, its AuraFace embedding, its caption.

Each image is its own target (the adapter learns to reproduce the identity it
is given), so no per-person pairs are needed. For classifier-free guidance a
share of samples drop the caption, the identity, or both; a dropped identity
is a zero vector, which is also what inference uses as the negative.

Sketch-domain samples (p_sketch): the target becomes a pencil rendering of the
photo with a sketch caption, while the identity vector stays the photo's. The
research adapter, trained on photos only, lost most of its likeness under
sketch prompts (AuraFace cosine 0.53 -> ~0.38); this teaches the identity
tokens to survive a drawing style.

An optional `exclude.txt` (one id per line, e.g. from prep/diversity.py's
near-duplicate pass) removes images from training.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SKETCH_CAPTIONS = (
    "a pencil sketch drawing",
    "a graphite pencil portrait sketch",
    "a monochrome hand-drawn sketch",
    "a black and white pencil drawing",
)


def pencil_sketch(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Cheap pencil rendering of a photo (colour-dodge of a blurred negative),
    with a little random variation in stroke softness and darkness."""
    rng = rng or random
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = img.size[0] / rng.uniform(48, 96)
    blur = cv2.GaussianBlur(255.0 - gray, (0, 0), sigma)
    dodge = np.clip(gray * 255.0 / (256.0 - blur), 0, 255)
    out = 255.0 * (dodge / 255.0) ** rng.uniform(1.5, 2.5)          # darker, pencil-like strokes
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out).convert("RGB")


class FaceDataset(Dataset):
    def __init__(self, root, tokenizer, split="train", size=512, drop_text=0.05, drop_id=0.05,
                 drop_both=0.05, flip=True, p_sketch=0.0):
        root = Path(root)
        lines = [json.loads(line) for line in (root / "meta.jsonl").read_text().splitlines() if line]
        embeds = np.load(root / "embeds.npy", mmap_mode="r")
        if len(lines) != len(embeds):
            raise ValueError(f"meta ({len(lines)}) and embeds ({len(embeds)}) differ")
        exclude_file = root / "exclude.txt"
        exclude = set(exclude_file.read_text().split()) if exclude_file.exists() else set()
        self.items = [(m, i) for i, m in enumerate(lines) if m["split"] == split and m["id"] not in exclude]
        self.excluded = len(exclude)
        self.embeds, self.root, self.tok, self.size = embeds, root, tokenizer, size
        self.drop = (drop_text, drop_id, drop_both)
        self.flip = flip
        self.p_sketch = p_sketch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta, row = self.items[idx]
        img = Image.open(self.root / "images" / f"{meta['id']}.jpg").convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.LANCZOS)
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        caption = meta["caption"]
        if self.p_sketch and random.random() < self.p_sketch:
            img = pencil_sketch(img)
            caption = f"{random.choice(SKETCH_CAPTIONS)}. {caption}"
        px = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        emb = torch.from_numpy(np.asarray(self.embeds[row], np.float32))
        r = random.random()
        t, i, b = self.drop
        if r < t:
            caption = ""
        elif r < t + i:
            emb = torch.zeros_like(emb)
        elif r < t + i + b:
            caption, emb = "", torch.zeros_like(emb)
        ids = self.tok(caption, max_length=self.tok.model_max_length, padding="max_length",
                       truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": px, "input_ids": ids, "id_embed": emb}
