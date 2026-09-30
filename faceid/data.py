"""Training dataset: 512 px face image, its AuraFace embedding, its caption.

Each image is its own target (the adapter learns to reproduce the identity it
is given), so no per-person pairs are needed. For classifier-free guidance a
share of samples drop the caption, the identity, or both; a dropped identity
is a zero vector, which is also what inference uses as the negative.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root, tokenizer, split="train", size=512, drop_text=0.05, drop_id=0.05,
                 drop_both=0.05, flip=True):
        root = Path(root)
        lines = [json.loads(line) for line in (root / "meta.jsonl").read_text().splitlines() if line]
        embeds = np.load(root / "embeds.npy", mmap_mode="r")
        if len(lines) != len(embeds):
            raise ValueError(f"meta ({len(lines)}) and embeds ({len(embeds)}) differ")
        self.items = [(m, i) for i, m in enumerate(lines) if m["split"] == split]
        self.embeds, self.root, self.tok, self.size = embeds, root, tokenizer, size
        self.drop = (drop_text, drop_id, drop_both)
        self.flip = flip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta, row = self.items[idx]
        img = Image.open(self.root / "images" / f"{meta['id']}.jpg").convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.LANCZOS)
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        px = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        emb = torch.from_numpy(np.asarray(self.embeds[row], np.float32))
        caption = meta["caption"]
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
