"""AuraFace-v1 (Apache-2.0, InsightFace-compatible pack): face detection,
512-D identity embedding and gender. The embedding is L2-normalised, the same
vector the adapter is conditioned on at training and inference time."""

from pathlib import Path

import cv2
import numpy as np


class Aura:
    def __init__(self, pack_dir: str | Path, gpu: bool = False, det_size: int = 640):
        from insightface.app import FaceAnalysis

        pack = Path(pack_dir)
        if pack.parent.name != "models":
            # FaceAnalysis reads <root>/models/<name>
            raise ValueError(f"AuraFace pack must sit in a folder named 'models': {pack}")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=pack.name, root=str(pack.parent.parent),
                                allowed_modules=["detection", "recognition", "genderage"],
                                providers=providers)
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=(det_size, det_size))

    def faces(self, bgr: np.ndarray):
        """All faces, largest first."""
        found = self.app.get(bgr) or []
        return sorted(found, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def embed(self, bgr: np.ndarray):
        """(embedding float32[512] unit-norm, face) of the largest face, or (None, None)."""
        found = self.faces(bgr)
        if not found:
            return None, None
        f = found[0]
        return f.normed_embedding.astype(np.float32), f

    def embed_path(self, path: str | Path):
        return self.embed(cv2.imread(str(path)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
