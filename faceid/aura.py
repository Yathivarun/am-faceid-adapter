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
        if not (pack / "glintr100.onnx").is_file():
            # Fail here: insightface would otherwise try to download the pack
            raise FileNotFoundError(f"AuraFace-v1 pack not found at {pack} (needs glintr100.onnx, scrfd, genderage)")
        if pack.parent.name != "models":
            # FaceAnalysis reads <root>/models/<name>
            raise ValueError(f"AuraFace pack must sit in a folder named 'models': {pack}")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=pack.name, root=str(pack.parent.parent),
                                allowed_modules=["detection", "recognition", "genderage"],
                                providers=providers)
        self.app.prepare(ctx_id=0 if gpu else -1, det_size=(det_size, det_size))
        if gpu:
            # onnxruntime falls back to CPU silently when its CUDA build doesn't
            # match the machine's CUDA; with gpu=True that must be an error.
            used = {name: m.session.get_providers()[0] for name, m in self.app.models.items()}
            off = [name for name, p in used.items() if p != "CUDAExecutionProvider"]
            if off:
                raise RuntimeError(f"AuraFace asked for GPU but runs on CPU for {off}: onnxruntime-gpu "
                                   f"does not match this CUDA (e.g. Colab CUDA 12 needs onnxruntime-gpu==1.22.0)")

    def faces(self, bgr: np.ndarray, pad: float = 0.25):
        """All faces, largest first, in the coordinates of `bgr`.

        The image is padded by `pad` x its size on every side before detection:
        SCRFD misses faces that fill most of the frame (tight portrait crops
        such as FFHQ / CelebA-HQ: 57% missed unpadded, 0% at 0.25 on a CelebA
        sample). Boxes and keypoints are shifted back afterwards."""
        p = int(round(pad * max(bgr.shape[:2]))) if pad else 0
        img = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if p else bgr
        found = self.app.get(img) or []
        for f in found:
            f.bbox = f.bbox - p
            if getattr(f, "kps", None) is not None:
                f.kps = f.kps - p
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
