"""Commercial-path pieces: zip/CSV source with licence filter, sketch-domain
samples, near-duplicate filter, config inheritance. CPU only, no models."""

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prep"))

import build_dataset  # noqa: E402
import diversity  # noqa: E402
from faceid.data import FaceDataset, pencil_sketch  # noqa: E402


def _jpg():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "JPEG")
    return buf.getvalue()


def _sfhq_zip(path):
    rows = [("SDXL_image_0000002.jpg", "SDXL", "a man with glasses"),
            ("FLUX1_dev_image_0000001.jpg", "FLUX1_dev", "a woman"),
            ("FLUX1_schnell_image_0000003.jpg", "FLUX1_schnell", "a child")]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["image_filename", "model_used", "text_prompt", "configs"])
    for name, model, text in rows:
        w.writerow([name, model, text, "{}"])
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SFHQ_T2I_dataset.csv", out.getvalue())
        for name, _, _ in rows:
            zf.writestr(f"images/images/{name}", _jpg())


def test_zip_source_keeps_only_licensed_generators(tmp_path):
    z = tmp_path / "archive.zip"
    _sfhq_zip(z)
    items = list(build_dataset._zip_items(z, "SFHQ_T2I_dataset.csv", ["FLUX1_schnell", "SDXL"]))
    assert [(i[0], i[3], i[4]) for i in items] == [
        ("sfhq_FLUX1_schnell_image_0000003", "a child", "FLUX1_schnell"),
        ("sfhq_SDXL_image_0000002", "a man with glasses", "SDXL")]
    assert all(i[2][:2] == b"\xff\xd8" for i in items)           # JPEG bytes


def test_zip_source_without_filter_takes_everything(tmp_path):
    z = tmp_path / "archive.zip"
    _sfhq_zip(z)
    assert len(list(build_dataset._zip_items(z, "SFHQ_T2I_dataset.csv", []))) == 3


def test_pencil_sketch_is_grey_and_same_size():
    img = Image.new("RGB", (96, 96), (200, 50, 50))
    out = pencil_sketch(img)
    a = np.asarray(out)
    assert out.size == img.size and out.mode == "RGB"
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()


class _Tok:
    model_max_length = 8

    def __call__(self, text, **kw):
        import torch
        _Tok.last = text
        return type("T", (), {"input_ids": torch.zeros(1, 8, dtype=torch.long)})()


def _dataset(tmp_path, n=4, exclude=()):
    (tmp_path / "images").mkdir()
    lines = []
    for i in range(n):
        Image.new("RGB", (32, 32), (i * 40, 80, 120)).save(tmp_path / "images" / f"f{i}.jpg")
        lines.append({"id": f"f{i}", "caption": f"face {i}", "split": "train"})
    (tmp_path / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in lines))
    np.save(tmp_path / "embeds.npy", np.ones((n, 512), np.float16))
    if exclude:
        (tmp_path / "exclude.txt").write_text("\n".join(exclude))
    return tmp_path


def test_sketch_samples_get_a_sketch_caption_and_keep_the_identity(tmp_path):
    ds = FaceDataset(_dataset(tmp_path), _Tok(), size=32, drop_text=0, drop_id=0, drop_both=0,
                     p_sketch=1.0)
    item = ds[0]
    assert "sketch" in _Tok.last or "drawing" in _Tok.last
    assert _Tok.last.endswith("face 0")
    assert float(item["id_embed"].sum()) == 512.0                 # identity is the photo's


def test_exclude_file_removes_images(tmp_path):
    ds = FaceDataset(_dataset(tmp_path, exclude=("f1", "f3")), _Tok(), size=32)
    assert len(ds) == 2 and ds.excluded == 2


def test_near_duplicates_keep_the_first_of_each_cluster():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 512)).astype(np.float32)
    emb = np.stack([base[0], base[1], base[0] + 0.01, base[2], base[1] + 0.01])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert diversity.near_duplicates(emb, 0.9, chunk=2).tolist() == [2, 4]


def test_config_base_and_paths_from(tmp_path):
    from train import load_config

    (tmp_path / "base.yaml").write_text(
        "paths: {sd15: /m/sd, data: d1, out: o1}\nadapter: {num_tokens: 4, mult: 2}\n"
        "train: {lr: 0.1, p_sketch: 0.0}\nval: {steps: 3}\n")
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\npaths: {data: d2}\nadapter: {num_tokens: 16}\ntrain: {p_sketch: 0.3}\n")
    # the box config also names its own data/out: those must NOT leak into the recipe
    (tmp_path / "box.yaml").write_text("paths: {sd15: /box/sd, data: research, out: poc}\n")
    cfg = load_config(tmp_path / "child.yaml", {}, None, tmp_path / "box.yaml")
    assert cfg["adapter"] == {"num_tokens": 16, "mult": 2}
    assert cfg["train"] == {"lr": 0.1, "p_sketch": 0.3}
    assert Path(cfg["paths"]["sd15"]).as_posix().endswith("/box/sd")      # resolve() adds C: on Windows
    assert Path(cfg["paths"]["data"]).name == "d2"
    assert Path(cfg["paths"]["out"]).name == "o1"


@pytest.mark.parametrize("kind", ["mean", "coral"])
def test_aligner_moves_source_onto_target(kind):
    """A shifted, rescaled copy of the target cloud is mapped back onto it."""
    import eval as ev

    rng = np.random.default_rng(0)
    d = 512
    tgt = ev._unit(rng.normal(size=(4000, d)) * 0.05 + rng.normal(size=d) * 0.2)
    offset = rng.normal(size=d) * 0.3
    src = ev._unit(tgt[:2000] * 1.5 + offset)                  # same faces, shifted domain
    align = ev.make_aligner(kind, src, tgt)
    before = float(np.mean(src @ ev._unit(tgt.mean(0))))
    after = float(np.mean(align(src) @ ev._unit(tgt.mean(0))))
    own = float(np.mean(tgt @ ev._unit(tgt.mean(0))))
    assert after > before and abs(after - own) < abs(before - own) / 2
    assert np.allclose(np.linalg.norm(align(src), axis=1), 1.0, atol=1e-5)


def test_commercial_config_loads():
    from train import load_config

    cfg = load_config(ROOT / "configs" / "commercial.yaml", {})
    assert cfg["adapter"]["num_tokens"] == 16 and cfg["train"]["p_sketch"] == 0.3
    assert cfg["val"]["sketch_prompt"] and cfg["train"]["batch_size"] == 8
