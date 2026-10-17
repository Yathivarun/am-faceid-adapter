"""Stage A / eval v2 pieces: LR schedule, warm start, step naming, eval parsing and statistics."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval2 import cases, parse_mode, summarize  # noqa: E402
from faceid.model import adapter_state_dict, attach_adapter, load_adapter, save_adapter  # noqa: E402
from tests.test_model import tiny_unet  # noqa: E402
from train import lr_at, step_of  # noqa: E402

T = {"lr": 1e-4, "lr_warmup": 1000, "lr_decay_start": None, "lr_decay_steps": 5000, "lr_min": 0.0}


def test_warmup_then_constant():
    assert abs(lr_at(50000, T, 50000) - 1e-7) < 1e-12          # first update of the warm start
    assert abs(lr_at(50499, T, 50000) - 5e-5) < 1e-12
    assert lr_at(51000, T, 50000) == 1e-4
    assert lr_at(90000, T, 50000) == 1e-4
    assert lr_at(60000, T, None) == 1e-4                       # resumed: no warm-up


def test_cosine_cooldown():
    t = {**T, "lr_decay_start": 75000}
    assert lr_at(74999, t, None) == 1e-4
    assert lr_at(75000, t, None) == 1e-4
    assert abs(lr_at(77500, t, None) - 5e-5) < 1e-12
    assert lr_at(80000, t, None) == 0.0
    assert lr_at(85000, t, None) == 0.0


def test_step_of():
    assert step_of("checkpoints/poc/step-0050000/faceid_aura_sd15.safetensors") == 50000
    assert step_of("checkpoints/real_v3/state-0075000") == 75000
    assert step_of("weights/x.safetensors") == 0


def test_warm_start_roundtrip(tmp_path):
    a = tiny_unet()
    attach_adapter(a)
    with torch.no_grad():
        for p in a.encoder_hid_proj.parameters():
            p.add_(torch.randn_like(p))
    path = save_adapter(a, tmp_path / "w.safetensors", dtype=torch.float32)
    b = tiny_unet()
    attach_adapter(b)
    n = load_adapter(b, path)
    sa, sb = adapter_state_dict(a), adapter_state_dict(b)
    assert n == len(sa)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_warm_start_rejects_other_token_count(tmp_path):
    a = tiny_unet()
    attach_adapter(a, num_tokens=16)
    path = save_adapter(a, tmp_path / "w16.safetensors")
    b = tiny_unet()
    attach_adapter(b, num_tokens=4)
    try:
        load_adapter(b, path)
    except RuntimeError:
        return
    raise AssertionError("loading 16-token weights into a 4-token adapter must fail")


def test_parse_mode():
    assert parse_mode("ours", 0.8) == ("ours", ["ours"], [0.8])
    assert parse_mode("full-face@0.3", 0.8) == ("full-face", ["full-face"], [0.3])
    assert parse_mode("ours+plus-face@0.6,0.5", 0.8) == ("ours+plus-face", ["ours", "plus-face"], [0.6, 0.5])


def _evalset():
    def rec(p):
        return {"path": p, "flags": [], "box": [0, 0, 10, 10]}
    people = {}
    for i in "abc":
        people[i] = {"split": "dev", "protocols": {"reg2fru": True, "fru2fru": True},
                     "registration": [rec(f"{i}/r.jpg")], "fru": [rec(f"{i}/f{k}.jpg") for k in range(3)],
                     "usable": {"registration": [f"{i}/r.jpg"], "fru": [f"{i}/f{k}.jpg" for k in range(3)]}}
    people["d"] = {**people["a"], "split": "test"}
    return {"people": people}


def test_cases_never_score_against_the_input():
    for proto in ("reg2fru", "fru2fru"):
        for person, inp, _, targets, imp in cases(_evalset(), "dev", proto):
            assert inp not in targets and targets
            assert all(not t.startswith(person) for t in imp)
    assert len(cases(_evalset(), "dev", "reg2fru")) == 3
    assert len(cases(_evalset(), "dev", "fru2fru")) == 9


def test_summary_paired_difference():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(20):
        for mode, own in (("ours", 0.5), ("full-face", 0.4)):
            for prompt in ("photo", "sketch"):
                rows.append({"mode": mode, "person": str(p), "input": str(p), "prompt": prompt, "seed": 0,
                             "aura:own": own + rng.normal(0, 0.01), "aura:input": 0.6, "aura:impostor": 0.1})
    s = summarize(rows, baseline="full-face")
    d = s["ours"]["photo"]["aura"]["vs_full-face"]
    assert abs(d["margin_diff"] - 0.1) < 0.02
    assert d["ci95"][0] > 0 and d["p_signflip"] < 0.01 and d["wins"] == "20/20"
    assert s["ours"]["photo"]["aura"]["people"] == 20
