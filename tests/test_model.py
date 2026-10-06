"""Adapter wiring and weight format, on a tiny UNet (CPU, no downloads)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceid.model import adapter_state_dict, attach_adapter, id_embeds, save_adapter, split_state_dict  # noqa: E402


def tiny_unet():
    from diffusers import UNet2DConditionModel
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
        attention_head_dim=8, norm_num_groups=8)


def run(unet, emb, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(2, 4, 16, 16)
    text = torch.randn(2, 5, 32)
    with torch.no_grad():
        return unet(x, torch.tensor([10, 500]), encoder_hidden_states=text,
                    added_cond_kwargs={"image_embeds": id_embeds(emb)}).sample


def test_projection_gives_four_tokens():
    unet = tiny_unet()
    modules = attach_adapter(unet, num_tokens=4, mult=2)
    proj = modules[0]
    assert proj(torch.randn(3, 512)).shape == (3, 4, 32)


def test_identity_changes_the_output():
    unet = tiny_unet()
    attach_adapter(unet)
    a, b = torch.nn.functional.normalize(torch.randn(2, 2, 512), dim=-1)
    assert not torch.allclose(run(unet, a), run(unet, b))


def test_saved_weights_use_h94_faceid_keys():
    unet = tiny_unet()
    attach_adapter(unet)
    sd = adapter_state_dict(unet)
    assert {"image_proj.proj.0.weight", "image_proj.proj.2.weight", "image_proj.norm.weight"} <= set(sd)
    ip = sorted(int(k.split(".")[1]) for k in sd if k.startswith("ip_adapter.") and k.endswith("to_k_ip.weight"))
    assert ip == list(range(1, 2 * len(ip), 2))        # 1, 3, 5, ... one per cross-attention layer


import pytest  # noqa: E402


@pytest.mark.parametrize("num_tokens", [4, 16])
def test_round_trip_through_the_diffusers_loader(tmp_path, num_tokens):
    """Weights saved by us, loaded by diffusers' own IP-Adapter loader, give the
    same output, for the 4-token research and the 16-token commercial adapter
    (diffusers infers the token count from proj.2)."""
    from safetensors.torch import load_file

    trained = tiny_unet()
    modules = attach_adapter(trained, num_tokens=num_tokens)
    with torch.no_grad():                             # make the adapter non-trivial
        for p in modules.parameters():
            p.add_(0.01 * torch.randn_like(p))
    path = save_adapter(trained, tmp_path / "a.safetensors", dtype=torch.float32)

    fresh = tiny_unet()
    fresh._load_ip_adapter_weights([split_state_dict(load_file(str(path)))])
    emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    assert torch.allclose(run(trained, emb), run(fresh, emb), atol=1e-5)
