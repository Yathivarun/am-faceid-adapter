"""Attach an AuraFace identity adapter to a frozen SD1.5 UNet, and save/load it
in h94's IP-Adapter-FaceID format so diffusers can load it natively.

Architecture (identical to h94 IP-Adapter-FaceID SD1.5):
    face vector (512) -> IPAdapterFaceIDImageProjection
        FeedForward 512 -> 1024 (GELU) -> 4*768, LayerNorm(768)  => 4 tokens
    tokens -> decoupled cross-attention: per cross-attention layer a new
        to_k_ip / to_v_ip (768 -> hidden), added to the text attention.

The UNet is patched exactly the way diffusers' `_load_ip_adapter_weights`
does (same processor classes, `MultiIPAdapterImageProjection`,
`encoder_hid_dim_type = "ip_image_proj"`), so a trained adapter behaves
the same in training and after `pipe.load_ip_adapter(...)`.

Saved format (safetensors, flat keys, what `load_ip_adapter` reads):
    image_proj.proj.0.{weight,bias}  image_proj.proj.2.{weight,bias}  image_proj.norm.{weight,bias}
    ip_adapter.{k}.to_k_ip.weight    ip_adapter.{k}.to_v_ip.weight     k = 1, 3, 5, ... per cross-attn layer
"""

from pathlib import Path

import torch
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
from diffusers.models.embeddings import IPAdapterFaceIDImageProjection, MultiIPAdapterImageProjection

ID_DIM = 512
NUM_TOKENS = 4
MULT = 2            # h94 FaceID SD1.5: 512 -> 1024 hidden


def _hidden_size(unet, name: str) -> int:
    cfg = unet.config
    if name.startswith("mid_block"):
        return cfg.block_out_channels[-1]
    if name.startswith("up_blocks"):
        return list(reversed(cfg.block_out_channels))[int(name[len("up_blocks.")])]
    if name.startswith("down_blocks"):
        return cfg.block_out_channels[int(name[len("down_blocks.")])]
    raise ValueError(name)


def attach_adapter(unet, num_tokens: int = NUM_TOKENS, id_dim: int = ID_DIM, mult: int = MULT,
                   init_from_text: bool = True) -> torch.nn.ModuleList:
    """Patch `unet` in place; returns the trainable modules (projection first,
    then one processor per cross-attention layer, in attn_processors order).

    init_from_text: start to_k_ip / to_v_ip as copies of the text to_k / to_v
    (the IP-Adapter initialisation; converges much faster than random)."""
    cad = unet.config.cross_attention_dim
    modules = dict(unet.named_modules())
    procs, trainable = {}, []
    for name in unet.attn_processors.keys():
        if name.endswith("attn1.processor"):
            procs[name] = AttnProcessor2_0()
            continue
        proc = IPAdapterAttnProcessor2_0(hidden_size=_hidden_size(unet, name), cross_attention_dim=cad,
                                         scale=1.0, num_tokens=[num_tokens])
        if init_from_text:
            attn = modules[name[: -len(".processor")]]
            with torch.no_grad():
                proc.to_k_ip[0].weight.copy_(attn.to_k.weight.float())
                proc.to_v_ip[0].weight.copy_(attn.to_v.weight.float())
        procs[name] = proc
        trainable.append(proc)
    unet.set_attn_processor(procs)
    proj = IPAdapterFaceIDImageProjection(image_embed_dim=id_dim, cross_attention_dim=cad,
                                          mult=mult, num_tokens=num_tokens)
    unet.encoder_hid_proj = MultiIPAdapterImageProjection([proj])
    unet.config.encoder_hid_dim_type = "ip_image_proj"
    return torch.nn.ModuleList([proj, *trainable])


def adapter_state_dict(unet) -> dict[str, torch.Tensor]:
    """The adapter weights, flattened in h94 FaceID key layout."""
    proj = unet.encoder_hid_proj.image_projection_layers[0]
    out = {}
    for k, v in proj.state_dict().items():
        k = k.replace("ff.net.0.proj", "proj.0").replace("ff.net.2", "proj.2")
        out[f"image_proj.{k}"] = v.detach().cpu().contiguous()
    key_id = 1
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, IPAdapterAttnProcessor2_0):
            out[f"ip_adapter.{key_id}.to_k_ip.weight"] = proc.to_k_ip[0].weight.detach().cpu().contiguous()
            out[f"ip_adapter.{key_id}.to_v_ip.weight"] = proc.to_v_ip[0].weight.detach().cpu().contiguous()
            key_id += 2
    return out


def save_adapter(unet, path: str | Path, dtype=torch.float16) -> Path:
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(dtype) for k, v in adapter_state_dict(unet).items()}, str(path))
    return path


def load_adapter(unet, path: str | Path) -> int:
    """Copy saved adapter weights (h94 FaceID layout) into an adapter already
    attached with `attach_adapter`: the warm start for a continued run. Shapes
    and key counts must match exactly; returns the number of tensors loaded."""
    from safetensors.torch import load_file
    sd = split_state_dict(load_file(str(path)))
    proj = unet.encoder_hid_proj.image_projection_layers[0]
    proj_sd = {k.replace("proj.0", "ff.net.0.proj").replace("proj.2", "ff.net.2"): v
               for k, v in sd["image_proj"].items()}
    proj.load_state_dict(proj_sd, strict=True)        # raises on a token-count / size mismatch
    procs = [p for p in unet.attn_processors.values() if isinstance(p, IPAdapterAttnProcessor2_0)]
    want = {f"{1 + 2 * i}.{kv}.weight" for i in range(len(procs)) for kv in ("to_k_ip", "to_v_ip")}
    if set(sd["ip_adapter"]) != want:
        raise ValueError(f"{path}: ip_adapter keys do not match this UNet "
                         f"({len(sd['ip_adapter'])} saved vs {len(want)} expected)")
    with torch.no_grad():
        for i, proc in enumerate(procs):
            for kv in ("to_k_ip", "to_v_ip"):
                w = sd["ip_adapter"][f"{1 + 2 * i}.{kv}.weight"]
                dst = getattr(proc, kv)[0].weight
                if w.shape != dst.shape:
                    raise ValueError(f"{path}: {kv} of layer {i} is {tuple(w.shape)}, expected {tuple(dst.shape)}")
                dst.copy_(w.to(dst.dtype))
    return len(proj_sd) + len(want)


def split_state_dict(flat: dict) -> dict:
    """Flat safetensors keys -> {"image_proj": {...}, "ip_adapter": {...}} (diffusers' in-memory form)."""
    sd = {"image_proj": {}, "ip_adapter": {}}
    for k, v in flat.items():
        head, rest = k.split(".", 1)
        sd[head][rest] = v
    return sd


def id_embeds(emb: torch.Tensor) -> list[torch.Tensor]:
    """Batch of face vectors (B, 512) -> the `image_embeds` list the patched UNet expects."""
    return [emb.unsqueeze(1)]
