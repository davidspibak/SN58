"""The training forward of `vector_v1.1` on cached camera features, and its loss.

It is `Net.encode_prompt` and `Net.predict` up to the U-Net's condition, batched over prompts of
different lengths: the padding is masked out of the decoder's cross-attention, so each sample's
condition is the one inference computes for it alone. The loss is the scheduler's training
objective (epsilon prediction): noise added to the normalized HORIZON actions at a random one of its
50 training timesteps, one per sample, and the U-Net's squared error in predicting that noise.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vector_runtime.net import CAMERAS, HISTORY, SINKS, TOKENS, WIDTH

from .batches import Batch
from .episodes import PROPRIO_SLICES


def tokens(net: Any, feats: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    """(B, T, cameras, 768) features and (B, T, 20) raw proprioception -> (B, T, TOKENS, 768):
    `Net.tokens`, cameras then proprioception."""
    out = [feats[:, :, c].float() for c in range(len(CAMERAS))]
    for key, (a, b) in PROPRIO_SLICES.items():
        out.append(net.proprio[key](net.norm[key](prop[..., a:b])).float())
    return torch.stack(out, dim=2)


def memory(
    net: Any, feats: torch.Tensor, prop: torch.Tensor, act: torch.Tensor, length: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of prompts -> the decoder's memory (B, SINKS + P, 768) and its key padding mask
    (True where a prompt has ended): `Net.encode_prompt`."""
    tok = tokens(net, feats, prop)
    b, p = tok.shape[:2]
    a = net.action_proj(net.norm["action"](act).reshape(b, p, -1)).unsqueeze(2).float()
    pooled = net.prompt_pool(torch.cat([tok, a], dim=2).reshape(b * p, TOKENS + 1, WIDTH))
    pooled = pooled.float().reshape(b, p, WIDTH) + net.prompt_pos[:, :p, :]
    mem = torch.cat([net.sinks.expand(b, -1, -1), pooled], dim=1)
    pad = torch.arange(SINKS + p, device=mem.device)[None, :] >= (SINKS + length)[:, None]
    return mem, pad


def decoder_layer(layer: Any, x: torch.Tensor, mem: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """`DecoderLayer.forward` with a key padding mask on the cross-attention."""
    x = x + layer._sa_block(layer.norm1(x), None, None, False)
    y, _ = layer.multihead_attn(
        layer.norm2(x), mem, mem, key_padding_mask=pad, need_weights=False
    )
    x = x + layer.dropout2(y)
    return x + layer._ff_block(layer.norm3(x))


def condition(
    net: Any, feats: torch.Tensor, prop: torch.Tensor, mem: torch.Tensor, pad: torch.Tensor
) -> torch.Tensor:
    """The HISTORY frames and the prompt memory -> the U-Net's condition (B, 2 * 18 * 768):
    `Net.predict` before it samples."""
    now = tokens(net, feats, prop)
    b = now.shape[0]
    now = now.reshape(b, HISTORY * TOKENS, WIDTH)
    x = now + net.obs_pos
    for layer in net.decoder["layers"]:
        x = decoder_layer(layer, x, mem, pad)
    return torch.cat([now, x.float()], dim=1).reshape(b, -1)


def unet(u: Any, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """`Unet.forward` with one timestep per sample: `t` of shape (B,)."""
    x = x.transpose(1, 2)
    cond = torch.cat([u.diffusion_step_encoder(t), cond], dim=-1)
    skips = []
    for res, res2, down in u.down_modules:
        x = res2(res(x, cond), cond)
        skips.append(x)
        x = down(x)
    for mid in u.mid_modules:
        x = mid(x, cond)
    for res, res2, up in u.up_modules:
        x = up(res2(res(torch.cat((x, skips.pop()), dim=1), cond), cond))
    return u.final_conv(x).transpose(1, 2)


def loss(net: Any, batch: Batch, generator: torch.Generator | None = None) -> torch.Tensor:
    """Each sample's diffusion loss, (B,) float32."""
    mem, pad = memory(
        net, batch.prompt_feats, batch.prompt_prop, batch.prompt_act, batch.prompt_len
    )
    cond = condition(net, batch.hist_feats, batch.hist_prop, mem, pad)
    clean = net.norm["action"](batch.target)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    t = torch.randint(
        0,
        net.scheduler.config.num_train_timesteps,
        (len(batch),),
        generator=generator,
        device=clean.device,
    )
    pred = unet(net.unet, net.scheduler.add_noise(clean, noise, t), t, cond)
    return F.mse_loss(pred.float(), noise, reduction="none").mean(dim=(1, 2))
