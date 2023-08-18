# Simple ViT from https://github.com/lucidrains/vit-pytorch

# from collections import OrderedDict
from pprint import pprint
# import re

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from shared.modules.factories import get_norm_factory

from ._registry import register_model


################################################################################
# Simple ViT implementation originally based off
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
# @3e5d1be
################################################################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_factory):
        super().__init__()
        self.net = nn.Sequential(
            norm_factory(normalized_shape=dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, norm_factory=nn.LayerNorm):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = norm_factory(normalized_shape=dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm_factory):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, norm_factory=norm_factory),
                FeedForward(dim, mlp_dim, norm_factory=norm_factory)
            ]))
        self.norm = norm_factory(normalized_shape=dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, norm_cfg=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        if norm_cfg is None:
            norm_factory = nn.LayerNorm
        else:
            norm_factory_core = get_norm_factory(norm_cfg)
            norm_factory = lambda normalized_shape: norm_factory_core(
                normalized_shape=normalized_shape,
                num_features=normalized_shape
            )

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            norm_factory(normalized_shape=patch_dim),
            nn.Linear(patch_dim, dim),
            norm_factory(normalized_shape=dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, norm_factory)

        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)


################################################################################
# /
################################################################################

@register_model
def cifar_simple_vit(pretrained=False, **kwargs):
    assert not pretrained
    # Remove extra TIMM arguments that we don't support
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('drop_rate', None)
    kwargs.setdefault('channels', kwargs.pop('in_chans', 3))
    kwargs = {
        "depth": 6,
        "heads": 16,
        "dim": 512,
        "mlp_dim": 1024,
        "image_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        **kwargs,
    }
    return SimpleViT(**kwargs)
