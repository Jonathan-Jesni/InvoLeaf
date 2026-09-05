"""InvoLeaf: involution inside the ViT encoder block.

Motivation
----------
A timm ViT block is::

    x = x + ls1(attn(norm1(x)))
    x = x + ls2(mlp(norm2(x)))

The MLP is applied to each token independently -- token 47 never sees token 48 --
so *all* spatial mixing in a ViT happens inside attention.

AgriTL-ViT (ESWA 2025) replaces that MLP with a module it describes as extracting
"hierarchical spatial features", but the module is six 1x1 convolutions and two 3x1
max-pools. A 1x1 convolution is a per-token linear map: structurally it is another MLP
and cannot mix spatial information. So roughly 3.5M parameters buy essentially no
learnable spatial mixing. We replace that slot with involution, which genuinely is a
spatial operator, is content-adaptive, and is cheap because kernels are shared across
channels.

The channel-mixing caveat
-------------------------
Involution is channel-*agnostic*: it mixes space but not channels. Since the FFN is the
block's main channel-mixing stage, swapping it out wholesale removes channel mixing
from the block. Hence three variants, of which only the first is a naive swap:

``pure``      x + Inv(norm2(x))
              The faithful "replace their module" swap. Expected to underperform; it is
              the ablation that demonstrates why channel mixing matters.

``inv_ffn``   x + FFN_r2(Inv(norm2(x)))                       <-- PRIMARY
              Involution supplies the spatial mixing, a narrowed FFN (ratio 2 instead
              of 4) retains channel mixing. Still ~1.8M parameters lighter per block
              than the baseline ratio-4 FFN.

``parallel``  x + FFN_r4(norm2(x)) + Inv(norm2(x))
              Capacity-added control, isolating "did involution help, or did more
              parameters help?"

All three are drop-in replacements for ``block.mlp``, because every variant is a
function of ``norm2(x)`` alone. Nothing else in the block is touched.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
from timm.layers import Mlp

from involeaf.ops.involution import Involution2d

VARIANTS = ("pure", "inv_ffn", "parallel")


def resolve_block_indices(spec: str, depth: int) -> list[int]:
    """Turn a placement spec into concrete block indices.

    ``all`` | ``lastN`` (e.g. ``last4``) | ``firstN`` | ``everyN`` (e.g. ``every3``) |
    ``none`` | an explicit comma list such as ``0,3,6,9``.
    """
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(depth))
    if spec == "none":
        return []
    if spec.startswith("last"):
        return list(range(max(0, depth - int(spec[4:])), depth))
    if spec.startswith("first"):
        return list(range(min(depth, int(spec[5:]))))
    if spec.startswith("every"):
        return list(range(0, depth, int(spec[5:])))
    return [int(t) for t in spec.split(",") if t.strip() != ""]


class InvoLeafFFN(nn.Module):
    """Drop-in replacement for ``timm`` ``Block.mlp``.

    Consumes and returns ``(B, N, C)`` where ``N = num_prefix_tokens + H*W``. Prefix
    tokens (CLS, and register tokens on some timm ViTs) carry no spatial position and
    therefore cannot participate in a spatial operator; they bypass the involution.
    """

    def __init__(
        self,
        dim: int,
        variant: str = "inv_ffn",
        num_prefix_tokens: int = 1,
        kernel_size: int = 7,
        group_channels: int = 16,
        reduction_ratio: int = 4,
        norm_layer: str = "gn",
        impl: str = "shift",
        delta_init: bool = True,
        ffn_ratio: float = 2.0,
        cls_mode: str = "identity",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if cls_mode not in ("identity", "linear"):
            raise ValueError(f"unknown cls_mode {cls_mode!r}")

        self.dim = dim
        self.variant = variant
        self.num_prefix_tokens = num_prefix_tokens
        self.cls_mode = cls_mode

        self.involution = Involution2d(
            channels=dim,
            kernel_size=kernel_size,
            group_channels=group_channels,
            reduction_ratio=reduction_ratio,
            norm_layer=norm_layer,
            impl=impl,
            delta_init=delta_init,
        )

        if variant == "pure":
            self.ffn = None
        elif variant == "inv_ffn":
            self.ffn = Mlp(dim, hidden_features=int(dim * ffn_ratio), drop=drop)
        else:  # parallel
            self.ffn = Mlp(dim, hidden_features=int(dim * 4), drop=drop)

        # Everything in this module is freshly initialised inside a pretrained
        # encoder, so it needs its own learning rate (see new_parameter_names).
        self._involeaf_new = True

        # Prefix tokens skip the spatial op. "linear" gives them a parallel channel-mix
        # so they are not simply frozen relative to the patch tokens.
        self.cls_proj = nn.Linear(dim, dim) if cls_mode == "linear" else nn.Identity()
        if cls_mode == "linear":
            nn.init.zeros_(self.cls_proj.weight)
            nn.init.zeros_(self.cls_proj.bias)

    @staticmethod
    def _grid_hw(num_patches: int) -> tuple[int, int]:
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(
                f"expected a square token grid, got {num_patches} patch tokens. "
                "Non-square inputs are not supported."
            )
        return side, side

    def _spatial(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, P, C) -> involution -> (B, P, C)."""
        b, p, c = patches.shape
        h, w = self._grid_hw(p)
        grid = patches.transpose(1, 2).reshape(b, c, h, w)
        grid = self.involution(grid)
        return grid.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_pre = self.num_prefix_tokens
        prefix, patches = x[:, :n_pre], x[:, n_pre:]

        spatial = self._spatial(patches)
        if self.variant == "pure":
            patches_out = spatial
        elif self.variant == "inv_ffn":
            patches_out = self.ffn(spatial)
        else:  # parallel
            patches_out = self.ffn(patches) + spatial

        if n_pre == 0:
            return patches_out

        if self.variant == "parallel":
            prefix_out = self.ffn(prefix)
        elif self.variant == "inv_ffn":
            # The prefix token skips involution but still gets the channel mix, so the
            # CLS token stays in the same representational space as the patch tokens.
            prefix_out = self.ffn(prefix)
        else:
            prefix_out = self.cls_proj(prefix)

        return torch.cat([prefix_out, patches_out], dim=1)


@torch.no_grad()
def init_ffn_from_pretrained(new: Mlp, old: Mlp, mode: str = "slice") -> None:
    """Seed a narrowed FFN from the pretrained wide one.

    The replacement FFN is narrower than the pretrained MLP (ratio 2 vs 4). Rather than
    discard the pretrained weights, take the first ``hidden_new`` units of the hidden
    layer. ``slice_rescale`` additionally scales fc2 to compensate for summing half as
    many terms; whether that helps is an ablation, not an assumption.
    """
    if mode == "random":
        return
    if mode not in ("slice", "slice_rescale"):
        raise ValueError(f"unknown ffn_init {mode!r}")

    h_new = new.fc1.out_features
    h_old = old.fc1.out_features
    if h_new > h_old:
        raise ValueError(f"cannot slice {h_old} pretrained units down to {h_new}")

    new.fc1.weight.copy_(old.fc1.weight[:h_new])
    new.fc1.bias.copy_(old.fc1.bias[:h_new])
    new.fc2.weight.copy_(old.fc2.weight[:, :h_new])
    new.fc2.bias.copy_(old.fc2.bias)
    if mode == "slice_rescale":
        new.fc2.weight.mul_(h_old / h_new)


def build_involeaf(
    backbone: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    num_classes: int = 10,
    pretrained: bool = True,
    variant: str = "inv_ffn",
    inv_blocks: str = "all",
    kernel_size: int = 7,
    group_channels: int = 16,
    reduction_ratio: int = 4,
    norm_layer: str = "gn",
    impl: str = "shift",
    delta_init: bool = True,
    ffn_ratio: float = 2.0,
    ffn_init: str = "slice",
    cls_mode: str = "identity",
    dynamic_img_size: bool = True,
    **timm_kwargs,
) -> nn.Module:
    """Build a pretrained ViT and swap the FFN of selected blocks for InvoLeafFFN.

    ``dynamic_img_size=True`` lets the same weights be evaluated at 128 and 200 pixels
    for the resolution study; timm interpolates the position embedding.
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        dynamic_img_size=dynamic_img_size,
        **timm_kwargs,
    )
    depth = len(model.blocks)
    targets = resolve_block_indices(inv_blocks, depth)

    for i in targets:
        block = model.blocks[i]
        old_mlp = block.mlp
        new_mlp = InvoLeafFFN(
            dim=model.embed_dim,
            variant=variant,
            num_prefix_tokens=model.num_prefix_tokens,
            kernel_size=kernel_size,
            group_channels=group_channels,
            reduction_ratio=reduction_ratio,
            norm_layer=norm_layer,
            impl=impl,
            delta_init=delta_init,
            ffn_ratio=ffn_ratio,
            cls_mode=cls_mode,
        )
        if pretrained and new_mlp.ffn is not None and isinstance(old_mlp, Mlp):
            init_ffn_from_pretrained(new_mlp.ffn, old_mlp, mode=ffn_init)
        block.mlp = new_mlp

    model.involeaf_meta = {
        "backbone": backbone,
        "variant": variant,
        "inv_blocks": inv_blocks,
        "inv_block_indices": targets,
        "kernel_size": kernel_size,
        "group_channels": group_channels,
        "reduction_ratio": reduction_ratio,
        "norm_layer": norm_layer,
        "impl": impl,
        "ffn_ratio": ffn_ratio,
        "ffn_init": ffn_init,
        "cls_mode": cls_mode,
    }
    return model


def involution_modules(model: nn.Module) -> list[tuple[str, Involution2d]]:
    """All involution operators in the model, for kernel visualisation."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, Involution2d)]


def mark_new(module: nn.Module) -> nn.Module:
    """Tag a module as freshly initialised inside a pretrained encoder.

    Any architecture that performs surgery must mark what it created, so that
    ``new_parameter_names`` can give those parameters their own learning rate. This is
    deliberately not an isinstance check: the AgriTL-ViT reproduction introduces new
    modules too, and if it did not receive the same layer-wise treatment its freshly
    initialised weights would train at the low pretrained rate while ours trained at
    10x. That would handicap the baseline and rig the comparison in our favour.
    """
    module._involeaf_new = True
    return module


def new_parameter_names(model: nn.Module) -> set[str]:
    """Parameters introduced by surgery, so they can get their own learning rate.

    Newly created modules are randomly initialised while the surrounding encoder is
    pretrained, so they need a higher LR than the pretrained blocks. Applies equally to
    InvoLeaf and to the AgriTL-ViT reproduction.
    """
    names: set[str] = set()
    for mod_name, mod in model.named_modules():
        if getattr(mod, "_involeaf_new", False):
            prefix = f"{mod_name}." if mod_name else ""
            names.update(prefix + p for p, _ in mod.named_parameters())
    return names
