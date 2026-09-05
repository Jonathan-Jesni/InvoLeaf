"""Involution: spatial-specific, channel-agnostic dynamic convolution.

Reimplementation of the operator from:
    Li et al., "Involution: Inverting the Inherence of Convolution for Visual
    Recognition", CVPR 2021.  https://github.com/d-li14/involution

We deliberately do NOT depend on the authors' repository. Its fast path requires
mmcv plus a CuPy CUDA kernel, neither of which survives this project's move from
CUDA (RTX 5050) to ROCm (Radeon PRO W7800) in October. Everything below is plain
PyTorch -- pad, slice, multiply, add -- and therefore portable to any backend.

Two forward implementations are provided and are numerically equivalent (enforced by
tests/test_involution.py):

unfold
    The reference formulation, mirroring the paper. Materialises a tensor of shape
    (B, C*K*K, H*W). At C=768, K=7, HW=196, batch 16 that is ~470 MB per block in
    fp16; across a 12-block ViT-B it needs ~5.6 GB of saved activations and will OOM
    an 8 GB card before ViT's own activations are counted.

shift (default)
    Accumulates over the K*K kernel offsets by slicing a padded, pre-reshaped view of
    the input. The slices are *views* into a single padded tensor, so although autograd
    saves K*K of them for the backward pass they all share one storage. Peak memory
    falls from O(B*C*K*K*H*W) to O(B*C*Hp*Wp) -- roughly 25 MB per block instead of
    470 MB in the configuration above. The cost is K*K small, launch-bound kernels per
    block rather than one large one, so it trades wall-clock for the ability to run
    K=7 at all on 8 GB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _largest_divisor_at_most(n: int, cap: int = 32) -> int:
    for g in range(min(cap, n), 0, -1):
        if n % g == 0:
            return g
    return 1


def _build_norm(kind: str, channels: int) -> nn.Module:
    """GroupNorm by default, not BatchNorm.

    The original operator uses BatchNorm2d in its reduce path. Inside a ViT the
    surrounding architecture is LayerNorm-based and we train at batch 8-16 on an 8 GB
    card, where BN batch statistics are noisy. Passing "bn" is kept available so the
    choice can be ablated rather than merely asserted.
    """
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(_largest_divisor_at_most(channels), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm_layer {kind!r}")


class Involution2d(nn.Module):
    """Involution over an NCHW feature map.

    Args:
        channels: input == output channel count.
        kernel_size: spatial extent K of the generated kernel.
        stride: spatial stride (>1 forces the unfold path).
        group_channels: channels sharing one kernel. Table 6b of the paper finds 16
            is the sweet spot -- halving cost for ~0.2% accuracy.
        reduction_ratio: bottleneck ratio of the kernel-generating MLP.
        norm_layer: "gn" | "bn" | "none".
        impl: "shift" | "unfold".
        delta_init: start the operator as an exact spatial identity (see below).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        group_channels: int = 16,
        reduction_ratio: int = 4,
        norm_layer: str = "gn",
        impl: str = "shift",
        delta_init: bool = True,
    ) -> None:
        super().__init__()
        if channels % group_channels != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by group_channels "
                f"({group_channels})"
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the kernel has a centre tap")
        if impl not in ("shift", "unfold"):
            raise ValueError(f"unknown impl {impl!r}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group_channels = group_channels
        self.groups = channels // group_channels
        self.padding = (kernel_size - 1) // 2
        self.impl = "unfold" if stride > 1 else impl

        reduced = max(channels // reduction_ratio, 1)
        self.reduce = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.norm = _build_norm(norm_layer, reduced)
        self.act = nn.ReLU(inplace=True)
        self.span = nn.Conv2d(reduced, self.groups * kernel_size**2, kernel_size=1)
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

        if delta_init:
            self.delta_init_()

    @torch.no_grad()
    def delta_init_(self) -> None:
        """Initialise so the operator is the identity map at step zero.

        The involution block is inserted into a *pretrained* encoder, so a random
        initialisation would destroy the pretrained signal and cost several epochs of
        recovery. Zeroing span.weight and setting its bias to a delta kernel (centre
        tap 1, all other taps 0) makes the generated kernel exactly the identity
        regardless of input, so the surrounding pretrained weights see undisturbed
        activations on the first forward pass.

        Gradients still reach span.weight -- its gradient depends on the incoming
        activation, not on its own value -- so the operator lifts away from the
        identity after the first optimiser step. This is the standard zero-init
        residual trick.
        """
        nn.init.zeros_(self.span.weight)
        bias = torch.zeros(self.groups, self.kernel_size**2)
        bias[:, self.kernel_size**2 // 2] = 1.0
        self.span.bias.copy_(bias.reshape(-1))

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the per-location kernels, shape (B, G, K*K, H_out, W_out).

        Exposed separately so the interpretability stage can visualise kernels
        directly, as in Section 4.3 of the paper.
        """
        w = self.span(self.act(self.norm(self.reduce(self.pool(x)))))
        b, _, h, w_out = w.shape
        return w.view(b, self.groups, self.kernel_size**2, h, w_out)

    def _forward_unfold(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        _, _, kk, h, w = weight.shape
        patches = F.unfold(
            x, self.kernel_size, padding=self.padding, stride=self.stride
        )
        patches = patches.view(b, self.groups, self.group_channels, kk, h, w)
        out = (weight.unsqueeze(2) * patches).sum(dim=3)
        return out.reshape(b, self.channels, h, w)

    def _forward_shift(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.padding
        # F.pad returns a contiguous tensor, so this view is legal and free. Slicing it
        # afterwards yields strided views that all share that one storage -- which is
        # the entire memory argument for this path.
        xp = F.pad(x, (p, p, p, p)).view(
            b, self.groups, self.group_channels, h + 2 * p, w + 2 * p
        )
        out = None
        for idx in range(self.kernel_size**2):
            i, j = divmod(idx, self.kernel_size)
            patch = xp[:, :, :, i : i + h, j : j + w]
            term = weight[:, :, idx].unsqueeze(2) * patch
            out = term if out is None else out + term
        return out.reshape(b, self.channels, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.kernel(x)
        if self.impl == "shift":
            return self._forward_shift(x, weight)
        return self._forward_unfold(x, weight)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, groups={self.groups}, "
            f"group_channels={self.group_channels}, impl={self.impl}"
        )
