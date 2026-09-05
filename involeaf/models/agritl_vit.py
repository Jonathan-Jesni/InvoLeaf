"""Reproduction of AgriTL-ViT (Expert Systems with Applications, 2025).

Reproduced from the paper's textual description, because the published per-class
metrics cannot be relied on as a target: Tomato Early_blight is reported as precision
0.977 / recall 0.958 / F1 0.857 (an F1 below both of its inputs is impossible), the
Rice support column is identical to the Tomato one across datasets of different sizes
and class counts, Table 7's caption refers to peach leaf disease in a paper containing
no peach data, and accuracies are written as "0.985%". We therefore implement the
architecture as described, train it under our own protocol, and report our measured
number as the baseline. Any gap to their 98.5% is itself a result.

Documented assumptions
----------------------
The paper specifies the module only as "six 1x1 Conv1D layers, two 3x1 max-pools, a
dropout branch, all summed residually". That does not determine a unique wiring. Our
interpretation, which honours every stated count, is three parallel branches:

    branch A:  Conv1d(1x1) -> ReLU -> Conv1d(1x1) -> MaxPool1d(3x1)
    branch B:  Conv1d(1x1) -> ReLU -> Conv1d(1x1) -> MaxPool1d(3x1)
    branch C:  Conv1d(1x1) -> Dropout -> Conv1d(1x1)
    out     =  A + B + C + residual

That is six 1x1 Conv1D layers, two 3x1 max-pools, one dropout branch, summed
residually. Alternatives are possible; this one is stated so it can be checked.

Why this module is the gap we are filling
-----------------------------------------
A 1x1 Conv1D over the token axis is a per-token linear map -- structurally another MLP,
incapable of mixing spatial information. The only spatial interaction in the whole
module comes from the 3x1 max-pools, which are non-parametric, lossy, and operate on the
*flattened raster sequence*: they pool token 13 (end of row 0) with token 14 (start of
row 1), which are spatially distant in the image. So a module justified as extracting
"hierarchical spatial features" adds ~3.5M parameters and delivers no learnable spatial
mixing at all. That empty slot is what InvoLeaf fills with involution.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from involeaf.models.vit_involution import mark_new


class ProposedResNetModule(nn.Module):
    """The paper's replacement for the ViT MLP. Drop-in for ``Block.mlp``."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Freshly initialised inside a pretrained encoder: needs the same layer-wise
        # learning rate treatment InvoLeaf gets, or the baseline is handicapped.
        self._involeaf_new = True
        self.branch_a = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.branch_c = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, C) -> (B, C, N): Conv1d expects channels on dim 1.
        h = x.transpose(1, 2)
        out = self.branch_a(h) + self.branch_b(h) + self.branch_c(h) + h
        return out.transpose(1, 2)


class DualAttentionBlock(nn.Module):
    """A timm ViT block whose MHSA stage runs twice in sequence.

    The paper presents this as part of a "less computationally expensive" design, but
    running the most expensive component twice per layer roughly doubles its cost. Their
    own Table 10 reflects this: the proposed model is slower than their Modified ResNet
    variant on all three crops. Measuring it here rather than restating the claim is the
    point of reproducing it.
    """

    def __init__(self, block: nn.Module, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

        # Second attention stage, structurally identical, separately parameterised.
        # Marked new so it receives the same learning rate as InvoLeaf's new modules.
        self.norm1b = mark_new(nn.LayerNorm(dim, eps=getattr(block.norm1, "eps", 1e-6)))
        self.attn_b = mark_new(
            timm.models.vision_transformer.Attention(dim, num_heads=num_heads)
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path1(self.ls1(self.attn_b(self.norm1b(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def build_agritl_vit(
    backbone: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    num_classes: int = 10,
    pretrained: bool = True,
    dual_attention: bool = True,
    resnet_module: bool = True,
    dropout: float = 0.1,
    dynamic_img_size: bool = True,
    **timm_kwargs,
) -> nn.Module:
    """Build the AgriTL-ViT reproduction.

    ``dual_attention`` and ``resnet_module`` are separable so the paper's own ablation
    (their "Modified ResNet" variant) can be reproduced as well.
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        dynamic_img_size=dynamic_img_size,
        **timm_kwargs,
    )
    dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads

    for i, block in enumerate(model.blocks):
        if resnet_module:
            block.mlp = ProposedResNetModule(dim, dropout=dropout)
        if dual_attention:
            model.blocks[i] = DualAttentionBlock(block, dim, num_heads)

    model.involeaf_meta = {
        "backbone": backbone,
        "architecture": "agritl_vit",
        "dual_attention": dual_attention,
        "resnet_module": resnet_module,
        "dropout": dropout,
        "reproduction_note": (
            "Reproduced from the textual description; module wiring is an "
            "interpretation documented in involeaf/models/agritl_vit.py."
        ),
    }
    return model
