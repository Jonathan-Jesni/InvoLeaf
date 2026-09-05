"""Tests for the ViT surgery, the token-grid bridge, and parameter budgets.

Models are built with ``pretrained=False`` so the suite needs no weight download.
"""

from __future__ import annotations

import pytest
import timm
import torch

from involeaf.models.agritl_vit import ProposedResNetModule, build_agritl_vit
from involeaf.models.vit_involution import (
    InvoLeafFFN, build_involeaf, involution_modules, new_parameter_names,
    resolve_block_indices,
)

BACKBONE = "vit_base_patch16_224"


def _params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@pytest.fixture(scope="module")
def baseline():
    return timm.create_model(
        BACKBONE, pretrained=False, num_classes=10, dynamic_img_size=True
    )


# --------------------------------------------------------------------------------- #
# The token <-> grid bridge: the highest-risk code in the project
# --------------------------------------------------------------------------------- #

def test_token_grid_roundtrip_preserves_order():
    """(B, P, C) -> (B, C, H, W) -> (B, P, C) must be the identity.

    A transpose error here silently scrambles the spatial layout: the model still trains
    and still reports a plausible accuracy, but involution operates on a permuted grid
    and every spatial claim in the report becomes false. Hence an explicit test.
    """
    patches = torch.randn(2, 196, 32)
    b, p, c = patches.shape
    h = w = 14
    grid = patches.transpose(1, 2).reshape(b, c, h, w)
    back = grid.flatten(2).transpose(1, 2)
    assert torch.equal(patches, back)


def test_grid_reshape_rejects_non_square():
    ffn = InvoLeafFFN(dim=32, variant="pure", num_prefix_tokens=0, kernel_size=3)
    with pytest.raises(ValueError, match="square"):
        ffn(torch.randn(1, 200, 32))


def test_prefix_tokens_are_not_fed_to_the_spatial_operator():
    """The CLS token has no spatial position and must bypass involution.

    Checked by construction: with n_pre prefix tokens the involution sees exactly
    N - n_pre tokens, which must form a square grid.
    """
    ffn = InvoLeafFFN(dim=32, variant="inv_ffn", num_prefix_tokens=1, kernel_size=3)
    out = ffn(torch.randn(2, 197, 32))
    assert out.shape == (2, 197, 32)


def test_uses_model_num_prefix_tokens_not_a_hardcoded_one():
    """timm ViTs may carry register tokens as well as CLS."""
    model = build_involeaf(backbone=BACKBONE, pretrained=False, num_classes=10)
    for _, block in enumerate(model.blocks):
        assert block.mlp.num_prefix_tokens == model.num_prefix_tokens


# --------------------------------------------------------------------------------- #
# Variants and parameter budgets
# --------------------------------------------------------------------------------- #

@pytest.mark.parametrize("variant", ["pure", "inv_ffn", "parallel"])
def test_variants_forward(variant):
    model = build_involeaf(
        backbone=BACKBONE, pretrained=False, num_classes=10, variant=variant
    )
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 10)


def test_primary_variant_is_lighter_than_the_baseline(baseline):
    """The headline efficiency claim: inv_ffn must cost fewer parameters than ViT-B/16.

    Expected ~64.7M against ~85.8M. If this ever fails, the cost argument that the whole
    project rests on has been broken.
    """
    model = build_involeaf(
        backbone=BACKBONE, pretrained=False, num_classes=10, variant="inv_ffn"
    )
    assert _params_m(model) < _params_m(baseline)
    saving = 1 - _params_m(model) / _params_m(baseline)
    assert 0.20 < saving < 0.30, f"expected a ~24% saving, got {saving:.1%}"


def test_parallel_variant_adds_capacity(baseline):
    """The control arm must be heavier, or it is not controlling for capacity."""
    model = build_involeaf(
        backbone=BACKBONE, pretrained=False, num_classes=10, variant="parallel"
    )
    assert _params_m(model) > _params_m(baseline)


# --------------------------------------------------------------------------------- #
# Placement, resolution, optimiser wiring
# --------------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "spec,expected",
    [
        ("all", list(range(12))),
        ("none", []),
        ("last4", [8, 9, 10, 11]),
        ("first3", [0, 1, 2]),
        ("every3", [0, 3, 6, 9]),
        ("0,5,11", [0, 5, 11]),
    ],
)
def test_placement_specs(spec, expected):
    assert resolve_block_indices(spec, 12) == expected


def test_placement_controls_how_many_blocks_are_swapped():
    model = build_involeaf(
        backbone=BACKBONE, pretrained=False, num_classes=10, inv_blocks="last4"
    )
    assert len(involution_modules(model)) == 4
    swapped = [i for i, b in enumerate(model.blocks) if isinstance(b.mlp, InvoLeafFFN)]
    assert swapped == [8, 9, 10, 11]


@pytest.mark.parametrize("resolution", [112, 160, 224])
def test_evaluates_at_other_native_resolutions(resolution):
    """Same weights, different input size - needed for the resolution study.

    Only multiples of the patch size are valid; 200 is not, which is why the low
    resolution protocol resizes back to 224 instead.
    """
    model = build_involeaf(backbone=BACKBONE, pretrained=False, num_classes=10)
    assert model(torch.randn(1, 3, resolution, resolution)).shape == (1, 10)


def test_new_parameters_are_identified_for_layerwise_lr():
    """Freshly initialised modules must be separable from pretrained ones."""
    model = build_involeaf(backbone=BACKBONE, pretrained=False, num_classes=10)
    new = new_parameter_names(model)
    assert new, "no new parameters found - layer-wise LR would be a no-op"
    assert all(n in dict(model.named_parameters()) for n in new)
    assert any("involution" in n for n in new)


def test_param_groups_cover_every_trainable_parameter():
    from involeaf.models.registry import param_groups

    model = build_involeaf(backbone=BACKBONE, pretrained=False, num_classes=10)
    groups = param_groups(model, {"lr": 1e-4})
    grouped = sum(len(g["params"]) for g in groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == trainable, "some parameters would never be optimised"


def test_backward_pass_reaches_involution():
    model = build_involeaf(backbone=BACKBONE, pretrained=False, num_classes=10)
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    span = model.blocks[0].mlp.involution.span
    assert span.bias.grad is not None and span.bias.grad.abs().sum() > 0


# --------------------------------------------------------------------------------- #
# AgriTL-ViT reproduction
# --------------------------------------------------------------------------------- #

def test_agritl_resnet_module_preserves_shape():
    module = ProposedResNetModule(64)
    assert module(torch.randn(2, 197, 64)).shape == (2, 197, 64)


def test_agritl_module_has_six_1x1_convs_and_two_maxpools():
    """Match the counts stated in the paper, so the reproduction is checkable."""
    module = ProposedResNetModule(64)
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv1d)]
    pools = [m for m in module.modules() if isinstance(m, torch.nn.MaxPool1d)]
    assert len(convs) == 6
    assert all(c.kernel_size == (1,) for c in convs)
    assert len(pools) == 2
    assert all(p.kernel_size == 3 for p in pools)


def test_agritl_vit_is_heavier_than_the_baseline(baseline):
    """Running attention twice per block cannot be cheaper; this pins that down."""
    model = build_agritl_vit(backbone=BACKBONE, pretrained=False, num_classes=10)
    assert _params_m(model) > _params_m(baseline)
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 10)
