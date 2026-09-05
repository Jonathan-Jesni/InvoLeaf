"""Tests for the involution operator itself."""

from __future__ import annotations

import pytest
import torch

from involeaf.ops.involution import Involution2d


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_shift_and_unfold_are_equivalent(kernel_size):
    """The memory-efficient path must compute exactly the reference formulation.

    This is the test that lets us use the shift path everywhere: it is an optimisation,
    not a different operator. Run in float64 so the comparison is not masked by
    floating-point slack.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 128, 14, 14, dtype=torch.float64)

    shift = Involution2d(128, kernel_size=kernel_size, delta_init=False).double().eval()
    unfold = Involution2d(
        128, kernel_size=kernel_size, impl="unfold", delta_init=False
    ).double().eval()
    unfold.load_state_dict(shift.state_dict())

    assert torch.allclose(shift(x), unfold(x), atol=1e-10)


def test_delta_init_is_exact_identity():
    """A freshly built operator must not perturb a pretrained encoder at step zero."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, 14, 14, dtype=torch.float64)
    op = Involution2d(64, kernel_size=7, delta_init=True).double().eval()
    assert torch.allclose(op(x), x, atol=1e-12)


def test_delta_init_unblocks_after_one_step():
    """Zero-initialising span.weight must not permanently starve the reduce path.

    At step 0 the gradient reaching ``reduce`` is exactly zero because span.weight is
    zero. After one optimiser step span.weight is non-zero and gradient flows. If this
    ever regressed, the operator would train only its bias and silently stay an identity.
    """
    torch.manual_seed(0)
    op = Involution2d(64, kernel_size=3)
    opt = torch.optim.SGD(op.parameters(), lr=1e-3)
    x = torch.randn(2, 64, 8, 8)

    op(x).sum().backward()
    assert op.reduce.weight.grad.abs().sum().item() == 0.0
    opt.step()

    op.zero_grad(set_to_none=True)
    op(x).sum().backward()
    assert op.reduce.weight.grad.abs().sum().item() > 0.0


def test_gradients_reach_every_parameter():
    op = Involution2d(64, kernel_size=3, delta_init=False)
    op(torch.randn(2, 64, 8, 8)).sum().backward()
    for name, p in op.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_output_shape_is_preserved():
    op = Involution2d(96, kernel_size=5, group_channels=16)
    out = op(torch.randn(3, 96, 14, 14))
    assert out.shape == (3, 96, 14, 14)


def test_kernel_shape_matches_groups():
    """The interpretability stage depends on this layout."""
    op = Involution2d(768, kernel_size=7, group_channels=16)
    k = op.kernel(torch.randn(2, 768, 14, 14))
    assert k.shape == (2, 768 // 16, 49, 14, 14)


def test_channel_agnostic_within_a_group():
    """Channels inside one group must share a kernel - that is the whole operator.

    Verified structurally: the generated kernel has one entry per group, not per channel.
    """
    op = Involution2d(64, kernel_size=3, group_channels=16)
    assert op.groups == 4
    assert op.span.out_channels == 4 * 9


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        Involution2d(100, group_channels=16)          # not divisible
    with pytest.raises(ValueError):
        Involution2d(64, kernel_size=4)               # even kernel has no centre tap
    with pytest.raises(ValueError):
        Involution2d(64, impl="magic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_shift_path_uses_far_less_memory():
    """The claim that makes K=7 trainable on 8 GB.

    The unfold path materialises B*C*K*K*H*W; the shift path slices views of one padded
    tensor. Anything under a 4x saving means the view-sharing has been broken, most
    likely by someone replacing a slice with a copy.
    """
    device = torch.device("cuda")
    peaks = {}
    for impl in ("unfold", "shift"):
        blocks = torch.nn.Sequential(
            *[Involution2d(768, kernel_size=7, impl=impl).to(device) for _ in range(4)]
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(8, 768, 14, 14, device=device)
        blocks(x).sum().backward()
        peaks[impl] = torch.cuda.max_memory_allocated(device)
        del blocks, x
        torch.cuda.empty_cache()

    assert peaks["shift"] * 4 < peaks["unfold"], (
        f"expected the shift path to save >4x memory, got "
        f"{peaks['unfold'] / peaks['shift']:.1f}x"
    )
