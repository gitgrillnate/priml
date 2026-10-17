"""Both distributions satisfy the protocol; the Gaussian draws reproducibly."""

from __future__ import annotations

import torch

from priml.model.ae.custom_types import Distribution
from priml.model.ae.distribution import DiagonalGaussian, PointMass


def _moments() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 4, 5, generator=generator), torch.randn(
        2,
        3,
        4,
        5,
        generator=generator,
    )


def test_both_satisfy_the_protocol() -> None:
    mean, logvar = _moments()
    assert isinstance(DiagonalGaussian(mean, logvar), Distribution)
    assert isinstance(PointMass(mean), Distribution)


def test_gaussian_mode_is_the_mean_and_std_is_half_logvar_exponentiated() -> None:
    mean, logvar = _moments()
    gaussian = DiagonalGaussian(mean, logvar)
    assert gaussian.mode() is mean
    assert torch.equal(gaussian.std, torch.exp(0.5 * logvar))


def test_gaussian_clamps_the_log_variance() -> None:
    mean, _ = _moments()
    gaussian = DiagonalGaussian(mean, torch.full_like(mean, 100.0))
    assert torch.equal(gaussian.logvar, torch.full_like(mean, 20.0))


def test_gaussian_draw_is_mean_plus_std_times_the_generator_noise() -> None:
    mean, logvar = _moments()
    gaussian = DiagonalGaussian(mean, logvar)
    draw = gaussian.sample(generator=torch.Generator().manual_seed(3))
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(3))
    assert torch.equal(draw, mean + gaussian.std * noise)
    assert not torch.equal(draw, gaussian.mode())


def test_point_mass_returns_its_value_either_way() -> None:
    mean, _ = _moments()
    point = PointMass(mean)
    assert point.mode() is mean
    assert point.sample(generator=torch.Generator().manual_seed(1)) is mean


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
