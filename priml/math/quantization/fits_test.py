"""Tests for the closed-form level choices."""

from __future__ import annotations

import math

import pytest
import torch

from priml.math.quantization.fits import (
    moment_matched_levels,
    quantile_levels,
    uniform_levels,
)


def test_uniform_levels_are_cell_centres_of_the_range() -> None:
    values = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    levels = uniform_levels(values, num_levels=4)
    assert levels.tolist() == pytest.approx([-0.75, -0.25, 0.25, 0.75])


def test_uniform_levels_can_span_a_clipped_range() -> None:
    values = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    spread = float(values.std())
    levels = uniform_levels(values, num_levels=2, clip_sigmas=0.5)
    assert levels.tolist() == pytest.approx([-spread / 4, spread / 4])


@pytest.mark.parametrize("clip_sigmas", [0.0, -1.0, math.nan, math.inf])
def test_uniform_levels_refuse_a_clip_that_is_not_positive_and_finite(
    clip_sigmas: float,
) -> None:
    with pytest.raises(ValueError, match="clip_sigmas"):
        _ = uniform_levels(torch.zeros(2), num_levels=2, clip_sigmas=clip_sigmas)


def test_quantile_levels_sit_at_equal_probability() -> None:
    values = torch.arange(16, dtype=torch.float64)
    assert quantile_levels(values, num_levels=4).tolist() == [2.0, 6.0, 10.0, 14.0]


def test_moment_matched_levels_shift_and_scale_a_unit_table() -> None:
    values = torch.tensor([1.0, 3.0], dtype=torch.float64)
    unit = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    spread = float(values.std())
    levels = moment_matched_levels(values, unit=unit)
    assert levels.tolist() == pytest.approx([2 - spread, 2.0, 2 + spread])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
