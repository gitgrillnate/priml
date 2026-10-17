"""Closed-form level choices: uniform, equal-probability, and moment-matched.

Each takes a 1-D sample and returns non-decreasing float64 levels to code it
with; :mod:`~priml.math.quantization.lloyd_max` fits the minimum-squared-error
levels iteratively instead.
"""

from __future__ import annotations

import math

from torch import Tensor

import torch


def uniform_levels(
    values: Tensor,
    num_levels: int,
    *,
    clip_sigmas: float | None = None,
) -> Tensor:
    """Return cell centres of an even partition of the sample's range.

    Args:
      values: 1-D sample.
      num_levels: Levels to place.
      clip_sigmas: Span ``mean +- clip_sigmas * std``; ``None`` spans min to max.

    Returns:
      levels: ``[num_levels]`` float64.

    Raises:
      ValueError: ``clip_sigmas`` is not positive and finite.

    """
    if clip_sigmas is not None and (
        math.isnan(clip_sigmas) or math.isinf(clip_sigmas) or clip_sigmas <= 0
    ):
        raise ValueError(f"clip_sigmas must be positive and finite; got {clip_sigmas}.")
    x = values.to(torch.float64)
    if clip_sigmas is None:
        low, high = x.min(), x.max()
    else:
        spread = clip_sigmas * x.std()
        low, high = x.mean() - spread, x.mean() + spread
    centres = torch.arange(num_levels, dtype=torch.float64, device=x.device) + 0.5
    return low + centres * (high - low) / num_levels


def quantile_levels(values: Tensor, num_levels: int) -> Tensor:
    """Return the ``(k + 1/2) / num_levels`` sample quantiles.

    Every index is then equally likely: index entropy is maximal, but the level
    density is ``p`` rather than the squared-error optimum ``p ** (1 / 3)``, so
    the tails are coarse.

    Args:
      values: 1-D sample.
      num_levels: Levels to place.

    Returns:
      levels: ``[num_levels]`` float64.

    """
    x = values.to(torch.float64).sort().values
    position = (
        torch.arange(num_levels, dtype=torch.float64, device=x.device) + 0.5
    ) / num_levels
    return x[(position * x.numel()).long().clamp(max=x.numel() - 1)]


def moment_matched_levels(values: Tensor, unit: Tensor) -> Tensor:
    """Return ``unit`` levels shifted to the sample's mean and scaled by its std.

    With ``unit`` the standard-normal Lloyd-Max table
    (:func:`~priml.math.quantization.lloyd_max.gaussian_levels`), this is the
    optimum for a Gaussian source of the sample's moments.

    Args:
      values: 1-D sample.
      unit: Levels for a zero-mean, unit-variance source.

    Returns:
      levels: ``unit``-shaped float64.

    """
    x = values.to(torch.float64)
    return x.mean() + x.std() * unit.to(device=x.device, dtype=torch.float64)
