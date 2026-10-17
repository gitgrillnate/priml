"""Nearest-level coding: each value becomes the index of its nearest level.

A quantizer is a sorted vector of reconstruction LEVELS. A value's nearest level
is found against the midpoints between neighbours; a value on a midpoint takes
the lower cell. :mod:`~priml.math.quantization.lloyd_max` and
:mod:`~priml.math.quantization.fits` choose the levels.
"""

from __future__ import annotations

from torch import Tensor

import torch


def midpoints(levels: Tensor) -> Tensor:
    """Return the nearest-level decision boundaries between sorted levels.

    Args:
      levels: ``[..., L]`` sorted levels.

    Returns:
      thresholds: ``[..., L - 1]`` midpoints, in ``levels``' dtype.

    """
    # Halving is exact, so this rounds once like ``(a + b) / 2``, without overflowing
    # when the sum of two finite levels would.
    return levels[..., 1:] / 2 + levels[..., :-1] / 2


def quantize(values: Tensor, thresholds: Tensor) -> Tensor:
    """Return each value's nearest-level index, one threshold row per row.

    A value equal to a threshold takes the lower cell, the convention
    :func:`~priml.math.quantization.lloyd_max.lloyd_max` fits with. Values past
    the outer thresholds saturate to the first or last level.

    Args:
      values: ``[R, M]`` finite values; a NaN codes as the last index, since
        checking values here would synchronize the device on every call.
      thresholds: ``[R, L - 1]`` sorted boundaries, one row per row of values.

    Returns:
      indices: ``[R, M]`` int64 in ``[0, L - 1]``.

    """
    return torch.searchsorted(
        thresholds.to(values.device).contiguous(),
        values.contiguous(),
    )


def dequantize(indices: Tensor, levels: Tensor) -> Tensor:
    """Return the level each index names, one level row per row.

    Args:
      indices: ``[R, M]`` integer indices.
      levels: ``[R, L]`` levels.

    Returns:
      values: ``[R, M]`` in ``levels``' dtype.

    """
    return torch.gather(levels.to(indices.device), dim=-1, index=indices.long())
