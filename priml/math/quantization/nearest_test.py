"""Tests for nearest-level coding."""

from __future__ import annotations

import torch

from priml.math.quantization.nearest import dequantize, midpoints, quantize


def test_midpoints_of_finite_levels_never_overflow() -> None:
    levels = torch.tensor([[40_000.0, 50_000.0]], dtype=torch.float16)
    thresholds = midpoints(levels)
    assert bool(thresholds.isfinite().all())
    values = torch.tensor([[41_000.0, 49_000.0]], dtype=torch.float16)
    assert quantize(values, thresholds=thresholds).tolist() == [[0, 1]]


def test_ties_take_the_lower_cell_and_outliers_saturate() -> None:
    thresholds = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[-5.0, 0.0, 0.5, 1.0, 1.5, 9.0]])
    assert quantize(values, thresholds=thresholds).tolist() == [[0, 0, 1, 1, 2, 2]]


def test_quantize_compiles_without_reading_a_python_scalar() -> None:
    values = torch.tensor([[-1.0, 0.0, 1.0]])
    thresholds = torch.tensor([[0.0]])
    compiled = torch.compile(quantize, backend="eager", fullgraph=True)
    assert torch.equal(
        compiled(values, thresholds),
        quantize(values, thresholds=thresholds),
    )


def test_quantize_and_dequantize_apply_one_table_per_row() -> None:
    levels = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
    values = torch.tensor([[0.2, 1.9, 1.4], [11.0, 26.0, 99.0]])
    indices = quantize(values, thresholds=midpoints(levels))
    assert indices.tolist() == [[0, 2, 1], [0, 2, 2]]
    assert dequantize(indices, levels=levels).tolist() == [
        [0.0, 2.0, 1.0],
        [10.0, 30.0, 30.0],
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
