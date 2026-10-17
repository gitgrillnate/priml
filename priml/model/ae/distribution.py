"""Distributions an encoder returns: a diagonal Gaussian, or a point mass."""

from __future__ import annotations

from torch import Tensor

import torch


class DiagonalGaussian:
    """Axis-aligned Gaussian over codes; the log-variance is clamped to ``[-30, 20]``."""

    def __init__(self, mean: Tensor, logvar: Tensor) -> None:
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Draw ``mean + std * noise``; ``None`` reads the global generator."""
        return self.mean + self.std * torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
        )

    def mode(self) -> Tensor:
        """Return the mean."""
        return self.mean


class PointMass:
    """All mass on one code: what a deterministic encoder returns."""

    def __init__(self, value: Tensor) -> None:
        self.value = value

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Return the code; there is nothing to draw."""
        del generator
        return self.value

    def mode(self) -> Tensor:
        """Return the code."""
        return self.value
