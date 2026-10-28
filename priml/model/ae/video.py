"""Layers the video autoencoders share: a causal 3D convolution and a per-frame map."""

from __future__ import annotations

from typing import cast, override

from torch import Tensor, nn
from torch.nn import functional

from priml.model.conv import Conv3d


class CausalConv3d(Conv3d):
    """A biased ``Conv3d`` padded in time on its past side only.

    Time is padded by ``past`` frames before the first; each side by
    ``(kernel - 1) // 2``, both ways, so a stride-1 convolution keeps the grid.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        past: int = 2,
        padding_mode: str = "constant",
    ) -> None:
        """Build the convolution.

        Args:
          channels_in: Input channels.
          channels_out: Output channels.
          kernel_size: ``(time, height, width)`` extents.
          stride: ``(time, height, width)`` steps.
          past: Frames padded before the first.
          padding_mode: ``functional.pad``'s mode: ``"constant"`` pads zeros,
            ``"replicate"`` the edge.

        """
        super().__init__(
            Conv3d.Config(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ),
        )
        height, width = ((side - 1) // 2 for side in kernel_size[1:])
        self.causal = (width, width, height, height, past, 0)
        # Not ``padding_mode``: that is ``Conv3d``'s own, and anything but "zeros"
        # makes it pad again.
        self.pad_mode = padding_mode

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(
            functional.pad(input, self.causal, mode=self.pad_mode),
        )


def per_frame(module: nn.Module, x: Tensor) -> Tensor:
    """Apply a 2D ``module`` to every frame of ``[N, C, T, H, W]``.

    Args:
      module: Maps ``[N T, C, H, W]`` to ``[N T, C', H', W']``.
      x: Video.

    Returns:
      y: ``[N, C', T, H', W']``.

    """
    b, _, t = x.shape[:3]
    frames = cast(Tensor, module(x.transpose(1, 2).flatten(0, 1)))
    return frames.unflatten(0, (b, t)).transpose(1, 2)
