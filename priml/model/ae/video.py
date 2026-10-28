"""Causal video KL autoencoders: the model around a family's halves, and their shared layers.

A family -- ``wan``, ``mochi`` -- supplies an encoder and a decoder;
:class:`CausalVideoVAE` maps uint8 clips to the posterior and latents back to
RGB around them. Every convolution is causal in time, so a clip is ``1 + k``
times the time factor in frames, and its first frame encodes as the image it is.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, map_cost, traffic
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.latent_norm import LatentNormalizer, ScaleLatents
from priml.model.conv import Conv3d, conv_cost
from priml.model.custom_types import ChannelsIn, ChannelsOut


if TYPE_CHECKING:
    from priml.math.custom_types import TensorFn


class LatentFactors(Protocol):
    """A half's latent width and its resampling between pixels and latent."""

    channels_latent: int

    def factors(self) -> tuple[int, int]:
        """Return the ``(time, space)`` downsampling from pixels to latent."""
        ...


class VideoEncoderConfig(Makeable[nn.Module], ChannelsIn, LatentFactors, Protocol):
    """Builds ``[N, C_in, T, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, t, h, w]`` moments."""


class VideoDecoderConfig(Makeable[nn.Module], ChannelsOut, LatentFactors, Protocol):
    """Builds ``[N, C, t, h, w]`` latents to ``[N, C_out, T, H, W]`` pixels in ``[-1, 1]``."""


class CausalVideoVAE(nn.Module):
    """A causal video KL autoencoder as an ``Autoencoder``; either half may be left out.

    ``encode`` returns a diagonal Gaussian; call ``sample()`` or ``mode()`` on
    it. The model is frozen and always in evaluation mode.
    """

    class Config(Fig["CausalVideoVAE"]):
        """The halves, the clip geometry, and the published latent normalizer."""

        encoder: VideoEncoderConfig | None = None
        """Pixels to posterior; ``None`` builds and loads no encoder."""

        decoder: VideoDecoderConfig | None = None
        """Latents to pixels; ``None`` builds and loads no decoder."""

        num_frames: int = 1
        """Frames in the clip the latent shape and the cost are stated at."""

        image_size: tuple[int, int] = (256, 256)
        """``(height, width)`` of those frames."""

        latent_norm: Makeable[LatentNormalizer] = field(
            default_factory=ScaleLatents.Config,
        )
        """The statistics the family's diffusion models train with; the autoencoder never applies them."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            half = self.encoder or self.decoder
            if half is None:
                raise ValueError("CausalVideoVAE.Config needs an encoder or a decoder.")
            return half.factors()

        def latent_shape(self) -> tuple[int, int, int, int]:
            """Return ``(channels_latent, frames, height, width)`` of the clip's latent."""
            time, space = self.factors()
            half = self.encoder or self.decoder
            assert half is not None
            height, width = self.image_size
            frames = 1 + (self.num_frames - 1) // time
            return half.channels_latent, frames, height // space, width // space

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one ``decode(encode(clip).sample())`` round trip of the stated clip.

            Forward only; the parameters are still owned.

            Args:
              batch_size: Clips in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to both halves.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            if self.encoder is None or self.decoder is None:
                raise ValueError(
                    "CausalVideoVAE.Config.cost prices both halves; one is None.",
                )
            channels, frames, height, width = self.latent_shape()
            clip = (self.num_frames, *self.image_size)
            pixels_in = batch_size * self.encoder.channels_in * math.prod(clip)
            pixels_out = batch_size * self.decoder.channels_out * math.prod(clip)
            latents = batch_size * channels * frames * height * width
            f32 = torch.float32
            full = (
                cost(
                    self.encoder,
                    **{
                        **kwargs,
                        "input_grid": clip,
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                + cost(
                    self.decoder,
                    **{
                        **kwargs,
                        "input_grid": (frames, height, width),
                        "batch_size": batch_size,
                        "dtype": dtype,
                    },
                )
                # ``clip.float() / 127.5 - 1`` in.
                + traffic(
                    "primal",
                    kernel="elementwise",
                    elements=pixels_in,
                    dtype=torch.uint8,
                )
                + traffic("primal", kernel="elementwise", elements=pixels_in, dtype=f32)
                + _POINTWISE(channels=pixels_in, dtype=f32).tile(2)
                # Clamp the log-variance; ``exp(0.5 * logvar)``.
                + _CLAMP(channels=latents, dtype=dtype)
                + _POINTWISE(channels=latents, dtype=dtype).tile(2)
                # ``mean + std * noise``: the noise written, a product, a sum.
                + traffic("primal", kernel="elementwise", elements=latents, dtype=dtype)
                + _SUM(channels=latents, dtype=dtype).tile(2)
                # ``(x + 1) / 2`` and a two-sided clamp out.
                + _POINTWISE(channels=pixels_out, dtype=dtype).tile(2)
                + _CLAMP(channels=pixels_out, dtype=dtype)
            )
            return Cost(
                cells=full.only("primal").cells,
                params=full.params,
                params_active=full.params_active,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        halves = [half for half in (config.encoder, config.decoder) if half is not None]
        if not halves:
            raise ValueError("CausalVideoVAE needs an encoder or a decoder.")
        if len({half.channels_latent for half in halves}) != 1:
            raise ValueError("The encoder and decoder disagree on channels_latent.")
        if len({half.factors() for half in halves}) != 1:
            raise ValueError(
                "The encoder and decoder disagree on their (time, space) factors, so "
                "the decoder would not return the clip.",
            )
        self.time, self.space = config.factors()
        self.encoder = None if config.encoder is None else config.encoder.make()
        self.decoder = None if config.decoder is None else config.decoder.make()
        self.eval()
        self.requires_grad_(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Stay in evaluation mode: the published weights are frozen."""
        del mode
        return super().train(False)

    def encode(self, x: Tensor, /) -> DiagonalGaussian:
        """Return the posterior over latents.

        Args:
          x: ``[..., 3, T, H, W]`` uint8 RGB, ``T = 1 + time k`` frames.

        Returns:
          posterior: Diagonal Gaussian over ``[..., channels_latent, t, h, w]``.

        Raises:
          TypeError: ``x`` is not uint8.
          ValueError: The model was built without an encoder, or the clip's
            frames or sides do not divide.

        """
        if self.encoder is None:
            raise ValueError("This CausalVideoVAE was built without an encoder.")
        if x.dtype != torch.uint8:
            raise TypeError(
                f"CausalVideoVAE encodes uint8 RGB in [0, 255]; got {x.dtype}.",
            )
        frames, height, width = x.shape[-3:]
        if (frames - 1) % self.time or height % self.space or width % self.space:
            raise ValueError(
                f"CausalVideoVAE encodes 1 + {self.time}k frames of sides divisible by "
                f"{self.space}; got {frames} frames of {height}x{width}.",
            )
        # ``x / 127.5 - 1``, the references' own map, as ``kl`` maps uint8 pixels.
        pixels = x.reshape(-1, *x.shape[-4:]).float() / 127.5 - 1
        moments = self.encoder(pixels)
        moments = moments.reshape(*x.shape[:-4], *moments.shape[1:])
        mean, logvar = moments.chunk(2, dim=-4)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: Tensor, /) -> Tensor:
        """Decode latents to RGB in ``[0, 1]``.

        Args:
          z: ``[..., channels_latent, t, h, w]`` raw latents.

        Returns:
          video: ``[..., 3, 1 + time (t - 1), H, W]`` float RGB, clamped.

        Raises:
          ValueError: The model was built without a decoder.

        """
        if self.decoder is None:
            raise ValueError("This CausalVideoVAE was built without a decoder.")
        decoded = cast(Tensor, self.decoder(z.reshape(-1, *z.shape[-4:])))
        decoded = decoded.reshape(*z.shape[:-4], *decoded.shape[1:])
        return ((decoded + 1) / 2).clamp(0, 1)


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


def causal_conv3d_cost(
    channels_in: int,
    channels_out: int,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
    *,
    stride: tuple[int, int, int] = (1, 1, 1),
    past: int = 2,
    input_grid: tuple[int, int, int],
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost :class:`CausalConv3d`: the pad's copy, then the unpadded convolution.

    Args:
      channels_in: Input channels.
      channels_out: Output channels.
      kernel_size: ``(time, height, width)`` extents.
      stride: ``(time, height, width)`` steps.
      past: Frames padded before the first.
      input_grid: ``(frames, height, width)`` in.
      batch_size: Clips in this invocation.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: Forward and adjoint.

    """
    t, h, w = input_grid
    height, width = ((side - 1) // 2 for side in kernel_size[1:])
    padded = (t + past, h + 2 * height, w + 2 * width)
    moved = batch_size * channels_in * (math.prod(input_grid) + math.prod(padded))
    return traffic(
        "primal",
        kernel="elementwise",
        elements=moved,
        dtype=dtype,
    ) + conv_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        kernel_size=kernel_size,
        ndim=3,
        groups=1,
        bias=True,
        input_grid=padded,
        batch_size=batch_size,
        stride=stride,
        padding=0,
        dtype=dtype,
    )


def per_frame(fn: TensorFn, x: Tensor) -> Tensor:
    """Apply a 2D map to every frame of ``[N, C, T, H, W]``.

    Args:
      fn: Maps ``[N T, C, H, W]`` to ``[N T, C', H', W']``.
      x: Video.

    Returns:
      y: ``[N, C', T, H', W']``.

    """
    b, _, t = x.shape[:3]
    frames = fn(x.transpose(1, 2).flatten(0, 1))
    return frames.unflatten(0, (b, t)).transpose(1, 2)


_POINTWISE: Final = map_cost(primal=1, adjoint=0, adjoint_inputs=0, adjoint_outputs=0)
"""A forward elementwise op with one scalar operand."""

_SUM: Final = map_cost(
    primal=1,
    adjoint=0,
    inputs=2,
    adjoint_inputs=0,
    adjoint_outputs=0,
)
"""A forward sum or product of two tensors."""

_CLAMP: Final = map_cost(primal=2, adjoint=0, adjoint_inputs=0, adjoint_outputs=0)
"""A forward two-sided clamp."""
