"""The protocols accept what they describe and refuse what lacks a method."""

from __future__ import annotations

from torch import Tensor

import torch

from priml.model.ae.custom_types import Autoencoder, Decoder, Distribution, Encoder
from priml.model.ae.distribution import PointMass


class _Identity:
    def encode(self, x: Tensor, /) -> PointMass:
        return PointMass(x)

    def decode(self, z: Tensor, /) -> Tensor:
        return z


class _EncodeOnly:
    def encode(self, x: Tensor, /) -> PointMass:
        return PointMass(x)


def test_a_full_autoencoder_satisfies_every_protocol() -> None:
    model = _Identity()
    assert isinstance(model, Autoencoder)
    assert isinstance(model, Encoder)
    assert isinstance(model, Decoder)
    x = torch.randn(2, 3)
    code = model.encode(x)
    assert isinstance(code, Distribution)
    assert torch.equal(model.decode(code.mode()), x)


def test_an_encoder_alone_is_not_an_autoencoder() -> None:
    model = _EncodeOnly()
    assert isinstance(model, Encoder)
    assert not isinstance(model, Decoder)
    assert not isinstance(model, Autoencoder)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
