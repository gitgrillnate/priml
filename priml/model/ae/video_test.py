"""CausalVideoVAE: the model around a family's halves, on Wan 2.1's halves at test size."""

from __future__ import annotations

import pytest
import torch

from priml.model.ae import wan
from priml.model.ae.custom_types import Autoencoder
from priml.model.ae.video import CausalVideoVAE


def tiny() -> CausalVideoVAE.Config:
    """Return Wan 2.1's halves at widths 2 and 4: 2x in time, 4x in space."""
    return CausalVideoVAE.Config(
        encoder=wan.Encoder.Config(
            channels_hidden=(2, 4, 4),
            temporal=(False, True),
            blocks_per_stage=1,
            channels_latent=3,
            checkpoint=None,
        ),
        decoder=wan.Decoder.Config(
            channels_hidden=(4, 4, 2),
            channels_upsampled=(2, 2),
            temporal=(True, False),
            blocks_per_stage=1,
            channels_latent=3,
            checkpoint=None,
        ),
        num_frames=5,
        image_size=(8, 12),
    )


def test_satisfies_the_autoencoder_protocol() -> None:
    assert isinstance(tiny().make(), Autoencoder)


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    clip = torch.randint(
        0,
        256,
        (2, 4, 3, 5, 8, 12),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    flat = model.encode(clip.flatten(0, 1)).mode()
    nested = model.encode(clip).mode()
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == clip.shape


@pytest.mark.parametrize(
    ("frames", "side"),
    [(6, 8), (5, 10)],
    ids=["frames", "sides"],
)
def test_encode_refuses_a_clip_the_resamplers_cannot_divide(
    frames: int,
    side: int,
) -> None:
    clip = torch.zeros(2, 3, frames, side, 12, dtype=torch.uint8)
    with pytest.raises(ValueError, match="1 \\+ 2k frames of sides divisible by 4"):
        _ = tiny().make().encode(clip)


def test_encode_refuses_a_float_clip() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 5, 8, 12))


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encode_only.make().decode(torch.zeros(2, 3, 2, 2, 3))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decode_only.make().encode(torch.zeros(2, 3, 5, 8, 12, dtype=torch.uint8))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert isinstance(config.decoder, wan.Decoder.Config)
    config.decoder.channels_latent = 5
    with pytest.raises(ValueError, match="channels_latent"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_factors() -> None:
    config = tiny()
    assert isinstance(config.decoder, wan.Decoder.Config)
    config.decoder.temporal = (True, True)
    with pytest.raises(ValueError, match="factors"):
        _ = config.make()


def test_config_needs_a_half() -> None:
    config = tiny()
    config.encoder = config.decoder = None
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.latent_shape()
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.make()


def test_cost_needs_both_halves() -> None:
    config = tiny()
    config.decoder = None
    with pytest.raises(ValueError, match="prices both halves"):
        _ = config.cost(batch_size=1, dtype=None)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
