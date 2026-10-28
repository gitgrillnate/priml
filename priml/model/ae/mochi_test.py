"""Mochi: the published architecture, its checkpoint names, and the reference's bits.

``testdata/mochi_reference.pt`` holds the reference at ``15c9bad`` built small
through its own arguments -- width 2, two stages, time downsampled 3x -- with
its GroupNorms and attention heads, hardcoded at 32, built at 2: its weights,
a clip, and its own moments and unclamped pixels, minted under
``host_agnostic_numerics``; its forward code ran unmodified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import pytest
import torch

from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.latent_norm import StdLatents
from priml.model.ae.mochi import MochiVAE, mochi, mochi_comfy, remap_mochi_state_dict
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden, mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"

_REFERENCE_COMMIT: Final = "15c9bad662df1fa3e7b53136423695b286484c08"


def tiny() -> MochiVAE.Config:
    """Return the published layout at the reference golden's size: widths 2 and 4."""
    config = mochi()
    assert config.encoder is not None
    assert config.decoder is not None
    encoder, decoder = config.encoder, config.decoder
    encoder.channels_hidden, decoder.channels_hidden = (2, 4), (4, 2)
    encoder.blocks_per_stage = decoder.blocks_per_stage = (1, 1, 1)
    encoder.attention = (False, True, True)
    encoder.temporal = decoder.temporal = (3,)
    encoder.spatial = decoder.spatial = (2,)
    encoder.channels_head = 2
    for half in (encoder, decoder):
        half.channels_latent = 3
        half.num_groups = 2
        half.checkpoint = None
    config.num_frames = 4
    config.image_size = (4, 6)
    return config


def _clip(*, frames: int = 4) -> Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(
        0,
        256,
        (2, 3, frames, 4, 6),
        generator=generator,
        dtype=torch.uint8,
    )


def _keys(listing: str, prefix: str = "") -> dict[str, tuple[int, ...]]:
    lines = (_TESTDATA / listing).read_text().splitlines()
    return {
        prefix + name: tuple(int(n) for n in shape.split("x"))
        for name, shape in (line.split() for line in lines)
    }


def _replay_reference(*, nudge: bool = False) -> list[str]:
    """Return how the port, loading the reference's tensors, misses its bits."""
    reference = read_tensors(_TESTDATA / "mochi_reference.pt")
    commit = reference.pop("source_commit").numpy().tobytes().decode()
    assert commit == _REFERENCE_COMMIT
    model = tiny().make()
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        state = {
            key.removeprefix(f"state.{half}."): value
            for key, value in reference.items()
            if key.startswith(f"state.{half}.")
        }
        module.load_state_dict(remap_mochi_state_dict(state, half=half))
    assert model.decoder is not None
    if nudge:
        bias = model.decoder.output_proj.bias
        assert isinstance(bias, Tensor)
        bias[0] = torch.nextafter(bias[0], torch.tensor(torch.inf))
    # Batch, frames, and sides tie at 2 and 4: the reference's sizes at the golden
    # ceiling, and its own bits catch any transposed axis.
    image = reference["image"]
    assert reference["decoded"].abs().max() < 1
    with host_agnostic_numerics():
        moments = (
            model.encoder.forward(image.float() / 127.5 - 1) if model.encoder else None
        )
        mean = model.encode(image).mode()
        decoded = model.decoder(mean)
        clamped = model.decode(mean)
    assert moments is not None
    expected = {
        "moments": reference["moments"],
        "decoded": reference["decoded"],
        "clamped": ((reference["decoded"] + 1) / 2).clamp(0, 1),
    }
    actual = {"moments": moments, "decoded": decoded, "clamped": clamped}
    return mismatches(expected, actual=actual)


def test_matches_the_reference_implementation() -> None:
    report = _replay_reference()
    assert not report, report


def test_the_reference_replay_sees_one_ulp() -> None:
    report = _replay_reference(nudge=True)
    assert [line.split(":")[0] for line in report] == ["clamped", "decoded"]


@pytest.mark.parametrize("comfy", [False, True], ids=["released", "comfy"])
def test_published_halves_load_exactly_the_published_tensors(comfy: bool) -> None:
    """Each released half's names and shapes are its state, strictly.

    Comfy's one file holds both halves, each under its prefix.
    """
    config = mochi_comfy() if comfy else mochi()
    assert config.encoder is not None
    assert config.decoder is not None
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert isinstance(config.decoder.checkpoint, HubFile.Config)
    config.encoder.checkpoint = config.decoder.checkpoint = None
    with torch.device("meta"):
        model = config.make()
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        published = _keys(f"mochi_{half}_keys.txt", f"{half}." if comfy else "")
        with torch.device("meta"):
            state = {key: torch.empty(shape) for key, shape in published.items()}
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_mochi_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected


def test_published_latent_and_statistics() -> None:
    config = mochi()
    assert config.latent_shape() == (12, 28, 60, 106)
    norm = config.latent_norm
    assert isinstance(norm, StdLatents.Config)
    assert len(norm.mean) == len(norm.std) == 12


def test_round_trip_shapes() -> None:
    config = tiny()
    model = config.make()
    clip = _clip()
    posterior = model.encode(clip)
    assert isinstance(posterior, DiagonalGaussian)
    assert posterior.mode().shape == (2, *config.latent_shape())
    video = model.decode(posterior.mode())
    assert video.shape == clip.shape
    assert video.min() >= 0
    assert video.max() <= 1


def test_the_first_frame_encodes_as_the_image_it_is() -> None:
    """Causal in time: no later frame reaches the first latent frame."""
    model = tiny().make()
    clip = _clip(frames=7)
    with host_agnostic_numerics():
        whole = model.encode(clip).mode()
        first = model.encode(clip[:, :, :1]).mode()
        decoded = model.decode(whole)
        decoded_first = model.decode(whole[:, :, :1])
    assert torch.equal(whole[:, :, :1], first)
    assert torch.equal(decoded[:, :, :1], decoded_first)


def test_both_halves_load_their_own_files(tmp_path: Path) -> None:
    source = tiny().make()
    config = tiny()
    for half, module in (("encoder", source.encoder), ("decoder", source.decoder)):
        assert module is not None
        state = {key: value.contiguous() for key, value in module.state_dict().items()}
        save_file(state, str(tmp_path / f"{half}.safetensors"))
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "encoder.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "decoder.safetensors")
    loaded = config.make()
    clip = _clip()
    latent = source.encode(clip).mode()
    assert torch.equal(loaded.encode(clip).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    clip = torch.randint(
        0,
        256,
        (2, 5, 3, 4, 4, 6),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    flat = model.encode(clip.flatten(0, 1)).mode()
    nested = model.encode(clip).mode()
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == clip.shape


@pytest.mark.parametrize(("frames", "side"), [(5, 4), (4, 5)], ids=["frames", "sides"])
def test_encode_refuses_a_clip_the_resamplers_cannot_divide(
    frames: int,
    side: int,
) -> None:
    clip = torch.zeros(2, 3, frames, side, 6, dtype=torch.uint8)
    with pytest.raises(ValueError, match="frames of sides divisible by 2"):
        _ = tiny().make().encode(clip)


def test_encode_refuses_a_float_clip() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 4, 4, 6))


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encode_only.make().decode(torch.zeros(2, 3, 2, 2, 3))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decode_only.make().encode(_clip())


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_latent = 5
    with pytest.raises(ValueError, match="channels_latent"):
        _ = config.make()


def test_rejects_per_stage_settings_that_miss_a_stage() -> None:
    encoder_short, decoder_short = tiny(), tiny()
    assert encoder_short.encoder is not None
    assert decoder_short.decoder is not None
    encoder_short.encoder.attention = (False, True)
    decoder_short.decoder.blocks_per_stage = (1, 1)
    for config in (encoder_short, decoder_short):
        with pytest.raises(ValueError, match="2 stages need"):
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


def _round_trip(module: nn.Module, clip: Tensor) -> Tensor:
    assert isinstance(module, MochiVAE)
    return module.decode(module.encode(clip).sample())


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    config = tiny()
    # Torch's FLOP counter has no entry for CPU flash attention; its math kernel
    # counts the same products as ``bmm``.
    with sdpa_kernel(SDPBackend.MATH):
        analytical = assert_cost_matches_torch(
            config,
            build_input=_clip,
            run=_round_trip,
            batch_size=2,
            dtype=None,
        )
    assert analytical["adjoint"].sum() == 0
    assert analytical.params == sum(p.numel() for p in config.make().parameters())


def test_published_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="mochi_config", config=mochi())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
