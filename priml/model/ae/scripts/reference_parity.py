#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prove each vision autoencoder port bit-identical to the reference it ports.

For INVAE, RAE, and VTP: clones the reference at its pinned commit, imports it
UNMODIFIED, gives both sides one set of weights, and runs both encode/decode
paths on the same inputs. Following ``docs/SKILL.bit-for-bit.md``:

1. Inventory. Every function, method, and class in the reference files on the
   reconstruction path is PAIRED with the port code that mirrors it, or NOT
   PORTED with a reason. A definition in neither table fails the run.
2. Randomness. ``torch.randn`` is pinned on both sides to the same stored
   draws, and each call site is counted; every other seeded random op raises,
   a native kernel's included.
3. Bisection. Forward hooks record every submodule's outputs under its module
   name on both sides, and every name both sides share is compared with
   ``torch.equal``. All mismatches are collected into one report.
4. Coverage. ``coverage`` measures branches over both sides. Every paired
   function must run, no not-ported one may, and every branch in paired code
   must go both ways unless an allowlist names it with a reason.
5. Goldens. ``--mint`` writes ``testdata/rae_reference.pt`` and
   ``testdata/vtp_reference.pt`` from the reference side. Without it, the
   run re-derives both and requires the checked-in bits.

Weights are compared too: each port half loads the reference's tensors through
its published-name remap, and every tensor must match in bits.

Exactly these things are supplied or changed on the reference side:

- RAE: its Hugging Face encoder is switched to eager attention through HF's
  own ``set_attn_implementation``, the kernel the port fixes.
- INVAE: REG's ``models/invae.py`` imports ``dictdot`` for one return wrapper;
  a stub supplies a dict with attribute access.
- VTP: ``vtp/__init__`` imports the CLIP side's dependencies, so ``vtp`` and
  ``vtp.models`` are registered as bare namespace packages and only the
  reconstruction modules import. The bfloat16 case autocasts on the CPU where
  the reference's evaluation autocasts on CUDA.

INVAE has no checked-in reference golden: REG hardcodes 32 GroupNorm groups,
so the smallest model it builds is far over the 32,768-byte golden ceiling.
This run is its reference proof; ``testdata/invae.pt`` guards the port.

Examples:
  reference_parity.py
  reference_parity.py --model rae vtp --mint

'''
# fmt: on

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import (
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import types

from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from treekle import json_loads

import torch

from priml.hub import get_cache_dir
from priml.model.ae.checkpoint import LocalFile, read_state
from priml.model.ae.invae import (
    INVAE,
    Decoder as InvaeDecoder,
    Encoder as InvaeEncoder,
    remap_hf_state_dict as remap_invae_state_dict,
)
from priml.model.ae.latent_norm import ElementwiseLatentStats
from priml.model.ae.rae import (
    RAE,
    Decoder as RaeDecoder,
    Encoder as RaeEncoder,
    remap_decoder_state_dict,
    remap_encoder_state_dict,
)
from priml.model.ae.vtp import (
    VTP,
    Decoder as VtpDecoder,
    Encoder as VtpEncoder,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    mismatches,
    read_tensors,
    write_tensors,
)


if TYPE_CHECKING:
    from safetensors.torch import save_file
    from torch._ops import OpOverload
    from torchvision.transforms import Normalize
    from torchvision.transforms.functional import to_tensor
    from transformers.models.dinov2_with_registers import (
        configuration_dinov2_with_registers,
        modeling_dinov2_with_registers,
    )

    import coverage
else:
    from wrapt import lazy_import

    # The branch measurement and the reference-side tools are what running the
    # comparison needs; importing this module needs none of them, and the
    # optional ``coverage`` and ``transformers`` are absent from a plain install.
    coverage = lazy_import("coverage")
    configuration_dinov2_with_registers = lazy_import(
        "transformers.models.dinov2_with_registers.configuration_dinov2_with_registers",
    )
    modeling_dinov2_with_registers = lazy_import(
        "transformers.models.dinov2_with_registers.modeling_dinov2_with_registers",
    )
    save_file = lazy_import("safetensors.torch", "save_file")
    Normalize = lazy_import("torchvision.transforms", "Normalize")
    to_tensor = lazy_import("torchvision.transforms.functional", "to_tensor")


_THIS: Final = Path(__file__).resolve()
_PRIML: Final = _THIS.parents[3]
_TESTDATA: Final = _PRIML / "model" / "ae" / "testdata"


@dataclass(frozen=True, slots=True, kw_only=True)
class Reference:
    """One reference repository, pinned, and its inventory against the port.

    Keys are ``"<file>::<Qualified.name>"``: reference files relative to the
    clone, port files relative to the ``priml`` package.
    """

    url: str
    """Repository to clone."""

    commit: str
    """The revision this comparison is against."""

    files: tuple[str, ...]
    """Reference files on the reconstruction path; every definition is inventoried."""

    paired: Mapping[str, tuple[str, ...]]
    """Reference definition -> the port definitions that mirror it."""

    not_ported: Mapping[str, str]
    """Reference definition (a class covers its methods) -> why nothing mirrors it."""

    reference_unreached: Mapping[str, str]
    """Reference branches no input drives both ways -> why.

    Keyed ``"<file>::<Qualified.name>::<source line>"``: the definition is part of
    the key, so an entry never silences the same line in another function.
    """

    port_unreached: Mapping[str, str]
    """The same, for branches in the paired port code."""

    extra_outputs: Mapping[str, str]
    """Module names whose calls return more tensors on one side -> why."""

    uncompared: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    """Module names whose outputs differ in form and are compared elsewhere -> why."""

    reference_calls: Mapping[str, tuple[tuple[int, ...], str]] = dataclasses.field(
        default_factory=dict[str, tuple[tuple[int, ...], str]],
    )
    """Module names the reference calls more often -> (calls the port mirrors, why)."""


_INVAE: Final = Reference(
    url="https://github.com/SwayStar123/REG.git",
    commit="3c51606c801dd9e87ee9ef778782766ab7c379ca",
    files=("models/invae.py",),
    paired={
        "models/invae.py::nonlinearity": ("model/ae/invae.py::_swish",),
        "models/invae.py::Normalize": ("model/ae/invae.py::_norm",),
        **{
            f"models/invae.py::{theirs}": (f"model/ae/invae.py::{ours}",)
            for theirs, ours in (
                ("Upsample", "_Upsample"),
                ("Upsample.__init__", "_Upsample.__init__"),
                ("Upsample.forward", "_Upsample.forward"),
                ("Downsample", "_Downsample"),
                ("Downsample.__init__", "_Downsample.__init__"),
                ("Downsample.forward", "_Downsample.forward"),
                ("ResnetBlock", "_Resnet"),
                ("ResnetBlock.__init__", "_Resnet.__init__"),
                ("ResnetBlock.forward", "_Resnet.forward"),
                ("AttnBlock", "_Attention"),
                ("AttnBlock.__init__", "_Attention.__init__"),
                ("AttnBlock.forward", "_Attention.forward"),
                ("Encoder", "Encoder"),
                ("Encoder.__init__", "Encoder.__init__"),
                ("Encoder.forward", "Encoder.forward"),
                ("Decoder", "Decoder"),
                ("Decoder.__init__", "Decoder.__init__"),
                ("Decoder.forward", "Decoder.forward"),
            )
        },
        "models/invae.py::DiagonalGaussianDistribution": (
            "model/ae/distribution.py::DiagonalGaussian",
        ),
        "models/invae.py::DiagonalGaussianDistribution.__init__": (
            "model/ae/distribution.py::DiagonalGaussian.__init__",
        ),
        "models/invae.py::DiagonalGaussianDistribution.sample": (
            "model/ae/distribution.py::DiagonalGaussian.sample",
        ),
        "models/invae.py::DiagonalGaussianDistribution.mode": (
            "model/ae/distribution.py::DiagonalGaussian.mode",
        ),
        "models/invae.py::AutoencoderKL": ("model/ae/invae.py::INVAE",),
        "models/invae.py::AutoencoderKL.__init__": (
            "model/ae/invae.py::INVAE.__init__",
            "model/ae/invae.py::_conv",
            "model/ae/invae.py::_Stage.__init__",
            "model/ae/invae.py::_Middle.__init__",
        ),
        "models/invae.py::AutoencoderKL.encode": (
            "model/ae/invae.py::INVAE.encode",
            "model/ae/invae.py::_Stage.forward",
            "model/ae/invae.py::_Middle.forward",
        ),
        "models/invae.py::AutoencoderKL.decode": ("model/ae/invae.py::INVAE.decode",),
        "models/invae.py::VAE_F16D32": ("model/ae/invae.py::INVAE.Config",),
    },
    not_ported={
        "models/invae.py::DiagonalGaussianDistribution.kl": (
            "A training loss; nothing on the encode/decode path calls it."
        ),
        "models/invae.py::DiagonalGaussianDistribution.nll": (
            "A training loss; nothing on the encode/decode path calls it."
        ),
        "models/invae.py::AutoencoderKL.forward": (
            "The training round trip: encode, sample, decode, each paired above."
        ),
        "models/invae.py::VAE_F8D4": (
            "The f8d4 geometry is Encoder/Decoder.Config values; no checkpoint of "
            "it is published."
        ),
    },
    reference_unreached={
        "models/invae.py::Upsample.__init__::if self.with_conv:": "Every published model resamples with a convolution.",
        "models/invae.py::Upsample.forward::if self.with_conv:": "Every published model resamples with a convolution.",
        "models/invae.py::Downsample.__init__::if self.with_conv:": "Every published model resamples with a convolution.",
        "models/invae.py::Downsample.forward::if self.with_conv:": "Every published model resamples with a convolution.",
        "models/invae.py::ResnetBlock.forward::if temb is not None:": "INVAE has no time embedding.",
        "models/invae.py::ResnetBlock.__init__::if temb_channels > 0:": "INVAE has no time embedding.",
        "models/invae.py::ResnetBlock.__init__::if self.use_conv_shortcut:": "The encoder and decoder build every ResnetBlock without it.",
        "models/invae.py::ResnetBlock.forward::if self.use_conv_shortcut:": "The encoder and decoder build every ResnetBlock without it.",
        "models/invae.py::DiagonalGaussianDistribution.__init__::if self.deterministic:": "AutoencoderKL.encode never builds a deterministic posterior.",
        "models/invae.py::AutoencoderKL.encode::if not self.use_variational:": "VAE_F16D32 is variational.",
        "models/invae.py::Decoder.forward::if self.give_pre_end:": "Never set by AutoencoderKL.",
    },
    port_unreached={
        "model/ae/invae.py::INVAE.encode::if self.encoder is None:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::INVAE.encode::if x.dtype != torch.uint8:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::INVAE.decode::if self.decoder is None:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::INVAE.__init__::if not halves:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::INVAE.__init__::if config.image_size % downsample:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::INVAE.__init__::if len({half.channels_latent for half in halves}) != 1:": "An error guard; invae_test drives it.",
        "model/ae/invae.py::Encoder.__init__::if config.checkpoint is not None:": "The comparison loads both halves through the remap; invae_test loads checkpoints.",
        "model/ae/invae.py::Decoder.__init__::if config.checkpoint is not None:": "The comparison loads both halves through the remap; invae_test loads checkpoints.",
    },
    extra_outputs={},
)


_RAE_DECODER: Final = "src/stage1/decoders/decoder.py"
_RAE_MAE_ENCODER: Final = (
    "The MAE encoder's; GeneralDecoder builds only the decoder half."
)
_GUARD: Final = "An error guard; the model's own test drives it."
_LOADED: Final = (
    "The comparison loads every half from a checkpoint; the model's own test "
    "builds without one."
)

_RAE: Final = Reference(
    url="https://github.com/bytetriper/RAE.git",
    commit="a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041",
    files=(
        "src/stage1/rae.py",
        "src/stage1/encoders/dinov2.py",
        _RAE_DECODER,
        "src/stage1/decoders/utils.py",
    ),
    paired={
        "src/stage1/rae.py::RAE": ("model/ae/rae.py::RAE",),
        "src/stage1/rae.py::RAE.__init__": ("model/ae/rae.py::RAE.__init__",),
        "src/stage1/rae.py::RAE.encode": (
            "model/ae/rae.py::RAE.encode",
            "model/ae/latent_norm.py::ElementwiseLatentStats.__init__",
            "model/ae/latent_norm.py::ElementwiseLatentStats.normalize",
            "math/pixel.py::rgb2float",
        ),
        "src/stage1/rae.py::RAE.decode": (
            "model/ae/rae.py::RAE.decode",
            "model/ae/latent_norm.py::ElementwiseLatentStats.denormalize",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm": ("model/ae/rae.py::Encoder",),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.__init__": (
            "model/ae/rae.py::Encoder.__init__",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.dinov2_forward": (
            "model/ae/rae.py::Encoder.forward",
            "model/ae/rae.py::Encoder._positions",
        ),
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.forward": (
            "model/ae/rae.py::Encoder.forward",
        ),
        **{
            f"{_RAE_DECODER}::{theirs}": (f"model/ae/rae.py::{ours}",)
            for theirs, ours in (
                ("ViTMAESelfAttention", "_Attention"),
                ("ViTMAESelfAttention.__init__", "_Attention.__init__"),
                ("ViTMAESelfAttention.forward", "_Attention.forward"),
                ("ViTMAESelfAttention.transpose_for_scores", "_Attention._heads"),
                ("ViTMAESelfOutput", "_Attention"),
                ("ViTMAESelfOutput.__init__", "_Attention.__init__"),
                ("ViTMAESelfOutput.forward", "_Attention.forward"),
                ("ViTMAEAttention", "_Attention"),
                ("ViTMAEAttention.__init__", "_Attention.__init__"),
                ("ViTMAEAttention.forward", "_Attention.forward"),
                ("ViTMAEIntermediate", "_Block"),
                ("ViTMAEIntermediate.__init__", "_Block.__init__"),
                ("ViTMAEIntermediate.forward", "_Block.forward"),
                ("ViTMAEOutput", "_Block"),
                ("ViTMAEOutput.__init__", "_Block.__init__"),
                ("ViTMAEOutput.forward", "_Block.forward"),
                ("ViTMAELayer", "_Block"),
                ("ViTMAELayer.__init__", "_Block.__init__"),
                ("ViTMAELayer.forward", "_Block.forward"),
                ("GeneralDecoder", "Decoder"),
                ("GeneralDecoder.__init__", "Decoder.__init__"),
                ("GeneralDecoder.initialize_weights", "Decoder.__init__"),
                ("GeneralDecoder.set_trainable_cls_token", "Decoder.__init__"),
                ("GeneralDecoder.interpolate_latent", "Decoder.forward"),
                ("GeneralDecoder.unpatchify", "Decoder.forward"),
                ("GeneralDecoder.forward", "Decoder.forward"),
            )
        },
        **{
            f"{_RAE_DECODER}::{name}": (
                "math/position_embedding.py::sincos_position_table",
            )
            for name in (
                "get_2d_sincos_pos_embed",
                "get_2d_sincos_pos_embed_from_grid",
                "get_1d_sincos_pos_embed_from_grid",
            )
        },
        "src/stage1/decoders/utils.py::ViTMAEConfig": (
            "model/ae/rae.py::Decoder.Config",
        ),
    },
    not_ported={
        "src/stage1/rae.py::Stage1Protocal": (
            "A typing Protocol; model/ae/custom_types.py states the contract."
        ),
        "src/stage1/rae.py::RAE.noising": (
            "A decoder-training augmentation; encode runs it only in training "
            "mode with noise_tau > 0, and the port never trains."
        ),
        "src/stage1/rae.py::RAE.forward": "Encode then decode, both paired.",
        "src/stage1/decoders/utils.py::ViTMAEConfig.__init__": (
            "RAE.__init__ reads its decoder config through transformers' "
            "AutoConfig, which builds HF's own ViTMAEConfig."
        ),
        f"{_RAE_DECODER}::ViTMAEModelOutput": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEForPreTrainingOutput": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEEmbeddings": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEPatchEmbeddings": _RAE_MAE_ENCODER,
        f"{_RAE_DECODER}::ViTMAEDecoderOutput": (
            "An output record; the port's Decoder returns pixels."
        ),
        f"{_RAE_DECODER}::ViTMAESdpaSelfAttention": (
            "Selected by an sdpa attention config; the decoder config the "
            "published RAE uses selects the eager class."
        ),
        f"{_RAE_DECODER}::GeneralDecoder.interpolate_pos_encoding": (
            "Resizes a latent grid the decoder was not built for; the port "
            "raises instead."
        ),
    },
    reference_unreached={
        "src/stage1/rae.py::RAE.__init__::if len(keys.missing_keys) > 0:": "A load report; the port loads strictly, so no key is ever missing.",
        "src/stage1/rae.py::RAE.encode::if self.training and self.noise_tau > 0:": "Training-time noising; both sides only evaluate.",
        "src/stage1/rae.py::RAE.encode::if self.reshape_to_2d:": "The published RAE reshapes; the port always does.",
        "src/stage1/rae.py::RAE.decode::if self.reshape_to_2d:": "The published RAE reshapes; the port always does.",
        "src/stage1/encoders/dinov2.py::Dinov2withNorm.__init__::if normalize:": "The published RAE strips the final norm's affine; the port always does.",
        'src/stage1/decoders/decoder.py::ViTMAESelfAttention.__init__::if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):': "A misconfiguration guard; the port's matching one is tested in rae_test.",
        "src/stage1/decoders/decoder.py::ViTMAESelfAttention.forward::if head_mask is not None:": "No caller passes a head mask.",
        "src/stage1/decoders/decoder.py::ViTMAEIntermediate.__init__::if isinstance(config.hidden_act, str):": "Every ViTMAE config names its activation; the port injects the function.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.unpatchify::if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:": "A shape guard the decoder's own output always satisfies.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if drop_cls_token:": "RAE.decode passes drop_cls_token=False.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.interpolate_latent::if l == self.num_patches:": "The latent always has the decoder's grid; on another the port raises, which rae_test drives.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if not return_dict:": "RAE.decode reads the returned record.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if output_attentions:": "RAE.decode asks for no attentions.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if interpolate_pos_encoding:": "RAE.decode never asks; the port raises on another grid instead.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if output_hidden_states:": "RAE.decode asks for no hidden states.",
        "src/stage1/decoders/decoder.py::GeneralDecoder.forward::if self.gradient_checkpointing and self.training:": "Training-only.",
        "src/stage1/decoders/decoder.py::get_2d_sincos_pos_embed::if add_cls_token:": "initialize_weights always asks for the CLS row.",
        "src/stage1/decoders/decoder.py::get_2d_sincos_pos_embed_from_grid::if embed_dim % 2 != 0:": "A misconfiguration guard; sincos_position_table refuses a width it cannot split.",
        "src/stage1/decoders/decoder.py::get_1d_sincos_pos_embed_from_grid::if embed_dim % 2 != 0:": "A misconfiguration guard; sincos_position_table refuses a width it cannot split.",
    },
    port_unreached={
        "model/ae/latent_norm.py::ElementwiseLatentStats.__init__::if math.isnan(eps) or math.isinf(eps) or eps < 0:": _GUARD,
        "model/ae/latent_norm.py::ElementwiseLatentStats.__init__::if config.stats is None:": _GUARD,
        "model/ae/latent_norm.py::ElementwiseLatentStats.__init__::if var is None:": _GUARD,
        "model/ae/rae.py::RAE.__init__::if config.encoder is None and config.decoder is None:": _GUARD,
        "model/ae/rae.py::RAE.__init__::if any(math.isnan(v) or math.isinf(v) for v in (*mean, *std)) or 0 in std:": _GUARD,
        "model/ae/rae.py::RAE.__init__::if config.encoder is not None:": "The comparison builds both halves; rae_test leaves each out.",
        "model/ae/rae.py::RAE.__init__::if config.encoder_image_size % patch:": _GUARD,
        "model/ae/rae.py::RAE.__init__::if config.decoder is not None:": "The comparison builds both halves; rae_test leaves each out.",
        "model/ae/rae.py::RAE.__init__::if (decoder.channels_in, decoder.num_patches) != (channels, grid * grid):": _GUARD,
        "model/ae/rae.py::RAE.__init__::if decoder.patch_size * grid != config.image_size:": _GUARD,
        "model/ae/rae.py::RAE.encode::if self.encoder is None:": _GUARD,
        "model/ae/rae.py::RAE.encode::if x.dtype != torch.uint8:": _GUARD,
        "model/ae/rae.py::RAE.decode::if self.decoder is None:": _GUARD,
        "model/ae/rae.py::RAE.decode::if tuple(z.shape[-3:]) != self.latent_shape:": "Raises where the reference reshapes; rae_test drives it.",
        "model/ae/rae.py::Encoder.__init__::if config.checkpoint is not None:": _LOADED,
        "model/ae/rae.py::Encoder._positions::if num_patches == num_positions and height == width:": (
            "The encoder side is fixed and its grid is smaller than the table's, as "
            "224 is against the published 518; rae_test drives the other way."
        ),
        "model/ae/rae.py::Decoder.__init__::if n <= 0 or math.isqrt(n) ** 2 != n:": _GUARD,
        "model/ae/rae.py::Decoder.__init__::if config.checkpoint is not None:": _LOADED,
        "model/ae/rae.py::Decoder.forward::if tokens.shape[1] != self.num_patches:": "Raises where the reference resizes; rae_test drives it.",
        "model/ae/rae.py::_Attention.__init__::if channels % num_heads:": _GUARD,
        "math/pixel.py::rgb2float::if not x_.dtype.is_floating_point:": "An input-type guard; the encoders always pass float_dtype=float32.",
        "math/pixel.py::rgb2float::if inplace and x_.requires_grad and x_.is_leaf:": "The encoders never scale a leaf requiring grad in place.",
        "math/pixel.py::rgb2float::if unit_interval:": "Both encoders normalize from [0, 1], so they always pass unit_interval=True.",
        "math/position_embedding.py::sincos_position_table::if channels % 4:": "A misconfiguration guard; position_embedding_test drives it.",
    },
    extra_outputs={
        "encoder.encoder.encoder.layer.0.attention": (
            "HF's attention also returns its attention weights; the attended "
            "output, first, is compared."
        ),
    },
)


_VTP_LAYERS: Final = "vtp/models/layers"
_VTP_TEXT: Final = "The CLIP text and alignment side; reconstruction never reads it."
_VTP_TRAINING: Final = "Training-only; the evaluation forward never runs it."
_VTP_TRUNK: Final = "vtp/models/encoders"

_VTP_REFERENCE_UNREACHED: Final = {
    "vtp/models/vtp_hf/modeling_vtp.py::VTPPreTrainedModel._init_weights::elif isinstance(module, nn.Embedding):": "VTP's reconstruction modules hold no embedding tables.",
    "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._patch_tokens_to_4d::if N != feat_h * feat_w:": "A shape guard the trunk's own tokens always satisfy.",
    "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.__init__::if config.train_clip:": "Reconstruction builds VTPModel without the CLIP side.",
    "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._init_vision_components::if config.train_clip:": "Reconstruction builds VTPModel without the CLIP side.",
    "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._init_vision_components::if config.train_reconstruction:": "Reconstruction always builds the pixel decoder.",
    "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.get_latents_decoded_images::if self.pixel_decoder is None:": "Reconstruction always builds the pixel decoder.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.__init__::if vit_feature_bottleneck is None:": "Every published trunk has a bottleneck.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.__init__::if self.vit_feature_bottleneck != self.original_embed_dim:": "Every published bottleneck narrows; the port refuses an equal width.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck._apply_feature_bottleneck::if self.feature_bottleneck is not None:": "Every published trunk has a bottleneck.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.forward_features::if self.feature_bottleneck is None or not use_bottleneck:": "get_reconstruction_latents always asks for the bottleneck.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.forward_features::if isinstance(output, list):": "Reconstruction encodes one tensor, not a crop list.",
    "vtp/models/encoders/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.forward_features::for item in output:": "Reconstruction encodes one tensor, not a crop list.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.__init__::if len(ignored_kwargs) > 0:": "A warning for unknown constructor keywords; none are passed.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.__init__::if self.n_storage_tokens > 0:": "VTP's trunk has no register tokens.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.init_weights::if self.n_storage_tokens > 0:": "VTP's trunk has no register tokens.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.prepare_tokens_with_masks::if self.n_storage_tokens > 0:": "VTP's trunk has no register tokens.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.__init__::if untie_cls_and_patch_norms:": "VTP ties the CLS and patch norms.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.__init__::if untie_global_and_local_cls_norm:": "VTP ties the global and local CLS norms.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward_features_list::if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:": "VTP ties both norm pairs.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward_features_list::if self.untie_global_and_local_cls_norm and self.training and idx == 1:": "VTP ties both norm pairs, so this branch is never reached.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward_features_list::elif self.untie_cls_and_patch_norms:": "VTP ties both norm pairs, so this branch is never reached.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.prepare_tokens_with_masks::if masks is not None:": "Masked-token pretraining; reconstruction masks nothing.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward_features_list::if self.rope_embed is not None:": "VTP always uses rotary positions.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward_features::if isinstance(x, torch.Tensor):": "Reconstruction encodes one tensor, not a crop list.",
    "vtp/models/encoders/vision_transformer.py::DinoVisionTransformer.forward::if is_training:": "get_reconstruction_latents always asks for the token dict.",
    "vtp/models/encoders/vision_transformer.py::init_weights_vit::if isinstance(module, LayerScale):": "LayerScale is identity at VTP's layerscale_init=None.",
    "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.__init__::if len(ignored_kwargs) > 0:": "A warning for unknown constructor keywords; none are passed.",
    "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.init_weights::if isinstance(m, nn.Conv2d):": "Its only modules are the two convolutions.",
    "vtp/models/decoders/pixel_decoder.py::DinoV3PixelDecoder.init_weights::if m.bias is not None:": "Both convolutions are biased.",
    "vtp/models/layers/attention.py::SelfAttention.compute_attention::if rope is not None:": "VTP always uses rotary positions.",
    "vtp/models/layers/block.py::SelfAttentionBlock.forward::elif isinstance(x_or_x_list, list):": "The trunk and decoder pass one tensor.",
    "vtp/models/layers/block.py::SelfAttentionBlock.forward::if rope_or_rope_list is None:": "The trunk and decoder pass one tensor.",
    "vtp/models/layers/block.py::SelfAttentionBlock._forward_list::if rope_list is not None:": "Reached only for a list input; the blocks get one tensor.",
    "vtp/models/layers/block.py::SelfAttentionBlock._forward_list::if self.training and effective_drop_ratio > 0.0:": "Training-only stochastic depth.",
    "vtp/models/layers/embeddings.py::make_2tuple::if isinstance(x, tuple):": "Every published patch size is an int.",
    "vtp/models/layers/embeddings.py::PatchEmbed.forward::if not self.flatten_embedding:": "VTP flattens its patch embedding.",
    "vtp/models/layers/embeddings.py::PatchEmbed.reset_parameters::if self.proj.bias is not None:": "The patch projection is biased.",
    "vtp/models/layers/embeddings.py::RopePositionEmbedding.__init__::if (base is None and not both_periods) or (base is not None and both_periods):": "A misconfiguration guard; VTP configures a base.",
    "vtp/models/layers/embeddings.py::RopePositionEmbedding._init_weights::if self.base is not None:": "VTP configures a base, not explicit periods.",
    'vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::if self.normalize_coords == "max":': "VTP normalizes each side separately, the one mode the port implements.",
    'vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::elif self.normalize_coords == "min":': "VTP normalizes each side separately, the one mode the port implements.",
    'vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::elif self.normalize_coords == "separate":': "VTP normalizes each side separately, the one mode the port implements.",
    "vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::if self.training and self.shift_coords is not None:": "Training-only coordinate augmentation.",
    "vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::if self.training and self.jitter_coords is not None:": "Training-only coordinate augmentation.",
    "vtp/models/layers/embeddings.py::RopePositionEmbedding.forward::if self.training and self.rescale_coords is not None:": "Training-only coordinate augmentation.",
}


_VTP: Final = Reference(
    url="https://github.com/MiniMax-AI/VTP.git",
    commit="5ce1eb67010fff3c1eed483352483be6a1838556",
    files=(
        "vtp/models/vtp_hf/modeling_vtp.py",
        "vtp/models/vtp_hf/configuration_vtp.py",
        f"{_VTP_TRUNK}/vision_transformer_bottleneck.py",
        f"{_VTP_TRUNK}/vision_transformer.py",
        "vtp/models/decoders/pixel_decoder.py",
        f"{_VTP_LAYERS}/attention.py",
        f"{_VTP_LAYERS}/block.py",
        f"{_VTP_LAYERS}/embeddings.py",
        f"{_VTP_LAYERS}/ffn.py",
        f"{_VTP_LAYERS}/normalization.py",
        f"{_VTP_LAYERS}/misc.py",
        f"{_VTP_LAYERS}/activation.py",
    ),
    paired={
        **{
            f"vtp/models/vtp_hf/{theirs}": (f"model/ae/vtp.py::{ours}",)
            for theirs, ours in (
                ("modeling_vtp.py::VTPPreTrainedModel", "VTP"),
                ("modeling_vtp.py::VTPPreTrainedModel._init_weights", "VTP.__init__"),
                ("modeling_vtp.py::VTPModel", "VTP"),
                ("modeling_vtp.py::VTPModel.__init__", "VTP.__init__"),
                ("modeling_vtp.py::VTPModel._init_vision_components", "VTP.__init__"),
                ("modeling_vtp.py::VTPModel.get_latents_decoded_images", "VTP.decode"),
                ("modeling_vtp.py::VTPModel._patch_tokens_to_4d", "Encoder.forward"),
                ("configuration_vtp.py::VTPConfig", "VTP.Config"),
                ("configuration_vtp.py::VTPConfig.__init__", "VTP.Config"),
            )
        },
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.get_reconstruction_latents": (
            "model/ae/vtp.py::VTP.encode",
            "math/pixel.py::rgb2float",
        ),
        **{
            f"{_VTP_TRUNK}/{theirs}": (f"model/ae/vtp.py::{ours}",)
            for theirs, ours in (
                (
                    "vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck",
                    "Encoder",
                ),
                *(
                    (
                        (
                            "vision_transformer_bottleneck.py::"
                            f"DinoVisionTransformerWithBottleneck.{method}"
                        ),
                        f"Encoder.{ours}",
                    )
                    for method, ours in (
                        ("__init__", "__init__"),
                        ("init_weights", "__init__"),
                        ("_apply_feature_bottleneck", "forward"),
                        ("forward_features", "forward"),
                        ("_process_output_dict", "forward"),
                    )
                ),
                ("vision_transformer.py::DinoVisionTransformer", "Encoder"),
                *(
                    (
                        f"vision_transformer.py::DinoVisionTransformer.{method}",
                        f"Encoder.{ours}",
                    )
                    for method, ours in (
                        ("__init__", "__init__"),
                        ("init_weights", "__init__"),
                        ("prepare_tokens_with_masks", "forward"),
                        ("forward_features_list", "forward"),
                        ("forward_features", "forward"),
                        ("forward", "forward"),
                    )
                ),
                ("vision_transformer.py::init_weights_vit", "Encoder.__init__"),
            )
        },
        **{
            f"vtp/models/decoders/pixel_decoder.py::{theirs}": (
                f"model/ae/vtp.py::{ours}",
            )
            for theirs, ours in (
                ("DinoV3PixelDecoder", "Decoder"),
                ("DinoV3PixelDecoder.__init__", "Decoder.__init__"),
                ("DinoV3PixelDecoder.init_weights", "Decoder.__init__"),
                ("DinoV3PixelDecoder.forward", "Decoder.forward"),
            )
        },
        **{
            f"{_VTP_LAYERS}/{theirs}": (f"model/ae/vtp.py::{ours}",)
            for theirs, ours in (
                ("attention.py::rope_rotate_half", "_rotate_after_prefix"),
                ("attention.py::rope_apply", "_rotate_after_prefix"),
                ("attention.py::SelfAttention", "_Attention"),
                ("attention.py::SelfAttention.__init__", "_Attention.__init__"),
                ("attention.py::SelfAttention.apply_rope", "_rotate_after_prefix"),
                ("attention.py::SelfAttention.forward", "_Attention.forward"),
                ("attention.py::SelfAttention.compute_attention", "_Attention.forward"),
                ("block.py::SelfAttentionBlock", "_Block"),
                ("block.py::SelfAttentionBlock.__init__", "_Block.__init__"),
                ("block.py::SelfAttentionBlock._forward_list", "_Block.forward"),
                ("block.py::SelfAttentionBlock.forward", "_Block.forward"),
                ("embeddings.py::make_2tuple", "Encoder.__init__"),
                ("embeddings.py::PatchEmbed", "Encoder"),
                ("embeddings.py::PatchEmbed.__init__", "Encoder.__init__"),
                ("embeddings.py::PatchEmbed.forward", "Encoder.forward"),
                ("embeddings.py::PatchEmbed.reset_parameters", "Encoder.__init__"),
                ("embeddings.py::RopePositionEmbedding", "_Rotary"),
                ("embeddings.py::RopePositionEmbedding.__init__", "_Rotary.__init__"),
                ("embeddings.py::RopePositionEmbedding.forward", "_Rotary.forward"),
                (
                    "embeddings.py::RopePositionEmbedding._init_weights",
                    "_Rotary.__init__",
                ),
                ("ffn.py::SwiGLUFFN", "_Block"),
                ("ffn.py::SwiGLUFFN.__init__", "_Block.__init__"),
                ("ffn.py::SwiGLUFFN.forward", "_Block.forward"),
                ("normalization.py::RMSNorm", "Encoder"),
                ("normalization.py::RMSNorm.__init__", "Encoder.__init__"),
                ("normalization.py::RMSNorm.reset_parameters", "Encoder.__init__"),
                ("normalization.py::RMSNorm._norm", "Encoder.forward"),
                ("normalization.py::RMSNorm.forward", "Encoder.forward"),
            )
        },
    },
    not_ported={
        **{
            f"vtp/models/vtp_hf/modeling_vtp.py::VTPModel.{name}": _VTP_TEXT
            for name in (
                "_init_text_components",
                "get_clip_image_feature",
                "get_clip_text_feature",
                "get_clip_logits",
                "_forward_clip",
            )
        },
        **{
            f"vtp/models/vtp_hf/modeling_vtp.py::VTPModel.{name}": (
                "A representation probe for linear probing, not the autoencoder."
            )
            for name in (
                "get_last_layer_feature",
                "get_intermediate_layers_feature",
                "_forward_feature",
            )
        },
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel.forward": (
            "The training forward; reconstruction's two calls are paired."
        ),
        "vtp/models/vtp_hf/modeling_vtp.py::VTPModel._forward_reconstruction": (
            "The training forward; reconstruction's two calls are paired."
        ),
        "vtp/models/vtp_hf/configuration_vtp.py::VTPConfig.from_vtp_yaml": (
            "Reads a training YAML; VTP.Config.from_hf reads the published config.json."
        ),
        f"{_VTP_TRUNK}/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.get_intermediate_layers": (
            "A representation probe, not the autoencoder."
        ),
        f"{_VTP_TRUNK}/vision_transformer_bottleneck.py::DinoVisionTransformerWithBottleneck.encode": (
            "An entry point hardcoding a 16-pixel patch; reconstruction goes "
            "through get_reconstruction_latents."
        ),
        f"{_VTP_TRUNK}/vision_transformer.py::DinoVisionTransformer._get_intermediate_layers_not_chunked": (
            "A representation probe, not the autoencoder."
        ),
        f"{_VTP_TRUNK}/vision_transformer.py::DinoVisionTransformer.get_intermediate_layers": (
            "A representation probe, not the autoencoder."
        ),
        **{
            f"{_VTP_TRUNK}/vision_transformer.py::{name}": (
                "A backbone preset; VTPModel builds its trunk directly, and "
                "vtp_small, vtp_base, and vtp_large state the published sizes."
            )
            for name in (
                "vit_small",
                "vit_base",
                "vit_large",
                "vit_so400m",
                "vit_huge2",
                "vit_giant2",
                "vit_7b",
            )
        },
        **{
            f"vtp/models/decoders/pixel_decoder.py::{name}": (
                "A decoder preset; VTPModel builds its decoder directly."
            )
            for name in (
                "dinov3_pixel_decoder_small",
                "dinov3_pixel_decoder_base",
                "dinov3_pixel_decoder_large",
            )
        },
        f"{_VTP_LAYERS}/attention.py::LinearKMaskedBias": (
            "A masked-key-bias projection another configuration selects; VTP's "
            "qkv is a plain biased Linear."
        ),
        f"{_VTP_LAYERS}/attention.py::SelfAttention.forward_list": (
            "The list-batched forward; every block calls forward."
        ),
        f"{_VTP_LAYERS}/attention.py::CausalSelfAttention": _VTP_TEXT,
        f"{_VTP_LAYERS}/attention.py::Attention": _VTP_TEXT,
        f"{_VTP_LAYERS}/attention.py::AttentionalPooler": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::get_branges_scales": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::clear_sampling_cache": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::print_sampling_cache_info": _VTP_TRAINING,
        f"{_VTP_LAYERS}/block.py::SelfAttentionBlock._maybe_index_rope": (
            "Indexes the rotary table for dropped patches; training-only."
        ),
        f"{_VTP_LAYERS}/block.py::SelfAttentionBlock._forward": (
            "The unbatched path; forward wraps a single tensor into _forward_list."
        ),
        f"{_VTP_LAYERS}/block.py::CausalSelfAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::ResidualAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/block.py::CustomResidualAttentionBlock": _VTP_TEXT,
        f"{_VTP_LAYERS}/embeddings.py::PatchEmbed.flops": (
            "A FLOP estimate; Encoder.Config.cost prices the patch convolution."
        ),
        **{
            f"{_VTP_LAYERS}/embeddings.py::{name}": (
                "Absolute position tables; the trunk and decoder use rotary positions."
            )
            for name in (
                "get_2d_sincos_pos_embed",
                "get_2d_sincos_pos_embed_from_grid",
                "get_1d_sincos_pos_embed_from_grid",
                "interpolate_pos_embed",
            )
        },
        f"{_VTP_LAYERS}/ffn.py::ListForwardMixin": (
            "List dispatch the blocks never use: they call the FFN on a tensor."
        ),
        f"{_VTP_LAYERS}/ffn.py::Mlp": "The GELU MLP; VTP's configurations choose SwiGLU.",
        f"{_VTP_LAYERS}/normalization.py::LayerNorm": _VTP_TEXT,
        f"{_VTP_LAYERS}/normalization.py::LayerNormFp32": _VTP_TEXT,
        f"{_VTP_LAYERS}/misc.py::LayerScale": (
            "Identity at VTP's layerscale_init=None; the block builds nn.Identity."
        ),
        f"{_VTP_LAYERS}/misc.py::PatchDropout": _VTP_TRAINING,
        f"{_VTP_LAYERS}/activation.py::QuickGELU": _VTP_TEXT,
    },
    reference_unreached=_VTP_REFERENCE_UNREACHED,
    port_unreached={
        "model/ae/vtp.py::VTP.__init__::if config.encoder is None and config.decoder is None:": _GUARD,
        "model/ae/vtp.py::VTP.__init__::if any(math.isnan(v) or math.isinf(v) for v in stats) or 0 in config.pixel_std:": _GUARD,
        "model/ae/vtp.py::VTP.__init__::if config.image_size % patch:": _GUARD,
        "model/ae/vtp.py::VTP.__init__::if config.encoder is not None and config.decoder is not None:": "The comparison builds both halves; vtp_test leaves each out.",
        "model/ae/vtp.py::VTP.__init__::if (decoder.channels_in, decoder.upscale_factor) != (": _GUARD,
        "model/ae/vtp.py::VTP.encode::if self.encoder is None:": _GUARD,
        "model/ae/vtp.py::VTP.encode::if x.dtype != torch.uint8:": _GUARD,
        "model/ae/vtp.py::VTP.encode::if height % self.patch_size or width % self.patch_size:": _GUARD,
        "model/ae/vtp.py::VTP.decode::if self.decoder is None:": _GUARD,
        "model/ae/vtp.py::Encoder.__init__::if config.channels_out == config.channels_hidden:": _GUARD,
        "model/ae/vtp.py::Encoder.__init__::if config.checkpoint is not None:": _LOADED,
        "model/ae/vtp.py::Decoder.__init__::if config.checkpoint is not None:": _LOADED,
        "model/ae/vtp.py::_Rotary.__init__::if width % (4 * num_heads):": _GUARD,
        "math/pixel.py::rgb2float::if not x_.dtype.is_floating_point:": "An input-type guard; the encoders always pass float_dtype=float32.",
        "math/pixel.py::rgb2float::if inplace and x_.requires_grad and x_.is_leaf:": "The encoders never scale a leaf requiring grad in place.",
        "math/pixel.py::rgb2float::if unit_interval:": "Both encoders normalize from [0, 1], so they always pass unit_interval=True.",
    },
    extra_outputs={},
    uncompared={
        "trunk": (
            "The reference trunk returns a dict of tokens; its patch tokens, "
            "reshaped, are the compared latent."
        ),
    },
    reference_calls={
        "trunk.feature_bottleneck": (
            (1,),
            (
                "The reference also projects the CLS token first, which "
                "reconstruction discards; its second call, the patches, is the port's."
            ),
        ),
    },
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Definition:
    """One function, method, or class, and the body lines that are its own."""

    key: str
    """``"<file>::<Qualified.name>"``, as the inventory spells it."""

    path: Path
    """The file the definition is in."""

    function: bool
    """True for a ``def``, False for a ``class``."""

    lines: frozenset[int]
    """Body lines, less those of any definition nested inside it."""


def definitions(root: Path, relative: str) -> list[Definition]:
    """Return every definition in a file, nested ones included.

    Args:
      root: Directory ``relative`` resolves against.
      relative: The file, as inventory keys spell it.

    Returns:
      definitions: One per ``def`` and ``class``, keyed ``"<relative>::<qualname>"``,
        under an ``if``, ``try``, or ``with`` as much as at the top.

    """
    path = root / relative
    tree = ast.parse(path.read_text())
    return list(_definitions(tree, path=path, prefix=f"{relative}::"))


class Measured(Protocol):
    """What the inventory reads of a ``coverage.Coverage`` run with branches."""

    def get_data(self) -> MeasuredLines:
        """Return the lines each file ran."""
        ...

    def branch_stats(self, morf: str) -> dict[int, tuple[int, int]]:
        """Return ``line -> (exits, exits taken)`` for each branch in ``morf``."""
        ...


class MeasuredLines(Protocol):
    """The executed lines of a measured run."""

    def lines(self, filename: str) -> list[int] | None:
        """Return the lines ``filename`` ran, or ``None`` if it never ran."""
        ...


def inventory_problems(
    reference: Reference,
    *,
    clone: Path,
    measured: Measured,
) -> list[str]:
    """Check the inventory against the reference files and the measured run.

    Args:
      reference: The pinned reference and its tables.
      clone: The reference's clone.
      measured: Coverage of one run of both sides, with branches.

    Returns:
      problems: Every gap, stale entry, and unexplained branch, one per line.

    """
    problems: list[str] = []
    theirs = {
        d.key: d
        for relative in reference.files
        for d in definitions(clone, relative=relative)
    }
    ours: dict[str, Definition] = {}
    for keys in reference.paired.values():
        for key in keys:
            relative = key.split("::", 1)[0]
            ours.update({d.key: d for d in definitions(_PRIML, relative=relative)})

    def not_ported(key: str) -> bool:
        return any(key == k or key.startswith(f"{k}.") for k in reference.not_ported)

    problems.extend(
        f"unpaired reference definition: {key}"
        for key in theirs
        if key not in reference.paired and not not_ported(key)
    )
    problems.extend(
        f"inventory names a definition the reference lacks: {key}"
        for key in (*reference.paired, *reference.not_ported)
        if key not in theirs
    )
    for keys in reference.paired.values():
        problems.extend(
            f"inventory names a port definition that does not exist: {key}"
            for key in keys
            if key not in ours
        )

    data = measured.get_data()

    def ran(definition: Definition) -> bool:
        return bool(definition.lines & set(data.lines(str(definition.path)) or ()))

    for key, definition in theirs.items():
        if not definition.function:
            continue
        if key in reference.paired and not ran(definition):
            problems.append(f"paired but never ran: {key}")
        if not_ported(key) and ran(definition):
            problems.append(f"not ported but ran, so it is on the path: {key}")
    port_keys = {key for keys in reference.paired.values() for key in keys}
    problems.extend(
        f"port definition never ran: {key}"
        for key in sorted(port_keys & ours.keys())
        if ours[key].function and not ran(ours[key])
    )

    problems += _branch_problems(
        [theirs[k] for k in reference.paired if k in theirs and theirs[k].function],
        allowed=reference.reference_unreached,
        measured=measured,
    )
    problems += _branch_problems(
        [ours[k] for k in sorted(port_keys & ours.keys()) if ours[k].function],
        allowed=reference.port_unreached,
        measured=measured,
    )
    return problems


class PinnedRandom(TorchDispatchMode):
    """Serve ``randn`` from stored draws; refuse every other seeded random op.

    A dispatch mode, so it sees what a native kernel draws (dropout's mask) and
    a ``randn`` bound before entry as surely as a Python call. Each draw is
    counted under the first frame outside torch that asked for it.
    """

    def __init__(self, draws: Sequence[Tensor]) -> None:
        super().__init__()
        self._draws = list(draws)
        self.sites: Counter[str] = Counter()

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Leave the mode and require every draw consumed."""
        super().__exit__(exc_type, exc_val, exc_tb)
        if self._draws and exc_type is None:
            raise AssertionError(f"{len(self._draws)} pinned draws were never taken")

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        kwargs = kwargs or {}
        if not _draws(func, args=args, kwargs=kwargs):
            return func(*args, **kwargs)
        if func.overloadpacket is not torch.ops.aten.randn:
            raise AssertionError(f"unpinned random op {func} was called")
        self.sites[_caller()] += 1
        shape = tuple(cast("Sequence[int]", args[0]))
        if not self._draws:
            raise AssertionError(f"an unpinned torch.randn{shape} was drawn")
        draw = self._draws.pop(0)
        if tuple(draw.shape) != shape:
            raise AssertionError(
                f"torch.randn{shape} drawn; the pinned draw is {tuple(draw.shape)}",
            )
        return draw.to(
            device=cast("torch.device | None", kwargs.get("device")),
            dtype=cast("torch.dtype | None", kwargs.get("dtype")),
        )


type Record = dict[str, list[list[Tensor]]]
"""Module name -> one entry per call -> that call's output tensors, in order."""


@contextmanager
def recorded(model: nn.Module) -> Generator[Record]:
    """Record every submodule's tensor outputs, per call, by module name.

    Args:
      model: The module whose named submodules are hooked; the root is not.

    Yields:
      outputs: Filled as the model runs; the hooks come off on exit.

    """
    outputs: Record = defaultdict(list)
    handles = [
        module.register_forward_hook(_recorder(outputs[name]))
        for name, module in model.named_modules()
        if name
    ]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def compare(
    label: str,
    theirs: Record,
    ours: Record,
    *,
    allowances: Reference,
) -> tuple[list[str], int]:
    """Compare every name both records share, exactly.

    Args:
      label: Prefix for each reported line.
      theirs: The reference's record.
      ours: The port's record.
      allowances: The reference whose ``extra_outputs``, ``uncompared``, and
        ``reference_calls`` name each structural difference, with a reason.

    Returns:
      problems: One line per differing call count, output count, dtype,
        shape, or bits.
      compared: How many names both sides recorded.

    """
    problems: list[str] = []
    shared = sorted(theirs.keys() & ours.keys())
    for name in shared:
        if name in allowances.uncompared:
            continue
        want_calls, got_calls = theirs[name], ours[name]
        if name in allowances.reference_calls:
            keep, _ = allowances.reference_calls[name]
            want_calls = [want_calls[i] for i in keep if i < len(want_calls)]
        if len(want_calls) != len(got_calls):
            problems.append(
                f"{label} {name}: {len(got_calls)} calls vs {len(want_calls)}",
            )
            continue
        for call, (want, got) in enumerate(zip(want_calls, got_calls, strict=True)):
            if len(want) != len(got) and name not in allowances.extra_outputs:
                problems.append(
                    f"{label} {name} call {call}: {len(got)} outputs vs {len(want)}",
                )
            for index, (a, b) in enumerate(zip(want, got, strict=False)):
                where = f"{label} {name} call {call} output {index}"
                if a.dtype != b.dtype or a.shape != b.shape:
                    problems.append(
                        f"{where}: {b.dtype}{list(b.shape)} vs "
                        f"{a.dtype}{list(a.shape)}",
                    )
                elif not torch.equal(a, b):
                    problems.append(
                        f"{where}: {int((a != b).sum())}/{a.numel()} differ",
                    )
    return problems, len(shared)


def state_problems(
    label: str,
    *,
    expected: Mapping[str, Tensor],
    ours: nn.Module,
) -> list[str]:
    """Report where the port's state differs from the reference's, renamed.

    Args:
      label: Prefix for each reported line.
      expected: The reference state, under the port's names.
      ours: The port module.

    Returns:
      problems: One line if the names differ, else one per differing tensor.

    """
    got = ours.state_dict()
    if expected.keys() != got.keys():
        return [
            (
                f"{label}: state keys differ: "
                f"{sorted(got.keys() - expected.keys())[:4]} vs "
                f"{sorted(expected.keys() - got.keys())[:4]}"
            ),
        ]
    return [
        f"{label}: {key} differs"
        for key, want in expected.items()
        if want.dtype != got[key].dtype or not torch.equal(want, got[key])
    ]


def clone_upstream(reference: Reference, root: Path) -> Path:
    """Fetch the reference at its pinned commit, or verify an existing clone.

    Args:
      reference: What to fetch.
      root: Directory the clone lives in.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: A git command fails, or an existing clone is dirty or at
        another commit, so what it contains is no longer the reference this
        comparison names.

    """
    if not (root / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        for arguments in (
            ("init", "--quiet"),
            ("fetch", "--quiet", "--depth", "1", reference.url, reference.commit),
            ("checkout", "--quiet", "FETCH_HEAD"),
        ):
            _ = _git(root, *arguments)
    head = _git(root, "rev-parse", "HEAD")
    if head != reference.commit:
        raise RuntimeError(f"{root} is at {head}, expected {reference.commit}")
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"{root} has local modifications:\n{dirty}")
    return root.resolve()


@dataclass(slots=True, kw_only=True)
class Outcome:
    """What one model's comparison found."""

    problems: list[str]
    """One line per mismatch."""

    compared: int = 0
    """How many module names both sides recorded."""

    sites: Counter[str] = dataclasses.field(default_factory=Counter[str])
    """``file:line`` -> how many pinned ``torch.randn`` draws it took."""

    golden: dict[str, Tensor] = dataclasses.field(default_factory=dict[str, Tensor])
    """The record to mint or check against the checked-in golden; empty if none."""


def run_invae(clone: Path, work: Path) -> Outcome:
    """Compare INVAE with REG's ``AutoencoderKL`` at the published f16d32 architecture.

    Both sides hold one set of weights: REG's, renamed to the published
    diffusers names the port loads. Module outputs are compared under REG's
    module names, the port's renamed to them.

    Args:
      clone: REG's clone.
      work: Scratch directory, unused.

    Returns:
      outcome: Mismatches, the modules compared, and the randn call sites.

    """
    del work
    _ = sys.modules.setdefault("dictdot", _dictdot_stub())
    sys.path.insert(0, str(clone))
    upstream = cast("_InvaeModule", importlib.import_module("models.invae"))
    outcome = Outcome(problems=[])

    torch.manual_seed(0)
    theirs = upstream.VAE_F16D32()
    their_invae = cast("_TheirInvae", theirs)
    _randomize(theirs, seed=1)
    ours = INVAE.Config(
        encoder=InvaeEncoder.Config(checkpoint=None),
        decoder=InvaeDecoder.Config(checkpoint=None),
    ).make()
    if ours.encoder is None:
        raise ValueError("Expected ours.encoder is not None.")
    if ours.decoder is None:
        raise ValueError("Expected ours.decoder is not None.")
    published = _ldm_to_diffusers(theirs.state_dict())
    for half, module in (("encoder", ours.encoder), ("decoder", ours.decoder)):
        module.load_state_dict(remap_invae_state_dict(published, half=half))
    # The port's ``encoder`` ends in ``quant_conv``, REG's before it; the port's
    # output is compared as ``quant_conv``.
    reference_names = {
        _invae_module_name(name): name
        for name, _ in theirs.named_modules()
        if name and name != "encoder"
    }
    theirs.eval()
    generator = torch.Generator().manual_seed(0)
    for index in range(2):
        image = torch.randint(
            0,
            256,
            (2, 3, 32, 32),
            generator=generator,
            dtype=torch.uint8,
        )
        for draw in range(3):
            noise = torch.randn(2, 32, 2, 2, generator=generator)
            label = f"invae image {index} draw {draw}"
            with torch.no_grad(), host_agnostic_numerics():
                with PinnedRandom([noise]) as pinned, recorded(theirs) as their_modules:
                    # REG's own preprocessing: preprocessing/encoders.py,
                    # InvaeEncoder.encode.
                    upstream_posterior = their_invae.encode(
                        image.to(torch.float32) / 127.5 - 1,
                    )
                    their_sample = upstream_posterior.sample()
                    # REG's decode: generate.py, ``(samples + 1) / 2`` then a clamp.
                    their_decoded = (
                        (their_invae.decode(their_sample).sample + 1) / 2
                    ).clamp(0, 1)
                outcome.sites.update(pinned.sites)
                with PinnedRandom([noise]) as pinned, recorded(ours) as our_modules:
                    our_posterior = ours.encode(image)
                    our_sample = our_posterior.sample()
                    our_decoded = ours.decode(our_sample)
                outcome.sites.update(pinned.sites)
            problems, compared = compare(
                label,
                theirs={
                    **their_modules,
                    "mean": [[upstream_posterior.mean]],
                    "std": [[upstream_posterior.std]],
                    "mode": [[upstream_posterior.mode()]],
                    "sample": [[their_sample]],
                    "decoded": [[their_decoded]],
                },
                ours={
                    **_renamed(our_modules, rename=reference_names.get),
                    "mean": [[our_posterior.mean]],
                    "std": [[our_posterior.std]],
                    "mode": [[our_posterior.mode()]],
                    "sample": [[our_sample]],
                    "decoded": [[our_decoded]],
                },
                allowances=_INVAE,
            )
            outcome.problems += problems
            outcome.compared = compared
    return outcome


def _ldm_to_diffusers(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename REG's LDM ``AutoencoderKL`` state to the published diffusers names."""
    return {
        _invae_published_name(key): tensor.reshape(_diffusers_shape(key, tensor))
        for key, tensor in state.items()
    }


def _diffusers_shape(key: str, tensor: Tensor) -> tuple[int, ...]:
    """Return a tensor's diffusers shape: attention projections are ``Linear``."""
    if re.search(r"\.(attn_1|attn\.\d+)\.(q|k|v|proj_out)\.weight$", key):
        return tuple(tensor.shape[:2])
    return tuple(tensor.shape)


def _invae_published_name(key: str) -> str:
    """Map one REG ``AutoencoderKL`` key to its diffusers name."""
    name = _invae_module_name(key.rsplit(".", 1)[0]) + "." + key.rsplit(".", 1)[1]
    half = name.split(".", 1)[0]
    if half in {"quant_conv", "post_quant_conv"}:
        return key
    rest = name.split(".", 1)[1]
    if rest.startswith(("quant_conv.", "post_quant_conv.")):
        return rest
    return name.replace(".to_out.", ".to_out.0.")


def _invae_module_name(name: str) -> str:
    """Map a REG ``AutoencoderKL`` module name to the port's, both halves prefixed."""
    if name in {"quant_conv", "post_quant_conv"}:
        return {
            "quant_conv": "encoder.quant_conv",
            "post_quant_conv": "decoder.post_quant_conv",
        }[name]
    name = re.sub(
        r"\.mid\.block_(\d)",
        lambda m: f".mid_block.resnets.{int(m[1]) - 1}",
        name,
    )
    name = name.replace(".mid.attn_1", ".mid_block.attentions.0")
    name = name.replace(".nin_shortcut", ".conv_shortcut").replace(
        ".norm_out",
        ".conv_norm_out",
    )
    name = re.sub(r"\.down\.(\d+)\.block", r".down_blocks.\1.resnets", name)
    name = re.sub(r"\.down\.(\d+)\.attn", r".down_blocks.\1.attentions", name)
    name = re.sub(r"\.down\.(\d+)\.downsample", r".down_blocks.\1.downsamplers.0", name)
    match = re.match(r"decoder\.up\.(\d+)\.(block|attn|upsample)(.*)", name)
    if match:
        # REG stores ``up`` finest first; the port, as diffusers, coarsest first.
        level = (
            len(InvaeDecoder.Config(checkpoint=None).channels_hidden)
            - 1
            - int(match[1])
        )
        kind = {"block": "resnets", "attn": "attentions", "upsample": "upsamplers.0"}[
            match[2]
        ]
        name = f"decoder.up_blocks.{level}.{kind}{match[3]}"
    name = re.sub(r"(attentions\.\d+)\.norm$", r"\1.group_norm", name)
    name = re.sub(r"(attentions\.\d+)\.(q|k|v)$", r"\1.to_\2", name)
    return re.sub(r"(attentions\.\d+)\.proj_out$", r"\1.to_out", name)


def _renamed(record: Record, rename: Callable[[str], str | None]) -> Record:
    """Key a record by reference module names, dropping port-only modules."""
    return {
        theirs: calls
        for ours, calls in record.items()
        if (theirs := rename(ours)) is not None
    }


def run_rae(clone: Path, work: Path) -> Outcome:
    """Compare RAE with the reference ``stage1.RAE`` on ``rae_test.tiny``'s geometry.

    Width 8, a 2x2 latent grid, 8px out, with a 16px input that drives the
    bicubic resize and an 8px one that skips it. Latent normalization is
    compared with and without a mean. Each half loads the reference's own
    checkpoint files through its published-name remap.

    Args:
      clone: The RAE clone.
      work: Scratch directory for the tiny checkpoints and configs.

    Returns:
      outcome: Mismatches, the modules compared, and the golden record.

    """
    sys.path.insert(0, str(clone / "src"))
    stage1 = cast("_Stage1Module", importlib.import_module("stage1"))
    outcome = Outcome(problems=[])
    encoder_dir, decoder_dir = work / "enc", work / "dec"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)

    hf = modeling_dinov2_with_registers.Dinov2WithRegistersModel(
        configuration_dinov2_with_registers.Dinov2WithRegistersConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            mlp_ratio=4,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            image_size=12,
            patch_size=4,
            num_channels=3,
            qkv_bias=True,
            layerscale_value=1.0,
            num_register_tokens=4,
            use_swiglu_ffn=False,
        ),
    )
    _randomize(hf, seed=1)
    hf.save_pretrained(encoder_dir)
    _ = (encoder_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "crop_size": {"height": 224, "width": 224},
                "do_center_crop": True,
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_processor_type": "BitImageProcessor",
                "image_std": [0.229, 0.224, 0.225],
                "resample": 3,
                "rescale_factor": 1 / 255,
                "size": {"shortest_edge": 256},
            },
        ),
    )
    decoder_json = json_loads(
        (clone / "configs/decoder/ViTXL/config.json").read_text(),
        dict[str, object],
    )
    decoder_json.update(
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        decoder_num_attention_heads=2,
        decoder_num_hidden_layers=1,
        patch_size=4,
    )
    _ = (decoder_dir / "config.json").write_text(json.dumps(decoder_json))
    bootstrap = cast("_TheirRae", _their_rae(stage1, work=work))
    _randomize(bootstrap.decoder, seed=2)
    torch.save(bootstrap.decoder.state_dict(), work / "decoder.pt")

    config = RAE.Config(
        encoder=RaeEncoder.Config(
            channels_hidden=8,
            num_layers=1,
            heads=2,
            patch_size=4,
            image_size=12,
            checkpoint=None,
        ),
        decoder=RaeDecoder.Config(
            channels_in=8,
            num_patches=4,
            channels_hidden=8,
            channels_hidden_mlp=16,
            num_layers=1,
            heads=2,
            patch_size=4,
            checkpoint=None,
        ),
        image_size=8,
        encoder_image_size=8,
    )
    if config.encoder is None:
        raise ValueError("Expected config.encoder is not None.")
    if config.decoder is None:
        raise ValueError("Expected config.decoder is not None.")
    config.encoder.checkpoint = LocalFile.Config(path=encoder_dir / "model.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=work / "decoder.pt")
    ours = config.make()
    published = {
        **{
            f"state.encoder.encoder.{k}": v
            for k, v in read_state(encoder_dir / "model.safetensors").items()
            if not k.startswith("layernorm.")
        },
        **{f"state.decoder.{k}": v for k, v in read_state(work / "decoder.pt").items()},
    }

    mean = torch.randn(8, 2, 2, generator=torch.Generator().manual_seed(3))
    var = torch.rand(8, 2, 2, generator=torch.Generator().manual_seed(4)) + 0.5
    torch.save({"mean": mean, "var": var}, work / "stats.pt")
    torch.save({"mean": None, "var": var}, work / "stats_var.pt")

    large = torch.randint(
        0,
        256,
        (2, 3, 16, 16),
        generator=torch.Generator().manual_seed(0),
        dtype=torch.uint8,
    )
    exact = torch.randint(
        0,
        256,
        (2, 3, 8, 8),
        generator=torch.Generator().manual_seed(5),
        dtype=torch.uint8,
    )
    for stats in (None, work / "stats.pt", work / "stats_var.pt"):
        theirs = _their_rae(stage1, work=work, decoder=work / "decoder.pt", stats=stats)
        theirs.eval()
        # The one reference-side change: the kernel the port fixes, through HF's setter.
        their_rae = cast("_TheirRae", theirs)
        their_rae.encoder.encoder.set_attn_implementation("eager")
        outcome.problems += state_problems(
            "rae weights",
            expected=_rae_port_state(theirs.state_dict()),
            ours=ours,
        )
        norm = None
        if stats is not None:
            norm = ElementwiseLatentStats.Config(
                stats=LocalFile.Config(path=stats),
            ).make()
        for name, image in (("16px", large), ("8px", exact)):
            label = f"rae {name} stats={None if stats is None else stats.name}"
            with torch.no_grad(), host_agnostic_numerics(), PinnedRandom([]):
                with recorded(theirs) as their_modules:
                    their_latent = their_rae.encode(image.float() / 255)
                    their_decoded = their_rae.decode(their_latent)
                with recorded(ours) as our_modules:
                    raw = ours.encode(image).mode()
                    our_latent = raw if norm is None else norm.normalize(raw)
                    our_decoded = ours.decode(
                        our_latent if norm is None else norm.denormalize(our_latent),
                    )
            problems, compared = compare(
                label,
                theirs={
                    **their_modules,
                    "latent": [[their_latent]],
                    "decoded": [[their_decoded.clamp(0, 1)]],
                },
                ours={
                    **_renamed(our_modules, rename=_rae_module_name),
                    "latent": [[our_latent]],
                    "decoded": [[our_decoded]],
                },
                allowances=_RAE,
            )
            outcome.problems += problems
            outcome.compared = max(outcome.compared, compared)
            if stats is None and name == "16px":
                outcome.golden = {
                    "source_commit": _commit_tensor(_RAE),
                    **published,
                    "image": image,
                    "latent": their_latent.float(),
                    "decoded": their_decoded.float(),
                }

    return outcome


def _rae_port_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename the reference RAE's state to the port's, through the port's remaps."""
    encoder = {
        k.removeprefix("encoder.encoder."): v
        for k, v in state.items()
        if k.startswith("encoder.encoder.")
    }
    decoder = {
        k.removeprefix("decoder."): v
        for k, v in state.items()
        if k.startswith("decoder.")
    }
    return {
        **{f"encoder.{k}": v for k, v in remap_encoder_state_dict(encoder).items()},
        **{f"decoder.{k}": v for k, v in remap_decoder_state_dict(decoder).items()},
    }


_RAE_ENCODER_MODULES: Final = (
    (
        r"^encoder\.patch_embed$",
        "encoder.encoder.embeddings.patch_embeddings.projection",
    ),
    (r"^encoder\.layers\.", "encoder.encoder.encoder.layer."),
    (r"^encoder\.norm$", "encoder.encoder.layernorm"),
    (r"\.attention\.query$", ".attention.q_proj"),
    (r"\.attention\.key$", ".attention.k_proj"),
    (r"\.attention\.value$", ".attention.v_proj"),
    (r"\.attention\.out$", ".attention.o_proj"),
    (r"\.mlp_in$", ".mlp.fc1"),
    (r"\.mlp_out$", ".mlp.fc2"),
)

_RAE_DECODER_MODULES: Final = (
    (r"\.attention\.(query|key|value)$", r".attention.attention.\1"),
    (r"\.attention\.out$", ".attention.output.dense"),
    (r"\.norm1$", ".layernorm_before"),
    (r"\.norm2$", ".layernorm_after"),
    (r"\.mlp_in$", ".intermediate.dense"),
    (r"\.mlp_out$", ".output.dense"),
)


def _rae_module_name(name: str) -> str | None:
    """Map an RAE port module name to the reference's; ``None`` for port-only ones."""
    # The port's decoder returns pixels; the reference's, patch logits. Both
    # are compared: as ``decoder.decoder_pred`` and as the decoded image.
    if name == "decoder":
        return None
    rules = _RAE_ENCODER_MODULES if name.startswith("encoder") else _RAE_DECODER_MODULES
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def run_vtp(clone: Path, work: Path) -> Outcome:
    """Compare VTP with ``VTPModel``'s reconstruction on ``vtp_test.tiny``'s geometry.

    Both sides load the published-layout state stored in the reference golden;
    the port's halves read it through their own checkpoint path.

    Args:
      clone: The VTP clone.
      work: Scratch directory for the shared checkpoint.

    Returns:
      outcome: Mismatches, the modules compared, and the golden record.

    """
    for name, path in (
        ("vtp", clone / "vtp"),
        ("vtp.models", clone / "vtp" / "models"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    sys.path.insert(0, str(clone))
    decoders = cast("_DecoderModule", importlib.import_module("vtp.models.decoders"))
    hf = cast("_VtpHfModule", importlib.import_module("vtp.models.vtp_hf"))
    outcome = Outcome(problems=[])

    golden = read_tensors(_TESTDATA / "vtp_reference.pt")
    state = {
        k.removeprefix("state."): v for k, v in golden.items() if k.startswith("state.")
    }
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(work / "vtp.safetensors"),
    )
    theirs = hf.VTPModel(
        hf.VTPConfig(
            train_clip=False,
            image_size=8,
            vision_patch_size=4,
            vision_embed_dim=16,
            vision_depth=1,
            vision_num_heads=2,
            vision_mlp_ratio=1.0,
            vision_feature_bottleneck=4,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=2,
        ),
    )
    # VTPModel builds its decoder at ffn ratio 4 and upscale 16; the same
    # constructor at the golden's ratio 1 and upscale 4.
    theirs.pixel_decoder = decoders.DinoV3PixelDecoder(
        in_chans=4,
        embed_dim=16,
        num_heads=2,
        depth=1,
        ffn_ratio=1.0,
        ffn_layer="swiglu",
        norm_layer="layernorm",
        upscale_factor=4,
    )
    _ = theirs.load_state_dict(state, strict=True)
    theirs.eval()
    their_vtp = cast("_TheirVtp", theirs)
    config = VTP.Config(
        encoder=VtpEncoder.Config(
            patch_size=4,
            channels_hidden=16,
            num_layers=1,
            heads=2,
            expansion=1.0,
            channels_out=4,
            checkpoint=None,
        ),
        decoder=VtpDecoder.Config(
            channels_in=4,
            channels_hidden=16,
            num_layers=1,
            heads=2,
            expansion=1.0,
            upscale_factor=4,
            checkpoint=None,
        ),
        image_size=8,
    )
    if config.encoder is None:
        raise ValueError("Expected config.encoder is not None.")
    if config.decoder is None:
        raise ValueError("Expected config.decoder is not None.")
    config.encoder.checkpoint = LocalFile.Config(path=work / "vtp.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=work / "vtp.safetensors")
    ours = config.make()
    config.dtype_autocast = torch.bfloat16
    ours_bf16 = config.make()

    golden_input = golden["input"]
    square = torch.randint(
        0,
        256,
        (2, 3, 8, 8),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.uint8,
    )
    for label, image, model, autocast in (
        ("vtp golden input", golden_input, ours, None),
        ("vtp square input", square, ours, None),
        ("vtp bfloat16 autocast", golden_input, ours_bf16, torch.bfloat16),
    ):
        numerics = host_agnostic_numerics() if autocast is None else nullcontext()
        with torch.no_grad(), numerics, PinnedRandom([]):
            with recorded(theirs) as their_modules:
                expected = _their_round_trip(their_vtp, image=image, autocast=autocast)
            with recorded(model) as our_modules:
                latent = model.encode(image).mode()
                got = {"latent": latent, "image": model.decode(latent)}
        # The raw pixels are the recorded ``pixel_decoder`` output on both sides.
        problems, compared = compare(
            label,
            theirs={
                **their_modules,
                "latent": [[expected["latent"]]],
                "image": [[expected["image"]]],
            },
            ours={
                **_renamed(our_modules, rename=_vtp_module_name),
                **{k: [[v]] for k, v in got.items()},
            },
            allowances=_VTP,
        )
        outcome.problems += problems
        outcome.compared = max(outcome.compared, compared)
        if label == "vtp golden input":
            outcome.golden = {
                "source_commit": _commit_tensor(_VTP),
                **{k: v.float() for k, v in expected.items()},
                **{f"state.{k}": v for k, v in state.items()},
                "input": golden_input,
            }
    return outcome


_VTP_MODULES: Final = (
    (r"^encoder", "trunk"),
    (r"^decoder", "pixel_decoder"),
    (r"\.rope$", ".rope_embed"),
    (r"^trunk\.patch_embed$", "trunk.patch_embed.proj"),
    (r"^trunk\.bottleneck$", "trunk.feature_bottleneck"),
    (r"\.mlp\.down_proj$", ".mlp.w3"),
)


def _vtp_module_name(name: str) -> str | None:
    """Map a VTP port module name to the reference's; ``None`` for port-only ones."""
    # The reference projects gate and value separately; the port fuses them.
    if name.endswith(".mlp.up_proj"):
        return None
    for pattern, replacement in _VTP_MODULES:
        name = re.sub(pattern, replacement, name)
    return name


def measure(
    reference: Reference,
    run: Callable[[Path, Path], Outcome],
    *,
    clone: Path,
    work: Path,
) -> tuple[Outcome, Measured]:
    """Run one comparison under a branch measurement of its own.

    Args:
      reference: The pinned reference whose files are measured.
      run: The comparison, given the clone and a scratch directory.
      clone: The reference's clone.
      work: Scratch directory for the comparison.

    Returns:
      outcome: What the comparison found.
      measured: Branch coverage of this run alone, so no other model's run can
        satisfy its inventory.

    """
    measured = coverage.Coverage(
        branch=True,
        data_file=None,
        include=[
            *(str(clone / f) for f in reference.files),
            str(_PRIML / "model" / "ae" / "*.py"),
            str(_PRIML / "model" / "*.py"),
            str(_PRIML / "math" / "*.py"),
        ],
        config_file=False,
    )
    measured.start()
    try:
        outcome = run(clone, work)
    finally:
        measured.stop()
    return outcome, measured


def main() -> int:
    """Run the selected comparisons and print every problem found.

    Returns:
      status: 0 when every selected port matches its reference, else 1.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("_Flags", parser.parse_args())
    # Before the first matmul: pytest, which replays the goldens this mints, runs
    # one math thread with MKL's kernel pinned (priml's conftest), and another
    # reduction order or GEMM kernel can move a float64 result across a float32
    # rounding boundary.
    os.environ["MKL_CBWR"] = "COMPATIBLE"
    torch.set_num_threads(1)

    clones = {
        name: clone_upstream(_MODELS[name][0], root=flags.clone_dir / name)
        for name in flags.model
    }
    failed = False
    with tempfile.TemporaryDirectory(prefix="reference-parity-") as scratch:
        for name in flags.model:
            reference, run, golden_name = _MODELS[name]
            work = Path(scratch) / name
            work.mkdir()
            outcome, measured = measure(
                reference,
                run=run,
                clone=clones[name],
                work=work,
            )
            problems = list(outcome.problems)
            if golden_name is not None and outcome.golden:
                path = _TESTDATA / golden_name
                if flags.mint:
                    write_tensors(path, record=outcome.golden)
                else:
                    problems += [
                        f"{golden_name}: {line}"
                        for line in mismatches(
                            read_tensors(path),
                            actual=outcome.golden,
                        )
                    ]
            problems += inventory_problems(
                reference,
                clone=clones[name],
                measured=measured,
            )
            print(
                f"== {name} at {reference.commit[:7]}: "
                f"{outcome.compared} module outputs compared",
            )
            if outcome.sites:
                print(f"   torch.randn call sites: {dict(outcome.sites)}")
            for problem in problems:
                print(f"   {problem}")
            print(f"   {'FAIL' if problems else 'ok'}: {len(problems)} problems")
            failed |= bool(problems)
    return int(failed)


def _definitions(node: ast.AST, *, path: Path, prefix: str) -> Iterator[Definition]:
    """Yield every definition below ``node``, through any compound statement."""
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield from _definitions(child, path=path, prefix=prefix)
            continue
        key = f"{prefix}{child.name}"
        own = set(range(child.body[0].lineno, (child.end_lineno or child.lineno) + 1))
        for inner in ast.walk(child):
            if inner is not child and isinstance(
                inner,
                ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            ):
                own -= set(range(inner.lineno, (inner.end_lineno or inner.lineno) + 1))
        yield Definition(
            key=key,
            path=path,
            function=isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef),
            lines=frozenset(own),
        )
        yield from _definitions(child, path=path, prefix=f"{key}.")


def _branch_problems(
    functions: Sequence[Definition],
    *,
    allowed: Mapping[str, str],
    measured: Measured,
) -> list[str]:
    """Report each branch in ``functions`` not taken both ways, less the allowlist."""
    problems: list[str] = []
    used: set[str] = set()
    for definition in functions:
        source = definition.path.read_text().splitlines()
        for line, (total, taken) in measured.branch_stats(str(definition.path)).items():
            if line not in definition.lines or taken >= total:
                continue
            text = source[line - 1].strip()
            # Keyed by the definition as well as the text: one entry must not
            # silence the same line written in another function.
            key = f"{definition.key}::{text}"
            if key in allowed:
                used.add(key)
                continue
            problems.append(
                f"{definition.key}:{line} takes {taken} of {total} exits: {text}",
            )
    problems.extend(
        f"allowlisted branch now goes both ways, or no longer exists: {key}"
        for key in allowed
        if key not in used
    )
    return problems


def _draws(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Return whether ``func`` draws: seeded, and not attention at ``dropout_p == 0``."""
    if torch.Tag.nondeterministic_seeded not in func.tags:
        return False
    names = [a.name for a in func._schema.arguments]  # noqa: SLF001 -- The harness reads the op schema to find the dropout argument.
    if "dropout_p" not in names:
        return True
    index = names.index("dropout_p")
    return kwargs.get("dropout_p", args[index] if index < len(args) else 0.0) != 0


def _caller() -> str:
    """Return ``file:line`` of the innermost frame outside torch and this file."""
    torch_root = Path(torch.__file__).parent
    frame = inspect.currentframe()
    while frame is not None:
        source = Path(frame.f_code.co_filename)
        if source.resolve() != _THIS and not source.is_relative_to(torch_root):
            return f"{source.name}:{frame.f_lineno}"
        frame = frame.f_back
    return "unknown"


def _recorder(into: list[list[Tensor]]) -> Callable[[nn.Module, object, object], None]:
    """Return a forward hook appending each call's detached output tensors to ``into``."""

    def hook(module: nn.Module, args: object, output: object) -> None:
        del module, args
        into.append([t.detach().clone() for t in _tensors(output)])

    return hook


def _tensors(value: object) -> list[Tensor]:
    """Flatten a module output into its tensors, in order."""
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [
            t
            for item in cast("Mapping[str, object]", value).values()
            for t in _tensors(item)
        ]
    if isinstance(value, list | tuple):
        return [t for item in cast("Sequence[object]", value) for t in _tensors(item)]
    return []


def _git(root: Path, *arguments: str) -> str:
    """Return a git command's output in ``root``; on failure, raise with its stderr."""
    completed = subprocess.run(  # noqa: S603 -- The parity harness runs fixed git subcommands against the pinned reference.
        ["git", *arguments],  # noqa: S607 -- The parity harness runs fixed git subcommands against the pinned reference.
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {root}: {completed.stderr.strip()}",
        )
    return completed.stdout.strip()


def _dictdot_stub() -> types.ModuleType:
    """Supply ``dictdot.dictdot``: a dict whose keys read as attributes."""
    module = types.ModuleType("dictdot")

    class dictdot(dict[str, object]):  # noqa: N801 -- The name REG imports.
        def __getattr__(self, name: str) -> object:
            return self[name]

    module.__dict__["dictdot"] = dictdot
    return module


class _InvaeModule(Protocol):
    VAE_F16D32: Callable[[], nn.Module]


class _TheirInvae(Protocol):
    def encode(self, x: Tensor, /) -> _Posterior: ...

    def decode(self, z: Tensor, /) -> _Decoded: ...


class _Posterior(Protocol):
    mean: Tensor
    logvar: Tensor
    std: Tensor
    var: Tensor

    def sample(self) -> Tensor: ...

    def mode(self) -> Tensor: ...


class _Decoded(Protocol):
    sample: Tensor


def _randomize(module: nn.Module, *, seed: int) -> None:
    """Draw every parameter from ``N(0, 0.3 ** 2)``: weights far from any init."""
    generator = torch.Generator().manual_seed(seed)
    # By name: registration order moves between transformers releases.
    with torch.no_grad():
        for _, parameter in sorted(module.named_parameters()):
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)


def _their_rae(
    stage1: _Stage1Module,
    *,
    work: Path,
    decoder: Path | None = None,
    stats: Path | None = None,
) -> nn.Module:
    """Build the reference RAE on the tiny encoder and decoder configs in ``work``."""
    return stage1.RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path=str(work / "enc"),
        encoder_input_size=8,
        encoder_params={"dinov2_path": str(work / "enc"), "normalize": True},
        decoder_config_path=str(work / "dec"),
        decoder_patch_size=4,
        pretrained_decoder_path=None if decoder is None else str(decoder),
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=None if stats is None else str(stats),
    )


def _commit_tensor(reference: Reference) -> Tensor:
    """Return the reference's pinned commit as bytes, the golden's provenance."""
    return torch.frombuffer(bytearray(reference.commit.encode()), dtype=torch.uint8)


class _Stage1Module(Protocol):
    RAE: Callable[..., nn.Module]


class _DecoderModule(Protocol):
    GeneralDecoder: Callable[..., nn.Module]
    DinoV3PixelDecoder: Callable[..., nn.Module]


class _TheirRae(Protocol):
    encoder: _TheirDinov2
    decoder: nn.Module

    def encode(self, x: Tensor, /) -> Tensor: ...

    def decode(self, z: Tensor, /) -> Tensor: ...


class _TheirDinov2(Protocol):
    encoder: _HfEncoder


class _HfEncoder(Protocol):
    def set_attn_implementation(self, name: str, /) -> None: ...


def _their_round_trip(
    their_vtp: _TheirVtp,
    image: Tensor,
    *,
    autocast: torch.dtype | None,
) -> dict[str, Tensor]:
    """Reconstruct as the reference's evaluation, ``tools/test_reconstruction_hf.py``."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize(mean, std)
    denormalize = Normalize(
        [-m / s for m, s in zip(mean, std, strict=True)],
        [1 / s for s in std],
    )
    pixels_in = torch.stack(
        [normalize(to_tensor(img.permute(1, 2, 0).numpy())) for img in image],
    )
    cast_context = (
        nullcontext() if autocast is None else torch.autocast("cpu", dtype=autocast)
    )
    with cast_context:
        latent = their_vtp.get_reconstruction_latents(pixels_in)
    pixels = their_vtp.get_latents_decoded_images(latent.float())
    return {
        "latent": latent,
        "pixels": pixels,
        "image": torch.clamp(denormalize(pixels), 0, 1),
    }


class _VtpHfModule(Protocol):
    VTPModel: Callable[[object], nn.Module]
    VTPConfig: Callable[..., object]


class _TheirVtp(Protocol):
    def get_reconstruction_latents(self, image: Tensor, /) -> Tensor: ...

    def get_latents_decoded_images(self, latents: Tensor, /) -> Tensor: ...


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    _ = parser.add_argument(
        "--model",
        nargs="+",
        choices=sorted(_MODELS),
        default=sorted(_MODELS),
        help="Which ports to compare.",
    )
    _ = parser.add_argument(
        "--clone-dir",
        type=Path,
        default=get_cache_dir() / "reference",
        help="Where the pinned reference clones live.",
    )
    _ = parser.add_argument(
        "--mint",
        action="store_true",
        help="Rewrite the checked-in reference goldens instead of checking them.",
    )


_MODELS: Final = {
    "invae": (_INVAE, run_invae, None),
    "rae": (_RAE, run_rae, "rae_reference.pt"),
    "vtp": (_VTP, run_vtp, "vtp_reference.pt"),
}


class _Flags(Protocol):
    """Parsed command-line flags."""

    model: list[str]
    clone_dir: Path
    mint: bool


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
