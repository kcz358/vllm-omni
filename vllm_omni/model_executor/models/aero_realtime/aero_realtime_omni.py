"""AeroRealtime Omni — top-level dispatcher for the 3-stage pipeline.

Depending on `vllm_config.model_config.model_stage` in {"thinker", "talker", "code2wav"},
this class constructs the corresponding submodule with the correct sub-config, and
delegates weight loading + forward to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsRealtime,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.aero_realtime.aero_realtime import (
    AeroRealtimeDummyInputsBuilder,
    AeroRealtimeMultiModalProcessor,
    AeroRealtimeProcessingInfo,
)
from vllm_omni.transformers_utils.configs.aero_realtime_omni import (
    AeroRealtimeOmniConfig,
    AeroRealtimeTalkerConfig,
)

logger = init_logger(__name__)


class AeroRealtimeOmniProcessingInfo(AeroRealtimeProcessingInfo):
    """Processing info for the omni dispatcher.

    Unwraps ``AeroRealtimeOmniConfig`` → ``thinker_config`` (an
    ``AeroRealtimeConfig``) so all downstream processor logic keeps working
    unchanged.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config(AeroRealtimeOmniConfig).thinker_config


@MULTIMODAL_REGISTRY.register_processor(
    AeroRealtimeMultiModalProcessor,
    info=AeroRealtimeOmniProcessingInfo,
    dummy_inputs=AeroRealtimeDummyInputsBuilder,
)
class AeroRealtimeOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsRealtime,
):
    """Stage-dispatched wrapper for aero_realtime_omni.

    Stages:
      - "thinker"  → AeroRealtimeForConditionalGeneration (omni mode)
      - "talker"   → AeroRealtimeTalkerForConditionalGeneration
      - "code2wav" → Qwen3TTSCode2Wav (reused as-is)
    """

    realtime_max_tokens = 1

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        top_config: AeroRealtimeOmniConfig = vllm_config.model_config.hf_config
        self.config = top_config

        model_stage = getattr(vllm_config.model_config, "model_stage", None) or "thinker"
        self.model_stage = model_stage

        self.thinker = None
        self.talker = None
        self.code2wav = None

        if model_stage == "thinker":
            thinker_config = top_config.thinker_config
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["AeroRealtimeForConditionalGeneration"]
            )
            # Signal to the thinker that it should export hidden states + word embeds.
            setattr(thinker_vllm_config.model_config, "omni_mode", True)
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["AeroRealtimeForConditionalGeneration"],
            )
            self.model = self.thinker
        elif model_stage == "talker":
            talker_config: AeroRealtimeTalkerConfig = top_config.talker_config
            talker_vllm_config = vllm_config.with_hf_config(
                talker_config, architectures=["AeroRealtimeTalkerForConditionalGeneration"]
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["AeroRealtimeTalkerForConditionalGeneration"],
            )
            self.model = self.talker
        elif model_stage == "code2wav":
            # Qwen3TTSCode2Wav ignores hf_config and loads its weights from
            # `<model_path>/speech_tokenizer/`; we pass thinker_config only as
            # a required-but-unused placeholder. The user copies the
            # speech_tokenizer/ folder from the Qwen3-TTS-Base checkpoint into
            # the aero_realtime_omni checkpoint before deployment.
            code2wav_vllm_config = vllm_config.with_hf_config(
                top_config.thinker_config,
                architectures=["Qwen3TTSCode2Wav"],
            )
            self.code2wav = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=top_config.thinker_config,
                architectures=["Qwen3TTSCode2Wav"],
            )
            self.model = self.code2wav
        else:
            raise ValueError(f"Invalid model_stage: {model_stage!r}. Must be thinker | talker | code2wav")

        self.have_multimodal_outputs = getattr(self.model, "have_multimodal_outputs", False)
        self.has_preprocess = getattr(self.model, "has_preprocess", False)
        self.has_postprocess = getattr(self.model, "has_postprocess", False)
        self.streaming_accumulated_keys = getattr(self.model, "streaming_accumulated_keys", set())
        self.gpu_resident_buffer_keys = getattr(self.model, "gpu_resident_buffer_keys", set())
        if hasattr(self.model, "make_empty_intermediate_tensors"):
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids, **kwargs)

    def embed_multimodal(self, **kwargs: Any):
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)
        raise AttributeError(
            f"stage={self.model_stage!r} submodule has no embed_multimodal"
        )

    def get_mrope_input_positions(self, *args: Any, **kwargs: Any):
        if hasattr(self.model, "get_mrope_input_positions"):
            return self.model.get_mrope_input_positions(*args, **kwargs)
        raise AttributeError(
            f"stage={self.model_stage!r} submodule has no get_mrope_input_positions"
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def make_omni_output(self, *args, **kwargs):
        return self.model.make_omni_output(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        return self.model.preprocess(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self.model.postprocess(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route checkpoint weights to the appropriate submodule based on top-level prefix.

        The submodules are registered under their stage-specific attribute names
        (``self.thinker`` / ``self.talker`` / ``self.code2wav``), so PyTorch's
        state_dict layout naturally matches the checkpoint's ``thinker.*`` /
        ``talker.*`` / ``code2wav.*`` prefixes. Each submodule's own
        ``hf_to_vllm_mapper`` handles further name rewrites.

        The thinker's mapper uses unprefixed keys (e.g. ``language_model.``),
        so we strip ``thinker.`` before delegating. The talker's mapper keys
        include the ``talker.`` prefix, so we keep it. code2wav loads from a
        separate ``speech_tokenizer/`` checkpoint and is passed through.
        """
        loaded: set[str] = set()
        thinker_weights: list[tuple[str, torch.Tensor]] = []
        talker_weights: list[tuple[str, torch.Tensor]] = []
        code2wav_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name[len("thinker.") :], w))
            elif name.startswith("talker."):
                talker_weights.append((name, w))
            elif name.startswith("code2wav."):
                code2wav_weights.append((name, w))
            else:
                # Unknown prefix — skip silently for forward compat.
                pass

        if self.thinker is not None and thinker_weights:
            loaded_names = self.thinker.load_weights(thinker_weights)
            loaded |= {f"thinker.{n}" for n in loaded_names}

        if self.talker is not None and talker_weights:
            loaded_names = self.talker.load_weights(talker_weights)
            loaded |= {f"talker.{n}" for n in loaded_names}

        if self.code2wav is not None and code2wav_weights:
            loaded_names = self.code2wav.load_weights(code2wav_weights)
            loaded |= {f"code2wav.{n}" for n in loaded_names}

        return loaded

    @classmethod
    async def buffer_realtime_omni(cls, *args, **kwargs):
        from vllm_omni.model_executor.models.aero_realtime.aero_realtime import (
            AeroRealtimeForConditionalGeneration,
        )
        async for prompt in AeroRealtimeForConditionalGeneration.buffer_realtime_omni(*args, **kwargs):
            yield prompt

    @classmethod
    async def buffer_realtime_audio(cls, *args, **kwargs):
        from vllm_omni.model_executor.models.aero_realtime.aero_realtime import (
            AeroRealtimeForConditionalGeneration,
        )
        async for prompt in AeroRealtimeForConditionalGeneration.buffer_realtime_audio(*args, **kwargs):
            yield prompt


__all__ = ["AeroRealtimeOmniForConditionalGeneration"]
