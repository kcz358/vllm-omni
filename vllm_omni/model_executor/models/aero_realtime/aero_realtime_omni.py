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

from vllm_omni.transformers_utils.configs.aero_realtime_omni import (
    AeroRealtimeOmniConfig,
    AeroRealtimeTalkerConfig,
)

logger = init_logger(__name__)


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

        if model_stage == "thinker":
            thinker_config = top_config.thinker_config
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["AeroRealtimeForConditionalGeneration"]
            )
            # Signal to the thinker that it should export hidden states + word embeds.
            setattr(thinker_vllm_config.model_config, "omni_mode", True)
            self.model = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["AeroRealtimeForConditionalGeneration"],
            )
        elif model_stage == "talker":
            talker_config: AeroRealtimeTalkerConfig = top_config.talker_config
            talker_vllm_config = vllm_config.with_hf_config(
                talker_config, architectures=["AeroRealtimeTalkerForConditionalGeneration"]
            )
            self.model = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["AeroRealtimeTalkerForConditionalGeneration"],
            )
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
            self.model = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=top_config.thinker_config,
                architectures=["Qwen3TTSCode2Wav"],
            )
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
        # Filter to the prefix owned by this stage before delegating.
        if self.model_stage == "thinker":
            keep_prefix = "thinker."
            strip = True
        elif self.model_stage == "talker":
            keep_prefix = "talker."
            strip = False  # AeroRealtimeTalker's hf_to_vllm_mapper expects the "talker." prefix.
        else:  # code2wav — weights come from a separate speech_tokenizer/ checkpoint at load time.
            keep_prefix = None
            strip = False

        if keep_prefix is None:
            return self.model.load_weights(weights)

        def _filter():
            for name, w in weights:
                if not name.startswith(keep_prefix):
                    continue
                if strip:
                    yield name[len(keep_prefix) :], w
                else:
                    yield name, w

        return self.model.load_weights(_filter())

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
