"""AeroRealtime talker (Qwen3-decoder trunk + text_projection + codec_head + code_predictor).

Pipeline stage 1 of aero_realtime_omni. Consumes accumulated thinker hidden states
at `<|audio_pad|>` positions and emits 16-way codec codes (group 0 from the trunk,
groups 1..15 from a nested code_predictor AR).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
    Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
)
from vllm_omni.transformers_utils.configs.aero_realtime_omni import (
    AeroRealtimeOmniConfig,
    AeroRealtimeTalkerConfig,
)

logger = init_logger(__name__)


class AeroRealtimeTalkerResizeMLP(nn.Module):
    """Two-layer MLP for hidden-dim resize (thinker -> talker)."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class AeroRealtimeTalkerForConditionalGeneration(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "talker.model.layers.": "model.layers.",
            "talker.model.norm.": "model.norm.",
            "talker.model.codec_embedding.": "model.embed_tokens.",
            "talker.codec_head.": "codec_head.",
            "talker.model.text_embedding.": "text_embedding.",
            "talker.text_projection.": "text_projection.",
            "talker.code_predictor.": "code_predictor.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        top_config = vllm_config.model_config.hf_config
        # top_config may be either AeroRealtimeOmniConfig or AeroRealtimeTalkerConfig,
        # depending on whether we are hosted by the omni dispatcher or loaded standalone.
        if isinstance(top_config, AeroRealtimeOmniConfig):
            self.config: AeroRealtimeTalkerConfig = top_config.talker_config
        else:
            self.config = top_config
        talker_config: AeroRealtimeTalkerConfig = self.config

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        # Runner auto-concatenates these keys across streaming chunks. `embed.prefill`
        # is provided by the thinker as thinker-side word-embeddings at audio_pad
        # positions (informational; not consumed by preprocess).
        self.streaming_accumulated_keys: set[tuple[str, str]] = {
            ("embed", "prefill"),
            ("hidden_states", "output"),
            ("codes", "audio"),
            ("codes", "past_group0"),
        }
        # GPU-resident buffer keys (avoid CPU<->GPU round-trips inside the decode loop).
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "last"),
        }

        # Build the trunk from the talker_config, using vllm's Qwen3Model.
        # Qwen2Model reads vllm_config.model_config.hf_config.get_text_config(); for
        # AeroRealtimeTalkerConfig, get_text_config() returns self.
        trunk_vllm_config = vllm_config.with_hf_config(talker_config)
        self.model = Qwen3Model(vllm_config=trunk_vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Codec head: linear projection from talker hidden -> codec vocab logits (group 0).
        if get_pp_group().is_last_rank:
            self.codec_head = ParallelLMHead(
                talker_config.vocab_size,
                talker_config.hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "codec_head"),
            )
        else:
            self.codec_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(talker_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Text projection: thinker hidden -> talker hidden.
        # fc1 in=thinker_hidden_size (2560), fc1 out=text_hidden_size (2048).
        # fc2 in=text_hidden_size (2048), fc2 out=hidden_size (1024).
        self.text_projection = AeroRealtimeTalkerResizeMLP(
            input_size=talker_config.thinker_hidden_size,
            intermediate_size=talker_config.text_hidden_size,
            output_size=talker_config.hidden_size,
            bias=True,
        )

        # Text embedding: parked here purely so load_weights doesn't drop the tensor.
        # Not consumed at inference for aero-realtime pipeline (we use thinker hidden
        # states directly).
        self.text_embedding = nn.Embedding(talker_config.text_vocab_size, talker_config.text_hidden_size)

        # Code predictor for groups 1..15. Reuse Qwen3-TTS's CodePredictorWrapper
        # which already handles the nested AR + CUDA graph capture + sampling.
        predictor_compilation = dataclasses.replace(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        cp_vllm_config = dataclasses.replace(vllm_config, compilation_config=predictor_compilation)

        with set_current_vllm_config(cp_vllm_config):
            self.code_predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
                vllm_config=cp_vllm_config,
                config=talker_config.code_predictor_config,
                talker_config=talker_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )
        self._cp_vllm_config = cp_vllm_config

        # Cache tokens we need often as buffers (avoid CPU->GPU per step).
        speaker_id = int(next(iter(talker_config.speaker_id.values())))
        cond_ids = torch.tensor(
            [talker_config.codec_bos_id, talker_config.codec_nothink_id, speaker_id],
            dtype=torch.long,
        )
        self.register_buffer("_cond_ids", cond_ids, persistent=False)
        self.register_buffer(
            "_codec_bos_id_tensor",
            torch.tensor([talker_config.codec_bos_id], dtype=torch.long),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states, sampling_metadata=None) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.logits_processor(self.codec_head, hidden_states)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build the trunk prompt embeds from accumulated thinker hidden states.

        Prefill branch (span_len > 1): rebuild the full prompt [cond(3) + body(N_total)]
        every chunk since stage-1 KV is cleared per streaming update.
        Decode branch (span_len == 1): assemble the per-step input using the previous
        step's `hidden_states.last` and the last-sampled group0 from postprocess.
        """
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        hs = info_dict.get("hidden_states", {}) or {}
        codes = info_dict.get("codes", {}) or {}
        span_len = int(input_ids.shape[0])
        device = input_ids.device
        dtype = torch.bfloat16
        talker_cfg = self.config

        thinker_hidden_full = hs.get("output")
        past_group0 = codes.get("past_group0")

        if not isinstance(thinker_hidden_full, torch.Tensor):
            raise ValueError(
                "AeroRealtimeTalker.preprocess: missing accumulated hidden_states.output "
                "(expected thinker post-norm hidden at every <|audio_pad|> slot)"
            )

        n_total = int(thinker_hidden_full.shape[0])
        if n_total == 0:
            return input_ids, self.embed_input_ids(input_ids), {}

        if span_len > 1:
            thinker_h = thinker_hidden_full.to(device=device, dtype=dtype)
            text_h = self.text_projection(thinker_h)

            cond_ids = self._cond_ids.to(device=device)
            cond_emb = self.embed_input_ids(cond_ids)

            prev_ids = torch.empty(n_total, dtype=torch.long, device=device)
            prev_ids[0] = talker_cfg.codec_bos_id
            if n_total > 1:
                if isinstance(past_group0, torch.Tensor) and past_group0.numel() >= n_total - 1:
                    prev_ids[1:] = past_group0.to(device=device, dtype=torch.long)[: n_total - 1]
                else:
                    prev_ids[1:] = talker_cfg.codec_bos_id
            prev_emb = self.embed_input_ids(prev_ids)

            body_emb = text_h + prev_emb
            prompt_embeds_full = torch.cat([cond_emb.to(dtype=dtype), body_emb], dim=0)

            meta = info_dict.get("meta", {}) or {}
            total = int(prompt_embeds_full.shape[0])
            if span_len == total:
                offset = 0
            else:
                offset = int(meta.get("talker_prefill_offset", 0) or 0)
            offset = max(0, min(offset, total))
            end = min(offset + span_len, total)
            take = prompt_embeds_full[offset:end]
            if int(take.shape[0]) < span_len:
                pad_n = span_len - int(take.shape[0])
                pad_row = prompt_embeds_full[-1:].expand(pad_n, -1)
                take = torch.cat([take, pad_row], dim=0)

            input_ids_out = input_ids.clone()
            input_ids_out[:] = int(talker_cfg.codec_pad_id)

            info_update: dict[str, Any] = {
                "meta": {"talker_prefill_offset": offset + span_len},
                "codes": {
                    "audio": torch.zeros(
                        (span_len, int(talker_cfg.num_code_groups)),
                        dtype=torch.long,
                        device=device,
                    )
                },
            }
            return input_ids_out, take, info_update

        # Decode branch (span_len == 1).
        thinker_h_last = thinker_hidden_full[-1:].to(device=device, dtype=dtype)
        text_h_last = self.text_projection(thinker_h_last)

        if isinstance(past_group0, torch.Tensor) and past_group0.numel() > 0:
            prev_ids = past_group0[-1:].to(device=device, dtype=torch.long)
        else:
            prev_ids = self._codec_bos_id_tensor.to(device=device)
        prev_emb = self.embed_input_ids(prev_ids)

        body_emb = (text_h_last + prev_emb).reshape(1, -1)

        input_ids_out = input_ids.clone()
        input_ids_out[:] = int(talker_cfg.codec_pad_id)

        info_update = {
            "codes": {
                "audio": torch.zeros(
                    (1, int(talker_cfg.num_code_groups)),
                    dtype=torch.long,
                    device=device,
                )
            },
        }
        return input_ids_out, body_emb, info_update

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        sampled_token_ids: torch.Tensor | None = None,
        **info_dict: Any,
    ) -> dict[str, Any]:
        """After trunk produces last hidden + group0 is sampled, run the nested code_predictor AR."""
        talker_cfg = self.config
        if hidden_states is None or hidden_states.numel() == 0:
            return {}

        last_hidden = hidden_states[-1:, :].detach()  # [1, hidden]
        if sampled_token_ids is None:
            frame = torch.zeros(
                (1, int(talker_cfg.num_code_groups)),
                dtype=torch.long,
                device=hidden_states.device,
            )
            return {
                "hidden_states": {"last": last_hidden},
                "codes": {
                    "audio": frame,
                    "past_group0": torch.zeros((1,), dtype=torch.long, device=hidden_states.device),
                },
            }

        # sampled_token_ids: [1] on GPU, dtype long. This is group0 for the last frame.
        group0_id = sampled_token_ids.reshape(-1)[-1:].to(dtype=torch.long, device=hidden_states.device)

        # Nested 15-step code_predictor AR via CodePredictorWrapper.forward().
        # The wrapper returns [B, num_groups] with layer0 at column 0 and residuals at 1..G-1.
        layer0_embed = self.embed_input_ids(group0_id).reshape(1, 1, -1)
        past_hidden = last_hidden.reshape(1, 1, -1)
        audio_codes = self.code_predictor(
            layer0_code=group0_id.reshape(1, 1),
            layer0_embed=layer0_embed,
            last_talker_hidden=past_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
        )  # [1, num_code_groups]
        frame = audio_codes.reshape(1, -1).to(dtype=torch.long)

        return {
            "hidden_states": {"last": last_hidden},
            "codes": {
                "audio": frame,
                "past_group0": group0_id.reshape(1),
            },
        }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load talker.* weights.

        The vLLM `CodePredictorWrapper` inside `self.code_predictor` has its own weight
        loader. We split the weight stream so the code_predictor sees its own weights
        (with the `code_predictor.` prefix stripped) and everything else goes through
        the standard `AutoWeightsLoader`.
        """
        cp_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            mapped = self.hf_to_vllm_mapper._map_name(name)
            if mapped is None:
                continue
            if mapped.startswith("code_predictor."):
                cp_weights.append((mapped[len("code_predictor.") :], w))
            else:
                other_weights.append((name, w))

        loader = AutoWeightsLoader(self, skip_prefixes=["code_predictor."])
        loaded = loader.load_weights(other_weights, mapper=self.hf_to_vllm_mapper)
        cp_loaded = self.code_predictor.load_weights(iter(cp_weights))
        return loaded | {f"code_predictor.{n}" for n in cp_loaded}


__all__ = ["AeroRealtimeTalkerForConditionalGeneration", "AeroRealtimeTalkerResizeMLP"]
