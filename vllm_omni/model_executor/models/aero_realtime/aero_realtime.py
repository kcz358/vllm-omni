from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoVideoProcessor, BatchFeature
from transformers.activations import ACT2FN
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import ModelConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.inputs import MultiModalDataDict, PromptType, TokensPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsRealtime,
    _require_is_multimodal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLVideoInputs,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioFeatureInputs,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3LLMForCausalLM,
    Qwen3VLForConditionalGeneration,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    Qwen3_VisionTransformer,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.model_executor.model_loader.weight_utils import default_weight_loader as default_weight_loader_audio
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder, PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.collection_utils import is_list_of
from vllm.utils.tensor_schema import TensorShape
from typing import Annotated

from vllm_omni.transformers_utils.configs.aero_realtime import AeroRealtimeConfig


logger = init_logger(__name__)


class AeroRealtimeAudioFeatureInputs(Qwen2AudioFeatureInputs):
    input_features: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("na", "nmb", "t_mel")]
    feature_attention_mask: Annotated[torch.Tensor, TensorShape("na", "t_enc")]
    audio_chunks_per_item: Annotated[torch.Tensor | None, TensorShape("n")] = None


class AeroRealtimeMultiModalProjector(nn.Module):
    def __init__(self, config: AeroRealtimeConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_hidden_size * config.downsample_factor,
            config.text_config.hidden_size,
            bias=False,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(audio_features)))


class AeroRealtimeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(AeroRealtimeConfig)

    def get_hf_processor(self, **kwargs: object):
        from vllm_omni.transformers_utils.processors.aero_realtime import AeroRealtimeProcessor

        try:
            processor = self.ctx.get_hf_processor(AeroRealtimeProcessor, **kwargs)
        except Exception:
            processor = AeroRealtimeProcessor(
                image_processor=AutoImageProcessor.from_pretrained(self.model_id, **kwargs),
                video_processor=AutoVideoProcessor.from_pretrained(self.model_id, **kwargs),
                feature_extractor=AutoFeatureExtractor.from_pretrained(self.model_id, **kwargs),
                tokenizer=self.get_tokenizer(),
            )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "audio_start_token"):
            processor.audio_start_token = "<|audio_start|>"
        if not hasattr(processor, "audio_end_token"):
            processor.audio_end_token = "<|audio_end|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, **kwargs)
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


class AeroRealtimeDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        parts: list[str] = []
        parts.extend([f"{processor.vision_start_token}{processor.image_token}{processor.vision_end_token}"] * mm_counts.get("image", 0))
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        parts.extend([f"{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}"] * num_videos)
        parts.extend([processor.audio_token] * max(num_audios - num_videos, 0))
        return "".join(parts)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data = {}
        if (num_audios := mm_counts.get("audio", 0)) > 0:
            feature_extractor = self.info.get_feature_extractor()
            data["audio"] = self._get_dummy_audios(
                length=feature_extractor.sampling_rate,
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        if (num_images := mm_counts.get("image", 0)) > 0:
            data["image"] = self._get_dummy_images(width=32, height=32, num_images=num_images)
        if (num_videos := mm_counts.get("video", 0)) > 0:
            data["video"] = self._get_dummy_videos(width=32, height=32, num_frames=2, num_videos=num_videos)
        return data


def _aero_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths")
    config: dict[str, MultiModalFieldConfig] = {}
    if "pixel_values" in hf_inputs:
        config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", hf_inputs["image_grid_thw"].prod(-1))
        config["image_grid_thw"] = MultiModalFieldConfig.batched("image")
    if "pixel_values_videos" in hf_inputs:
        config["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
            "video", hf_inputs["video_grid_thw"].prod(-1)
        )
        config["video_grid_thw"] = MultiModalFieldConfig.batched("video")
        if "timestamps" in hf_inputs:
            config["timestamps"] = MultiModalFieldConfig.batched("video")
    if "input_features" in hf_inputs:
        chunks_per_item = hf_inputs.get("audio_chunks_per_item")
        if chunks_per_item is not None:
            config["input_features"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", chunks_per_item
            )
            config["feature_attention_mask"] = MultiModalFieldConfig.flat_from_sizes(
                "audio", chunks_per_item
            )
            config["audio_chunks_per_item"] = MultiModalFieldConfig.batched("audio")
        else:
            config["input_features"] = MultiModalFieldConfig.batched("audio")
            config["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")
        if audio_feature_lengths is not None:
            config["audio_feature_lengths"] = MultiModalFieldConfig.batched("audio")
    return config


class AeroRealtimeMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None) or mm_data.pop("audio", None) or []

        if audios:
            hf_inputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
            hf_processor = self.info.get_hf_processor(**mm_kwargs)
            sr = hf_processor.feature_extractor.sampling_rate

            short_idx, long_idx = [], []
            for i, a in enumerate(audios):
                n_tok, _ = AeroRealtimeForConditionalGeneration._get_audio_token_count(hf_processor, a, sr)
                (short_idx if n_tok <= 1 else long_idx).append(i)

            short_out = self.info.ctx.call_hf_processor(
                hf_processor,
                dict(text="", audio=[audios[i] for i in short_idx]),
                dict(**mm_kwargs, **tok_kwargs),
            ) if short_idx else None
            long_out = self.info.ctx.call_hf_processor(
                hf_processor,
                dict(text="", audio=[audios[i] for i in long_idx]),
                dict(**mm_kwargs, **tok_kwargs),
            ) if long_idx else None

            is_chunked = getattr(hf_processor, "chunk_audio", False)

            per_item_feats: list[torch.Tensor] = [None] * len(audios)
            per_item_fam: list[torch.Tensor] = [None] * len(audios)
            per_item_chunks: list[int] = [0] * len(audios)

            for idxs, out in ((short_idx, short_out), (long_idx, long_out)):
                if out is None:
                    continue
                feats = torch.as_tensor(out["input_features"])
                fam = torch.as_tensor(out.get("feature_attention_mask", out["audio_attention_mask"]))
                b = len(idxs)
                n_padded = fam.shape[0] // b
                chunk_valid = fam.any(dim=-1).view(b, n_padded)
                feats = feats.view(b, n_padded, *feats.shape[1:])
                fam = fam.view(b, n_padded, fam.shape[-1])
                for j, idx in enumerate(idxs):
                    valid = chunk_valid[j]
                    per_item_feats[idx] = feats[j][valid]
                    per_item_fam[idx] = fam[j][valid]
                    per_item_chunks[idx] = int(valid.sum().item())

            hf_inputs["input_features"] = torch.cat(per_item_feats, dim=0)
            hf_inputs["feature_attention_mask"] = torch.cat(per_item_fam, dim=0)
            if is_chunked:
                hf_inputs["audio_chunks_per_item"] = torch.as_tensor(per_item_chunks, dtype=torch.long)
            else:
                hf_inputs["audio_feature_lengths"] = hf_inputs["feature_attention_mask"].sum(-1)
            return hf_inputs

        hf_inputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _aero_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        parent_updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        merge_length = image_processor.merge_size**2
        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            return parent_updates

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_frames = int(grid_thw[0])
            tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length

            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                from vllm.multimodal.evs import compute_retained_tokens_count

                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=tokens_per_frame_base,
                    num_frames=num_frames,
                    q=video_pruning_rate,
                )
                tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
            else:
                tokens_per_frame = [tokens_per_frame_base] * num_frames

            # Aero realtime already wraps video tokens inside
            # <timestamp><|vision_start|><|audio_start|> ... <|audio_end|><|vision_end|>.
            return [video_token_id for count in tokens_per_frame for _ in range(count)]

        # Post-conv2 lengths per sample; LM token count = T_enc // df.
        downsample_factor = self.info.get_hf_config().downsample_factor
        is_chunked = getattr(hf_processor, "chunk_audio", False)
        chunks_per_item = out_mm_data.get("audio_chunks_per_item")
        if is_chunked and chunks_per_item is not None:
            n_per_item_t = torch.as_tensor(chunks_per_item)
            audio_output_lens = n_per_item_t.to(torch.long) * downsample_factor
        elif is_chunked and feature_attention_mask is not None:
            fam = torch.as_tensor(feature_attention_mask)
            b = mm_items.get_count("audio")
            n_per_item = fam.shape[0] // b
            audio_output_lens = torch.full(
                (b,), n_per_item * downsample_factor, dtype=torch.long
            )
        elif audio_feature_lengths is not None:
            audio_output_lens = torch.as_tensor(audio_feature_lengths)
        elif feature_attention_mask is not None:
            audio_output_lens = torch.as_tensor(feature_attention_mask).sum(-1)
        else:
            audio_output_lens = torch.empty(0, dtype=torch.long)

        audio_output_lengths = (audio_output_lens // downsample_factor).tolist()

        def get_replacement_audio(item_idx: int):
            return [audio_token_id] * audio_output_lengths[item_idx]

        audio_update = PromptReplacement(
            modality="audio",
            target="<|audio_pad|>",
            replacement=get_replacement_audio,
        )
        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=hf_processor.video_token,
                replacement=get_video_replacement,
            ),
            audio_update,
        ]


@dataclass
class AeroRealtimeStreamState:
    last_generated_token_id: int | None = None
    current_time_seconds: float = 0.0
    video_chunk_cursor: int = 0
    chat_started: bool = False
    audio_segment_open: bool = False
    text_stream_ids: list[int] = field(default_factory=list)


@dataclass
class AeroRealtimeChunk:
    audio: np.ndarray | None = None
    video: object | None = None
    timestamp: float | None = None


@MULTIMODAL_REGISTRY.register_processor(
    AeroRealtimeMultiModalProcessor,
    info=AeroRealtimeProcessingInfo,
    dummy_inputs=AeroRealtimeDummyInputsBuilder,
)
class AeroRealtimeForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsRealtime,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Language model + vision (unchanged).
            "model.language_model.": "language_model.model.",
            "language_model.": "language_model.model.",
            "model.vision_tower.": "visual.",
            "vision_tower.": "visual.",
            # Multi-modal projector (unchanged).
            "model.multi_modal_projector.": "multi_modal_projector.",
            # Audio tower: rewrite Aero HF-style names to the chunked
            # ``AeroRealtimeAudioEncoder`` paths. q/k/v -> qkv_proj and
            # fc1+gate/up -> gate_up_proj are handled by packed_modules.
            "model.audio_tower.embedder.": "audio_tower.",
            "audio_tower.embedder.": "audio_tower.",
            "model.audio_tower.norm.": "audio_tower.layer_norm.",
            "audio_tower.norm.": "audio_tower.layer_norm.",
            "model.audio_tower.layers.": "audio_tower.layers.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    realtime_max_tokens = 1
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
        **_: object,
    ) -> tuple[torch.Tensor, int]:
        vision_features = [feature for feature in mm_features if feature.modality in {"image", "video"}]
        if vision_features:
            llm_pos_ids_list = []
            st = 0
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            for feature in sorted(vision_features, key=lambda f: f.mm_position.offset):
                offset = feature.mm_position.offset
                actual_num_tokens = feature.mm_position.length
                if feature.modality == "image":
                    t, h, w = feature.data["image_grid_thw"].data.tolist()
                    assert t == 1, f"Image must have 1 frame, got {t}"
                else:
                    t, h, w = feature.data["video_grid_thw"].data.tolist()

                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                text_len = offset - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

                expected_tokens_per_frame = llm_grid_h * llm_grid_w
                if actual_num_tokens > expected_tokens_per_frame:
                    num_logical_frames = actual_num_tokens // expected_tokens_per_frame
                    remainder = actual_num_tokens % expected_tokens_per_frame
                    for _ in range(num_logical_frames):
                        grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                        llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                        st_idx = llm_pos_ids_list[-1].max() + 1
                        text_len = 0
                    if remainder > 0:
                        full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                        llm_pos_ids_list.append(full_grid[:, :remainder] + text_len + st_idx)
                else:
                    grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    llm_pos_ids_list.append(grid_indices[:, :actual_num_tokens] + text_len + st_idx)

                st = offset + actual_num_tokens

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
            return torch.from_numpy(llm_positions), mrope_position_delta

        seq_len = len(input_tokens)
        positions = torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1)
        return positions.clone(), 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config: AeroRealtimeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = AeroRealtimeAudioEncoder(
                config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )
            self.multi_modal_projector = AeroRealtimeMultiModalProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3LLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.start_layer >= len(config.vision_config.deepstack_visual_indexes)

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.has_preprocess = True

    @staticmethod
    def build_audio_realtime_text_stream_ids(
        input_ids: list[int],
        state: AeroRealtimeStreamState,
        *,
        audio_token_id: int,
        rt_pad_id: int,
    ) -> list[int]:
        """For each audio_pad slot in `input_ids`:
          - if `state.last_generated_token_id` is set: consume and write it
          - else: write `rt_pad`
        Non-audio_pad positions keep their original id.
        Spec invariant I5.
        """
        stream_ids = list(input_ids)
        for idx, token_id in enumerate(input_ids):
            if token_id != audio_token_id:
                continue
            if state.last_generated_token_id is not None:
                stream_ids[idx] = state.last_generated_token_id
                state.last_generated_token_id = None
            else:
                stream_ids[idx] = rt_pad_id
        return stream_ids

    @staticmethod
    def _expand_first_token(input_ids: list[int], token_id: int, count: int) -> list[int]:
        expanded: list[int] = []
        replaced = False
        for input_id in input_ids:
            if input_id == token_id and not replaced:
                expanded.extend([token_id] * max(count, 1))
                replaced = True
            else:
                expanded.append(input_id)
        return expanded

    @staticmethod
    async def _drain_generated_tokens(
        input_stream: asyncio.Queue[list[int]],
        state: AeroRealtimeStreamState,
    ) -> None:
        """Lockstep with the sampler: from the second chunk onward, block
        until the previous step's sampled token arrives. Keep the last id
        as ``state.last_generated_token_id`` for teacher forcing on the
        next ``audio_pad`` slot, then drain any extras non-blockingly."""
        if state.chat_started:
            token_ids = await input_stream.get()
            if token_ids:
                state.last_generated_token_id = token_ids[-1]
        while not input_stream.empty():
            token_ids = input_stream.get_nowait()
            if token_ids:
                state.last_generated_token_id = token_ids[-1]

    @staticmethod
    def _get_audio_token_count(
        processor: object,
        audio_chunk: np.ndarray,
        sampling_rate: int,
    ) -> tuple[int, int]:
        """Compute (num_audio_lm_tokens, num_mel_frames) for one audio chunk.

        Closed-form mel_frames derivation so we don't run the FE STFT for every
        80ms chunk (~45k calls per 1h video). Voxtral FE uses
        ``attention_mask[:, win_length-1::hop_length]`` to get mel_frames, and
        with ``padding="longest"`` no padding happens, so this reduces to
        ``mel = (n_samples - win_length) // hop_length + 1`` (or 0 if shorter).
        Verified numerically equivalent to the FE for arbitrary chunk lengths.
        """
        feature_extractor = processor.feature_extractor
        win_length = int(getattr(feature_extractor, "win_length", 400))
        hop_length = int(getattr(feature_extractor, "hop_length", 160))
        n_samples = int(np.asarray(audio_chunk).shape[-1])
        if n_samples < win_length:
            mel_frames = 0
        else:
            mel_frames = (n_samples - win_length) // hop_length + 1
        chunk_mel = int(processor.audio_length_per_tok)
        num_audio_tokens = (mel_frames + chunk_mel - 1) // chunk_mel
        return num_audio_tokens, mel_frames

    @staticmethod
    def _get_video_token_count(processor: object, video: object) -> int:
        video_processor = getattr(processor, "video_processor")
        video_metadata = None
        if isinstance(video, tuple) and len(video) == 2:
            video, video_metadata = video
        video_kwargs: dict[str, object] = {"return_metadata": True}
        if video_metadata is not None:
            video_kwargs["video_metadata"] = [video_metadata]
        video_inputs = video_processor(videos=video, **video_kwargs)
        video_grid_thw = video_inputs.get("video_grid_thw")
        if video_grid_thw is None:
            raise ValueError("Aero realtime video chunk did not produce video_grid_thw")
        grid = torch.as_tensor(video_grid_thw)
        if grid.ndim == 2:
            grid = grid[0]
        merge_size = getattr(video_processor, "merge_size", 2)
        return int(grid.prod().item()) // int(merge_size**2)

    @classmethod
    def _build_realtime_delta(
        cls,
        tokenizer: object,
        state: AeroRealtimeStreamState,
        *,
        num_audio_tokens: int,
        video_token_id: int,
        num_video_tokens: int = 0,
        text_prefix: str = "",
        timestamp: float | None = None,
        is_last_chunk: bool = False,
    ) -> list[int]:
        """Build delta prompt token ids for a single streaming update.

        Streaming layout follows Aero's separated-envelope design:

        - First update opens ``<|im_start|>user\n``.
        - Video deltas emit ``<t.t seconds><|vision_start|><|video_pad|>*S<|vision_end|>``.
        - Audio deltas emit ``<|audio_start|>`` on segment open then
          one ``<|audio_pad|>`` per chunk while audio is active (spec I1+I5);
          subsequent audio-only deltas only append ``<|audio_pad|>``.
        - When a non-audio event arrives while an audio segment is open we
          first append ``<|audio_end|>`` to close the segment.
        - Final update appends ``<|audio_end|><|im_end|>\n`` if still open.
        """
        parts: list[str] = []

        if not state.chat_started:
            parts.append(
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                "<|im_start|>user\n"
            )
            state.chat_started = True

        if text_prefix:
            if state.audio_segment_open:
                parts.append("<|audio_end|>")
                state.audio_segment_open = False
            parts.append(text_prefix)

        if num_video_tokens > 0:
            if state.audio_segment_open:
                parts.append("<|audio_end|>")
                state.audio_segment_open = False
            if timestamp is not None:
                parts.append(f"<{timestamp:.1f} seconds>")
            parts.append("<|vision_start|>")
            parts.append("<|video_pad|>")
            parts.append("<|vision_end|>")

        if num_audio_tokens > 0:
            if not state.audio_segment_open:
                parts.append("<|audio_start|>")
                state.audio_segment_open = True
            parts.append("<|audio_pad|>")

        if is_last_chunk:
            if state.audio_segment_open:
                parts.append("<|audio_end|>")
                state.audio_segment_open = False
            parts.append("<|im_end|>\n")

        if not parts:
            return []

        # Keep prompt_ids un-expanded for both video_pad and audio_pad; vllm
        # expands video_pad via get_video_replacement, and audio stays single
        # slot per chunk (spec I1+I5). Text-stream construction expands video
        # locally before pairing.
        return tokenizer.encode("".join(parts))

    @staticmethod
    def _chunk_audio(chunk: object) -> np.ndarray | None:
        if isinstance(chunk, AeroRealtimeChunk):
            return chunk.audio
        if isinstance(chunk, Mapping):
            audio = chunk.get("audio")
            return None if audio is None else np.asarray(audio, dtype=np.float32)
        if isinstance(chunk, tuple) and chunk:
            audio = chunk[0]
            return None if audio is None else np.asarray(audio, dtype=np.float32)
        return None if chunk is None else np.asarray(chunk, dtype=np.float32)

    @staticmethod
    def _chunk_video(chunk: object) -> object | None:
        if isinstance(chunk, AeroRealtimeChunk):
            return chunk.video
        if isinstance(chunk, Mapping):
            return chunk.get("video")
        if isinstance(chunk, tuple) and len(chunk) > 1:
            return chunk[1]
        return None

    @staticmethod
    def _chunk_text(chunk: object) -> str:
        if isinstance(chunk, Mapping):
            text = chunk.get("text", "")
            return str(text) if text else ""
        return ""

    @staticmethod
    def _chunk_timestamp(chunk: object) -> float | None:
        if isinstance(chunk, AeroRealtimeChunk):
            return chunk.timestamp
        if isinstance(chunk, Mapping):
            timestamp = chunk.get("timestamp")
            return float(timestamp) if timestamp is not None else None
        return None

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        rt_pad_id = tokenizer.convert_tokens_to_ids("<|rt_pad|>")
        state = AeroRealtimeStreamState()
        async for audio_chunk in audio_stream:
            await cls._drain_generated_tokens(input_stream, state)
            num_audio_tokens, _ = cls._get_audio_token_count(
                processor,
                audio_chunk,
                sampling_rate,
            )
            prompt_ids = cls._build_realtime_delta(
                tokenizer,
                state,
                num_audio_tokens=num_audio_tokens,
                video_token_id=video_token_id,
            )
            audio_duration = float(len(audio_chunk)) / float(sampling_rate)
            chunk_text_stream_ids = cls.build_audio_realtime_text_stream_ids(
                prompt_ids,
                state,
                audio_token_id=audio_token_id,
                rt_pad_id=rt_pad_id,
            )
            state.current_time_seconds += audio_duration
            yield OmniTokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data={"audio": audio_chunk},
                additional_information={
                    "ids": {"text_stream_ids": list(chunk_text_stream_ids)},
                    "meta": {"aero_realtime": True},
                },
            )

    @classmethod
    async def buffer_realtime_omni(
        cls,
        chunk_stream: AsyncGenerator[object, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
        state: "AeroRealtimeStreamState | None" = None,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        rt_pad_id = tokenizer.convert_tokens_to_ids("<|rt_pad|>")
        if state is None:
            state = AeroRealtimeStreamState()

        async for chunk in chunk_stream:
            audio_chunk = cls._chunk_audio(chunk)
            if audio_chunk is None or audio_chunk.size == 0:
                logger.warning("Skipping Aero realtime chunk without audio")
                continue
            video_chunk = cls._chunk_video(chunk)
            text_prefix = cls._chunk_text(chunk)
            timestamp = cls._chunk_timestamp(chunk)
            await cls._drain_generated_tokens(input_stream, state)

            if timestamp is None and video_chunk is not None:
                timestamp = state.current_time_seconds
            num_audio_tokens, _ = cls._get_audio_token_count(
                processor,
                audio_chunk,
                sampling_rate,
            )
            num_video_tokens = cls._get_video_token_count(processor, video_chunk) if video_chunk is not None else 0
            prompt_ids = cls._build_realtime_delta(
                tokenizer,
                state,
                num_audio_tokens=num_audio_tokens,
                video_token_id=video_token_id,
                num_video_tokens=num_video_tokens,
                text_prefix=text_prefix,
                timestamp=timestamp,
            )
            audio_duration = float(len(audio_chunk)) / float(sampling_rate)
            # Expand video_pad locally for text_stream pairing only;
            # prompt_token_ids stays un-expanded so vllm can expand it itself.
            expanded_for_stream = (
                cls._expand_first_token(prompt_ids, video_token_id, num_video_tokens)
                if num_video_tokens > 0
                else prompt_ids
            )
            chunk_text_stream_ids = cls.build_audio_realtime_text_stream_ids(
                expanded_for_stream,
                state,
                audio_token_id=audio_token_id,
                rt_pad_id=rt_pad_id,
            )
            state.current_time_seconds += audio_duration
            multi_modal_data: dict[str, object] = {"audio": audio_chunk}
            mm_processor_kwargs = chunk.get("mm_processor_kwargs", {}) if isinstance(chunk, Mapping) else {}
            if video_chunk is not None:
                multi_modal_data["video"] = video_chunk
            yield OmniTokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                additional_information={
                    "ids": {"text_stream_ids": list(chunk_text_stream_ids)},
                    "meta": {"aero_realtime": True},
                },
            )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> AeroRealtimeAudioFeatureInputs | None:
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        audio_chunks_per_item = kwargs.pop("audio_chunks_per_item", None)
        if input_features is None:
            return None
        return AeroRealtimeAudioFeatureInputs(
            type="audio_features",
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_chunks_per_item=audio_chunks_per_item,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict[str, object]:
        mm_input_by_modality: dict[str, object] = {}
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            mm_input_by_modality["image"] = image_input
        video_input = self._parse_and_validate_video_input(**kwargs)
        if video_input is not None:
            mm_input_by_modality["video"] = video_input
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is not None:
            mm_input_by_modality["audio"] = audio_input
        return mm_input_by_modality

    def _process_audio_input(self, audio_input: AeroRealtimeAudioFeatureInputs) -> tuple[torch.Tensor, ...]:
        """Run chunked mel features through the audio encoder + projector.

        Processor pre-chunks audio into ``[na_total, F, chunk_mel]`` where
        ``na_total`` is the total number of LM audio tokens across all audio
        items in the batch and each row encodes one 80ms chunk. The encoder
        runs each chunk independently (no cross-chunk attention, no KV cache)
        and returns ``[na_total, chunk_enc, hidden]``. ``chunk_enc`` equals
        ``downsample_factor`` (chunk_mel = 2 * df), so we concatenate frames
        along the hidden dim and project to LM hidden size.
        """
        input_features = audio_input["input_features"]
        chunks_per_item = audio_input.get("audio_chunks_per_item")
        conv_dtype = self.audio_tower.conv1.weight.dtype
        per_item_sizes: list[int] | None = None
        if isinstance(input_features, (list, tuple)):
            per_item_sizes = [int(t.shape[0]) for t in input_features]
            x = torch.cat([t.to(conv_dtype) for t in input_features], dim=0)
        else:
            x = input_features.to(conv_dtype)
            if x.ndim == 4:
                per_item_sizes = [int(x.shape[1])] * int(x.shape[0])
                x = x.reshape(-1, x.shape[-2], x.shape[-1])
            elif x.ndim == 3:
                if chunks_per_item is not None:
                    per_item_sizes = torch.as_tensor(chunks_per_item).reshape(-1).tolist()
                else:
                    per_item_sizes = [1] * int(x.shape[0])
        if x.shape[0] == 0:
            lm_hidden = self.config.text_config.hidden_size
            return (input_features.new_zeros((0, lm_hidden)),)

        hidden = self.audio_tower(x)  # [na_total, chunk_enc, hidden]
        na_total, chunk_enc, hidden_size = hidden.shape
        df = int(self.config.downsample_factor)
        if chunk_enc != df:
            raise ValueError(
                f"Aero audio encoder: chunk_enc ({chunk_enc}) must equal downsample_factor ({df})"
            )
        fat = hidden.reshape(na_total, hidden_size * df)
        proj = self.multi_modal_projector(fat)
        if per_item_sizes is not None and len(per_item_sizes) > 1:
            return tuple(proj.split(per_item_sizes))
        return (proj,)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None
        embeddings: list[torch.Tensor] = []
        if "image" in mm_input_by_modality:
            image_embeddings = self._process_image_input(mm_input_by_modality["image"])
            image_embeddings = self._postprocess_image_embeds_evs(image_embeddings, mm_input_by_modality["image"])
            embeddings.extend(image_embeddings)
        if "video" in mm_input_by_modality:
            video_embeddings = self._process_video_input(mm_input_by_modality["video"])
            video_embeddings = self._postprocess_video_embeds_evs(video_embeddings, mm_input_by_modality["video"])
            embeddings.extend(video_embeddings)
        if "audio" in mm_input_by_modality:
            embeddings.extend(self._process_audio_input(mm_input_by_modality["audio"]))
        return tuple(embeddings)

    @staticmethod
    def _take_embeddings(
        embeddings: Sequence[torch.Tensor], start: int, token_count: int
    ) -> tuple[list[torch.Tensor], int]:
        taken: list[torch.Tensor] = []
        remaining = token_count
        idx = start
        while idx < len(embeddings) and remaining > 0:
            emb = embeddings[idx]
            taken.append(emb)
            remaining -= emb.shape[0]
            idx += 1
        if remaining != 0:
            raise ValueError("Multimodal embedding/token count mismatch")
        return taken, idx

    @staticmethod
    def _iter_multimodal_token_runs(
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> Iterable[tuple[int, int]]:
        positions = mask.nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            return
        start = prev = positions[0]
        prev_token_id = int(input_ids[prev].item())
        for pos in positions[1:]:
            token_id = int(input_ids[pos].item())
            if pos == prev + 1 and token_id == prev_token_id:
                prev = pos
                continue
            yield start, prev + 1
            start = prev = pos
            prev_token_id = token_id
        yield start, prev + 1

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)
        image_mask = is_multimodal & (input_ids == self.config.image_token_id)
        video_mask = is_multimodal & (input_ids == self.config.video_token_id)
        audio_mask = is_multimodal & (input_ids == self.config.audio_token_id)
        mm_mask = image_mask | video_mask | audio_mask

        runs: list[tuple[int, int, int, torch.Tensor]] = []
        vision_embeddings: list[torch.Tensor] = []
        emb_idx = 0
        for start, end in self._iter_multimodal_token_runs(input_ids, mm_mask):
            run_len = end - start
            token_id = int(input_ids[start].item())
            embeds, emb_idx = self._take_embeddings(multimodal_embeddings, emb_idx, run_len)
            run_embeds = torch.cat(embeds, dim=0).to(inputs_embeds.dtype)
            runs.append((start, end, token_id, run_embeds))
            if token_id in (self.config.image_token_id, self.config.video_token_id):
                vision_embeddings.append(run_embeds)

        processed_vision_embeddings: tuple[torch.Tensor, ...] = tuple(vision_embeddings)
        if self.use_deepstack and vision_embeddings:
            deepstack_input_embeds, processed_vision_embeddings = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=tuple(vision_embeddings),
                is_multimodal=image_mask | video_mask,
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        vision_idx = 0
        for start, end, token_id, run_embeds in runs:
            if token_id == self.config.audio_token_id:
                inputs_embeds[start:end] = inputs_embeds[start:end] + run_embeds
            else:
                if token_id in (self.config.image_token_id, self.config.video_token_id):
                    run_embeds = processed_vision_embeddings[vision_idx].to(inputs_embeds.dtype)
                    vision_idx += 1
                inputs_embeds[start:end] = run_embeds

        if emb_idx != len(multimodal_embeddings):
            logger.debug(
                "Unused multimodal embeddings after scheduled token embedding: used=%d total=%d",
                emb_idx,
                len(multimodal_embeddings),
            )
        return inputs_embeds

    @staticmethod
    def _get_text_stream_ids(info_dict: Mapping[str, Any]) -> Any | None:
        text_stream_ids = info_dict.get("text_stream_ids")
        ids = info_dict.get("ids")
        if text_stream_ids is None and isinstance(ids, Mapping):
            text_stream_ids = ids.get("text_stream_ids")
            if text_stream_ids is None:
                text_stream_ids = ids.get("text_stream")
        return text_stream_ids

    @staticmethod
    def _as_1d_long_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim != 1:
            tensor = tensor.reshape(-1)
        return tensor.to(device=device, dtype=torch.long, non_blocking=True)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if input_embeds is None:
            input_embeds = self.language_model.embed_input_ids(input_ids)
        if span_len <= 0:
            return input_ids, input_embeds, {}

        stream_ids = self._as_1d_long_tensor(self._get_text_stream_ids(info_dict), input_ids.device)
        if stream_ids is None or stream_ids.numel() == 0:
            return input_ids, input_embeds, {}

        meta = info_dict.get("meta") if isinstance(info_dict.get("meta"), dict) else {}
        total = int(stream_ids.numel())
        if total == span_len:
            offset = 0
        else:
            offset = int(meta.get("num_processed_tokens", meta.get("aero_text_stream_offset", 0)))
        if offset < 0 or offset >= total:
            return input_ids, input_embeds, {}

        seg_len = min(span_len, total - offset)
        req_stream_ids = stream_ids[offset : offset + seg_len]
        req_input_ids = input_ids[:seg_len]
        changed_mask = req_stream_ids != req_input_ids
        changed_mask = changed_mask & (req_input_ids == self.config.audio_token_id)

        if bool(changed_mask.any().item()):
            req_embeds = input_embeds.clone()
            stream_embeds = self.language_model.embed_input_ids(req_stream_ids[changed_mask])
            structural_embeds = self.language_model.embed_input_ids(req_input_ids[changed_mask])
            req_embeds[:seg_len][changed_mask] = req_embeds[:seg_len][changed_mask] + (
                stream_embeds.to(req_embeds.dtype) - structural_embeds.to(req_embeds.dtype)
            )
        else:
            req_embeds = input_embeds

        next_offset = offset + seg_len
        return input_ids, req_embeds, {"meta": {"num_processed_tokens": next_offset, "aero_text_stream_offset": next_offset}}

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # AutoWeightsLoader can't stack q/k/v -> qkv_proj on its own without a
        # full model-level stacked_params_mapping (which would also affect the
        # language model). Manually fan q/k/v_proj into qkv_proj for the audio
        # tower and let AutoWeightsLoader handle everything else.
        audio_qkv_shards = {
            ".self_attn.q_proj": "q",
            ".self_attn.k_proj": "k",
            ".self_attn.v_proj": "v",
        }
        audio_params = dict(self.audio_tower.named_parameters())
        audio_loaded: set[str] = set()
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            mapped = self.hf_to_vllm_mapper._map_name(name)
            if mapped is None:
                continue
            if not mapped.startswith("audio_tower."):
                other_weights.append((name, w))
                continue
            inner = mapped[len("audio_tower."):]
            handled = False
            for shard_name, shard_id in audio_qkv_shards.items():
                if shard_name not in inner:
                    continue
                fused = inner.replace(shard_name, ".self_attn.qkv_proj")
                if fused not in audio_params:
                    continue
                param = audio_params[fused]
                param.weight_loader(param, w, shard_id)
                audio_loaded.add(f"audio_tower.{fused}")
                handled = True
                break
            if not handled:
                other_weights.append((name, w))

        loader = AutoWeightsLoader(self)
        other_loaded = loader.load_weights(other_weights, mapper=self.hf_to_vllm_mapper)
        return other_loaded | audio_loaded

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model=["visual.", "audio_tower."],
        )


# =============================================================================
# Aero Realtime audio encoder (chunked, no KV cache).
#
# Processor splits audio into independent ``audio_length_per_tok``-mel chunks
# (default 8 mel frames per chunk, yielding 4 post-conv2 frames). Each chunk
# is encoded independently — there is no cross-chunk attention. The encoder
# therefore runs without KV cache and without padding cache.
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class AeroRealtimeAudioRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _build_audio_norm(config) -> nn.Module:
    norm_type = getattr(config, "norm_type", "rms_norm")
    if norm_type == "rms_norm":
        return AeroRealtimeAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    if norm_type == "layer_norm":
        return nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


class AeroRealtimeAudioConv1d(nn.Conv1d):
    """Causal/symmetric padded conv1d. Chunks are independent so we never
    consult the streaming padding cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = "causal",
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=bias,
        )
        if padding_mode not in ("causal", "symmetric"):
            raise ValueError(f"padding_mode must be 'causal' or 'symmetric', got {padding_mode}")
        self.padding_mode_ = padding_mode

    @property
    def left_pad(self) -> int:
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.padding_mode_ == "symmetric":
            pad = self.left_pad // 2
            x = nn.functional.pad(x, (pad, self.left_pad - pad))
        else:
            x = nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class AeroRealtimeAudioRotaryEmbedding(nn.Module):
    """Standard RoPE for the audio encoder. Chunks are short (chunk_enc=4),
    positions are ``arange(chunk_enc)`` per chunk."""

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        base = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class AeroRealtimeAudioAttention(nn.Module):
    """Self-attention without KV cache. Each chunk in the batch is treated
    as an independent sequence (no cross-chunk attention)."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (self.embed_dim // self.num_heads)
        self.scaling = self.head_dim ** -0.5

        tp_size = get_tensor_model_parallel_world_size()
        if self.num_heads % tp_size != 0 or self.num_kv_heads % tp_size != 0:
            raise ValueError(
                "Aero audio attention requires num_heads/num_kv_heads divisible by TP size"
            )
        self.num_local_heads = self.num_heads // tp_size
        self.num_local_kv_heads = self.num_kv_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
        )
        # lmms uses bias=True only for q/v; k has optional bias. vllm
        # QKVParallelLinear ties all three. The ckpt stores k_proj.bias if
        # present; if not, weight loader will skip it (we tolerate by allowing
        # bias=True and zero-init if missing in load_weights mapper).

        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.embed_dim,
            bias=True,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_local_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [B=1, T_flat, n_heads, head_dim] for RoPE in lmms convention
        # (batch, n_heads, seq, head_dim) — here we flatten chunks along seq.
        q = q.view(seq_len, self.num_local_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
        k = k.view(seq_len, self.num_local_kv_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        q = q.transpose(1, 2).reshape(1, seq_len, self.num_local_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(1, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(seq_len, self.num_local_kv_heads, self.head_dim).unsqueeze(0)

        attn_output = self.attn(
            query=q, key=k, value=v,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        attn_output = attn_output.view(seq_len, -1)
        output, _ = self.o_proj(attn_output)
        return output


class AeroRealtimeAudioMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=2 * config.intermediate_size,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = get_act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        out, _ = self.down_proj(x)
        return out


class AeroRealtimeAudioGeluMLP(nn.Module):
    """fc1 -> act -> fc2 MLP."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=True,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            prefix=f"{prefix}.fc2",
        )
        self.act_fn = get_act_fn(config.activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


def _build_audio_mlp(config, prefix: str) -> nn.Module:
    mlp_type = getattr(config, "mlp_type", "swiglu")
    if mlp_type == "swiglu":
        return AeroRealtimeAudioMLP(config, prefix=prefix)
    if mlp_type == "gelu":
        return AeroRealtimeAudioGeluMLP(config, prefix=prefix)
    raise ValueError(f"Unknown mlp_type: {mlp_type!r}")


class AeroRealtimeAudioEncoderLayer(nn.Module):
    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = AeroRealtimeAudioAttention(config, prefix=f"{prefix}.self_attn")
        self.self_attn_layer_norm = _build_audio_norm(config)
        self.final_layer_norm = _build_audio_norm(config)
        self.mlp = _build_audio_mlp(config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, cu_seqlens, max_seqlen
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AeroRealtimeAudioEncoder(nn.Module):
    """Chunked (no KV cache) variant of the Aero realtime audio tower.

    Forward expects ``input_features`` shaped ``[B*N, num_mel_bins, chunk_mel]``
    (one row per LM audio token) and returns ``[B*N, chunk_enc, hidden]``
    (per-chunk post-encoder frames). chunk_enc = chunk_mel // 2 (one conv2
    stride=2).
    """

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        conv_padding = getattr(config, "conv_padding", "causal")
        self.conv1 = AeroRealtimeAudioConv1d(
            config.num_mel_bins, config.hidden_size, kernel_size=3,
            padding_mode=conv_padding,
        )
        self.conv2 = AeroRealtimeAudioConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2,
            padding_mode=conv_padding,
        )

        self.layers = nn.ModuleList([
            AeroRealtimeAudioEncoderLayer(config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.encoder_layers)
        ])
        self.layer_norm = _build_audio_norm(config)
        self.rotary_emb = AeroRealtimeAudioRotaryEmbedding(config)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: [B*N, F, chunk_mel]
        x = nn.functional.gelu(self.conv1(input_features))
        x = nn.functional.gelu(self.conv2(x))     # [B*N, hidden, chunk_enc]
        x = x.permute(0, 2, 1)                    # [B*N, chunk_enc, hidden]

        bn, chunk_enc, hidden_size = x.shape
        # Flatten chunks along sequence: [B*N * chunk_enc, hidden]
        flat = x.reshape(bn * chunk_enc, hidden_size)

        # RoPE positions per chunk: arange(chunk_enc), shared across chunks.
        position_ids = torch.arange(chunk_enc, device=x.device).unsqueeze(0)  # [1, chunk_enc]
        cos_one, sin_one = self.rotary_emb(x, position_ids)  # [1, chunk_enc, head_dim]
        # Repeat for every chunk to match the flat sequence layout.
        cos = cos_one.expand(bn, -1, -1).reshape(bn * chunk_enc, -1).unsqueeze(0)
        sin = sin_one.expand(bn, -1, -1).reshape(bn * chunk_enc, -1).unsqueeze(0)

        # cu_seqlens: chunk boundaries.
        cu_seqlens = torch.arange(
            0, (bn + 1) * chunk_enc, chunk_enc,
            device=x.device, dtype=torch.int32,
        )
        max_seqlen = torch.tensor(chunk_enc, device=x.device, dtype=torch.int32)

        for layer in self.layers:
            flat = layer(flat, (cos, sin), cu_seqlens, max_seqlen)

        flat = self.layer_norm(flat)
        return flat.reshape(bn, chunk_enc, hidden_size)
