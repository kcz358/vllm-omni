"""AeroRealtime model + audio-encoder config for vllm-omni.

Mirrors ``lmms_engine.models.aero_realtime.configuration_aero_realtime``
(text + vision backbone fixed to Qwen3-VL for now) and adds the extra
attributes that vllm's ``WhisperCausalEncoder`` reads when we reuse it as
the audio tower (``d_model``, ``encoder_layers``, ``encoder_attention_heads``,
``encoder_ffn_dim``, ``encoder_head_dim``, ``max_source_positions``,
``is_causal``, ``pos_embed``, ``scale_embedding``, ``block_pool_size``).
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


class AeroRealtimeAudioEncoderConfig(PretrainedConfig):
    """Config for the Aero-owned realtime audio encoder.

    Mirrors ``lmms_engine.models.aero_realtime.AeroRealtimeAudioEncoderConfig``
    field-for-field. ``attribute_map`` exposes legacy aliases (``d_model``,
    ``encoder_layers``, ...) for back-compat with code paths that read them.
    Strict causal sliding-window defaults: ``attention_window_right=0`` and
    ``attention_window_left=sliding_window-1``.
    """

    model_type = "aero_realtime_audio_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int = 64,
        num_mel_bins: int = 128,
        max_position_embeddings: int = 1500,
        activation_function: str = "gelu",
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        sliding_window: int | None = 750,
        attention_window_left: int | None = None,
        attention_window_right: int = 0,
        norm_type: str = "rms_norm",
        mlp_type: str = "swiglu",
        conv_padding: str = "causal",
        k_proj_bias: bool = False,
        rope_parameters: dict | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.sliding_window = sliding_window

        if attention_window_left is None:
            attention_window_left = (sliding_window - 1) if sliding_window is not None else -1
        self.attention_window_left = attention_window_left
        self.attention_window_right = attention_window_right

        if norm_type not in ("rms_norm", "layer_norm"):
            raise ValueError(f"norm_type must be 'rms_norm' or 'layer_norm', got {norm_type}")
        if mlp_type not in ("swiglu", "gelu"):
            raise ValueError(f"mlp_type must be 'swiglu' or 'gelu', got {mlp_type}")
        if conv_padding not in ("causal", "symmetric"):
            raise ValueError(f"conv_padding must be 'causal' or 'symmetric', got {conv_padding}")
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.conv_padding = conv_padding
        self.k_proj_bias = bool(k_proj_bias)

        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": 1000000.0}

        super().__init__(**kwargs)


class AeroRealtimeConfig(PretrainedConfig):
    """AeroRealtime top-level config.

    Text + vision backbones are fixed to Qwen3-VL. Audio defaults to the
    Aero realtime audio encoder defined above. ``downsample_factor`` is
    propagated into ``audio_config.block_pool_size`` so the vllm audio
    attention layers can size their per-LM-slot block correctly.
    """

    model_type = "aero_realtime"

    sub_configs = {
        "text_config": AutoConfig,
        "audio_config": AutoConfig,
        "vision_config": AutoConfig,
    }

    def __init__(
        self,
        backbone_family: str = "qwen3_vl",
        text_config=None,
        audio_config=None,
        vision_config=None,
        projector_hidden_act: str = "gelu",
        audio_length_per_tok: int = 8,
        downsample_factor: int = 4,
        audio_token_index: int = 151671,
        audio_start_token_index: int = 151669,
        audio_end_token_index: int = 151670,
        image_token_index: int = 151655,
        video_token_index: int = 151656,
        vision_start_token_index: int = 151652,
        vision_end_token_index: int = 151653,
        rt_start_token_index: int = 151672,
        rt_pad_token_index: int = 151673,
        rt_speak_token_index: int = 151674,
        rt_end_token_index: int = 151675,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        # Accept the field for forward-compat with multi-backbone ckpts,
        # but the text/vision class is currently fixed to Qwen3-VL.
        if backbone_family != "qwen3_vl":
            raise ValueError(
                f"AeroRealtimeConfig currently only supports backbone_family='qwen3_vl', "
                f"got {backbone_family!r}"
            )
        self.backbone_family = backbone_family

        self.projector_hidden_act = projector_hidden_act
        self.audio_length_per_tok = audio_length_per_tok
        self.downsample_factor = downsample_factor

        self.audio_token_index = audio_token_index
        self.audio_start_token_index = audio_start_token_index
        self.audio_end_token_index = audio_end_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.vision_start_token_index = vision_start_token_index
        self.vision_end_token_index = vision_end_token_index
        self.rt_start_token_index = rt_start_token_index
        self.rt_pad_token_index = rt_pad_token_index
        self.rt_speak_token_index = rt_speak_token_index
        self.rt_end_token_index = rt_end_token_index
        # Aliases for shared rope helper / vllm multimodal plumbing.
        self.image_token_id = image_token_index
        self.video_token_id = video_token_index
        self.vision_start_token_id = vision_start_token_index
        self.vision_end_token_id = vision_end_token_index
        self.audio_token_id = audio_token_index
        self.audio_start_token_id = audio_start_token_index
        self.audio_end_token_id = audio_end_token_index

        # --- text (Qwen3-VL fixed) ---
        if isinstance(text_config, dict):
            mt = text_config.get("model_type", "qwen3_vl_text")
            if mt not in ("qwen3_vl_text", "qwen3_vl"):
                raise ValueError(
                    f"text_config.model_type={mt!r} does not match qwen3_vl"
                )
            text_kwargs = dict(text_config)
            text_kwargs.pop("model_type", None)
            text_config = Qwen3VLTextConfig(**text_kwargs)
        elif text_config is None:
            text_config = Qwen3VLTextConfig()
        self.text_config = text_config

        # --- audio (Aero realtime audio encoder) ---
        if isinstance(audio_config, dict):
            audio_config = dict(audio_config)
            audio_config["model_type"] = audio_config.get(
                "model_type", AeroRealtimeAudioEncoderConfig.model_type
            )
            cfg_cls = CONFIG_MAPPING[audio_config["model_type"]]
            audio_config = cfg_cls(**audio_config)
        elif audio_config is None:
            audio_config = AeroRealtimeAudioEncoderConfig()
        self.audio_config = audio_config

        # --- vision (Qwen3-VL fixed) ---
        if isinstance(vision_config, dict):
            mt = vision_config.get("model_type", "qwen3_vl_vision")
            if mt not in ("qwen3_vl_vision", "qwen3_vl"):
                raise ValueError(
                    f"vision_config.model_type={mt!r} does not match qwen3_vl"
                )
            vision_kwargs = dict(vision_config)
            vision_kwargs.pop("model_type", None)
            vision_config = Qwen3VLVisionConfig(**vision_kwargs)
        elif vision_config is None:
            vision_config = Qwen3VLVisionConfig()
        self.vision_config = vision_config

        # Top-level convenience aliases.
        self.hidden_size = self.text_config.hidden_size
        self.audio_hidden_size = getattr(self.audio_config, "hidden_size", None) or getattr(
            self.audio_config, "d_model", None
        )

        if tie_word_embeddings is False and getattr(self.text_config, "tie_word_embeddings", False):
            tie_word_embeddings = True

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


AutoConfig.register(
    AeroRealtimeAudioEncoderConfig.model_type,
    AeroRealtimeAudioEncoderConfig,
    exist_ok=True,
)
AutoConfig.register(
    AeroRealtimeConfig.model_type,
    AeroRealtimeConfig,
    exist_ok=True,
)


__all__ = ["AeroRealtimeAudioEncoderConfig", "AeroRealtimeConfig"]
