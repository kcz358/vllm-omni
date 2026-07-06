"""AeroRealtime Omni configs (thinker + talker + code2wav wrapper).

Ports `lmms_engine.models.aero_realtime_omni.configuration_aero_realtime_omni`
and `configuration_aero_realtime_talker`. The three classes register with
transformers `AutoConfig` under model_types `aero_realtime_omni`,
`aero_realtime_talker`, and `aero_realtime_talker_code_predictor`.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import AutoConfig

from vllm_omni.transformers_utils.configs.aero_realtime import AeroRealtimeConfig


class AeroRealtimeTalkerCodePredictorConfig(PretrainedConfig):
    model_type = "aero_realtime_talker_code_predictor"

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        sliding_window: int | None = None,
        num_code_groups: int = 16,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        layer_types: list | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        else:
            self.rope_parameters = rope_scaling or {"rope_type": "default", "rope_theta": rope_theta}
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.layer_types = layer_types if layer_types is not None else ["full_attention"] * num_hidden_layers
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class AeroRealtimeTalkerConfig(PretrainedConfig):
    model_type = "aero_realtime_talker"
    sub_configs = {"code_predictor_config": AeroRealtimeTalkerCodePredictorConfig}

    def __init__(
        self,
        vocab_size: int = 3072,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        sliding_window: int | None = None,
        num_code_groups: int = 16,
        thinker_hidden_size: int = 2560,
        text_hidden_size: int = 2048,
        text_vocab_size: int = 151936,
        codec_bos_id: int = 2149,
        codec_eos_id: int = 2150,
        codec_pad_id: int = 2148,
        codec_nothink_id: int = 2155,
        speaker_id: dict | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        code_predictor_config=None,
        **kwargs,
    ):
        if code_predictor_config is None:
            code_predictor_config = AeroRealtimeTalkerCodePredictorConfig()
        elif isinstance(code_predictor_config, dict):
            code_predictor_config = AeroRealtimeTalkerCodePredictorConfig(**code_predictor_config)
        self.code_predictor_config = code_predictor_config

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        if rope_parameters is not None:
            self.rope_parameters = rope_parameters
        elif rope_scaling is not None:
            self.rope_parameters = rope_scaling
        else:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "mrope_section": [24, 20, 20],
                "interleaved": True,
            }
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.text_hidden_size = text_hidden_size
        self.text_vocab_size = text_vocab_size
        self.codec_bos_id = codec_bos_id
        self.codec_eos_id = codec_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_nothink_id = codec_nothink_id
        self.speaker_id = speaker_id if speaker_id is not None else {"ryan": 3061}
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class AeroRealtimeOmniConfig(PretrainedConfig):
    model_type = "aero_realtime_omni"
    sub_configs = {
        "thinker_config": AeroRealtimeConfig,
        "talker_config": AeroRealtimeTalkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        codec_loss_weight: float = 1.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_config = AeroRealtimeConfig()
        elif isinstance(thinker_config, dict):
            thinker_config = AeroRealtimeConfig(**thinker_config)
        self.thinker_config = thinker_config

        if talker_config is None:
            talker_config = AeroRealtimeTalkerConfig()
        elif isinstance(talker_config, dict):
            talker_config = AeroRealtimeTalkerConfig(**talker_config)
        self.talker_config = talker_config

        self.codec_loss_weight = codec_loss_weight

        # Convenience: expose thinker text hidden_size at top level (some vllm code paths
        # read config.hidden_size directly).
        self.hidden_size = getattr(getattr(thinker_config, "text_config", None), "hidden_size", None)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, **kwargs):
        return self.thinker_config.text_config


AutoConfig.register(
    AeroRealtimeTalkerCodePredictorConfig.model_type,
    AeroRealtimeTalkerCodePredictorConfig,
    exist_ok=True,
)
AutoConfig.register(
    AeroRealtimeTalkerConfig.model_type,
    AeroRealtimeTalkerConfig,
    exist_ok=True,
)
AutoConfig.register(
    AeroRealtimeOmniConfig.model_type,
    AeroRealtimeOmniConfig,
    exist_ok=True,
)


__all__ = [
    "AeroRealtimeOmniConfig",
    "AeroRealtimeTalkerConfig",
    "AeroRealtimeTalkerCodePredictorConfig",
]
