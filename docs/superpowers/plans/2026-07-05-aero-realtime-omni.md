# Aero Realtime Omni Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `aero_realtime_omni` 3-stage streaming pipeline (thinker → talker → code2wav) to vLLM-Omni, reusing the existing `aero_realtime` thinker unchanged and reusing `Qwen3TTSCode2Wav` as-is.

**Architecture:** New stage-0 `AeroRealtimeForConditionalGeneration` (additive: hidden-state export). New stage-1 `AeroRealtimeTalkerForConditionalGeneration` (Qwen3-decoder trunk + text_projection + codec_head + Qwen3-TTS's reusable `CodePredictorWrapper`) that consumes accumulated thinker hidden states via `streaming_accumulated_keys`. Stage-2 is `Qwen3TTSCode2Wav` behind an alias. Per-chunk full re-prefill for the talker; nested 15-step code_predictor AR inside `talker.postprocess`. New model_type `aero_realtime_omni`; existing `aero_realtime` unchanged.

**Tech Stack:** Python 3.12, PyTorch, vLLM-Omni engine, `vllm.model_executor.models.qwen3.Qwen3Model` trunk, `vllm_omni.model_executor.models.common.qwen3_code_predictor.CodePredictorWrapper`, `vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSCode2Wav`.

**Reference spec:** `docs/superpowers/specs/2026-07-05-aero-realtime-omni-design.md`

**Reference checkpoint:** `/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine`

---

## File map

New files:

| Path | Responsibility |
|---|---|
| `vllm_omni/transformers_utils/configs/aero_realtime_omni.py` | `AeroRealtimeOmniConfig`, `AeroRealtimeTalkerConfig`, `AeroRealtimeTalkerCodePredictorConfig` |
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime_omni.py` | Top-level `AeroRealtimeOmniForConditionalGeneration` (stage dispatcher) + `buffer_realtime_omni` classmethod |
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime_talker.py` | `AeroRealtimeTalkerForConditionalGeneration` (trunk + text_projection + codec_head + nested code_predictor) |
| `vllm_omni/model_executor/stage_input_processors/aero_realtime_omni.py` | `thinker2talker_async_chunk`, `talker2code2wav_async_chunk`, sync fallbacks |
| `vllm_omni/deploy/aero_realtime_omni.yaml` | 3-stage deploy config |
| `examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py` | E2E offline demo with WAV writer |
| `examples/offline_inference/aero_realtime_omni/README.md` | Deployment note (speech_tokenizer cp step) |

Modified files:

| Path | Change |
|---|---|
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py` | Add optional hidden-state export via `return_hidden_states` kwarg + `make_omni_output` + `has_postprocess=True` when in omni mode |
| `vllm_omni/model_executor/models/aero_realtime/pipeline.py` | Add `AERO_REALTIME_OMNI_PIPELINE` (3 stages); keep `AERO_REALTIME_PIPELINE` |
| `vllm_omni/model_executor/models/aero_realtime/__init__.py` | Export new classes + pipeline |
| `vllm_omni/model_executor/models/registry.py` | Register new architectures; alias `AeroRealtimeCode2Wav` → `Qwen3TTSCode2Wav` |

---

## Task 1: Add the config classes (`AeroRealtimeOmniConfig`, `AeroRealtimeTalkerConfig`, `AeroRealtimeTalkerCodePredictorConfig`)

**Files:**
- Create: `vllm_omni/transformers_utils/configs/aero_realtime_omni.py`
- Verify: `vllm_omni/transformers_utils/configs/aero_realtime.py` (unchanged — imported here)

**Note on rope storage (transformers 5+):** `PretrainedConfig.rope_scaling` is a property whose getter/setter is aliased to `self.rope_parameters` (see `transformers/configuration_utils.py` lines ~482-488 in v5.8.1). We store rope config **only** in `self.rope_parameters`; reads of `self.rope_scaling` transparently return the same dict via the property. Writing to both would be a no-op at best and a silent shared-reference bug at worst.

- [ ] **Step 1: Create the file with three config classes**

Write `/data/v-kaichen/vllm-omni/vllm_omni/transformers_utils/configs/aero_realtime_omni.py`:

```python
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
        # transformers 5.x: `rope_scaling` is a property alias for `rope_parameters`
        # (see transformers/configuration_utils.py:482-488). Writing to both would be
        # a no-op at best. Store only `rope_parameters`; reads of `.rope_scaling`
        # transparently return the same dict via the property.
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
        # See note above: only `rope_parameters` is stored; `rope_scaling` is a
        # transformers 5.x property alias.
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
```

- [ ] **Step 2: Verify import chain works**

Run:
```bash
python -c "
from vllm_omni.transformers_utils.configs.aero_realtime_omni import (
    AeroRealtimeOmniConfig, AeroRealtimeTalkerConfig, AeroRealtimeTalkerCodePredictorConfig,
)
print('cp:', AeroRealtimeTalkerCodePredictorConfig())
print('talker:', AeroRealtimeTalkerConfig())
print('omni:', AeroRealtimeOmniConfig())
"
```

Expected: three lines printed, no exception.

- [ ] **Step 3: Verify checkpoint config loads**

Run:
```bash
python -c "
from vllm_omni.transformers_utils.configs.aero_realtime_omni import AeroRealtimeOmniConfig
cfg = AeroRealtimeOmniConfig.from_pretrained(
    '/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine'
)
print('talker.num_hidden_layers =', cfg.talker_config.num_hidden_layers)
print('talker.codec_bos_id =', cfg.talker_config.codec_bos_id)
print('talker.speaker_id =', cfg.talker_config.speaker_id)
print('talker.code_predictor_config.num_code_groups =', cfg.talker_config.code_predictor_config.num_code_groups)
print('thinker.audio_token_index =', cfg.thinker_config.audio_token_index)
"
```

Expected:
```
talker.num_hidden_layers = 28
talker.codec_bos_id = 2149
talker.speaker_id = {'ryan': 3061}
talker.code_predictor_config.num_code_groups = 16
thinker.audio_token_index = 151671
```

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/transformers_utils/configs/aero_realtime_omni.py
git commit -s -m "feat(aero_realtime_omni): add config classes"
```

---

## Task 2: Add hidden-state export to the thinker (backward-compat)

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py:1106-1119` (forward + compute_logits region)
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py:467-520` (init region — add `_omni_mode` flag)

Goal: allow the thinker to optionally return a `(hidden_states, mm_dict)` tuple when in omni-mode, and expose it via `make_omni_output`. Default behavior when `_omni_mode == False` is unchanged.

- [ ] **Step 1: Read current forward** to confirm the shape of the change.

Run:
```bash
sed -n '1106,1120p' /data/v-kaichen/vllm-omni/vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
```

Expected: sees `def forward(...): ... return self.language_model.model(...)` at line ~1116.

- [ ] **Step 2: Add `_omni_mode` flag + `has_postprocess` in `__init__`**

Locate the block ending at line 519 (`self.has_preprocess = True`) and edit it to also set omni-mode fields:

```python
# In AeroRealtimeForConditionalGeneration.__init__, right after `self.has_preprocess = True`:
self._omni_mode: bool = bool(getattr(vllm_config.model_config, "omni_mode", False))
if self._omni_mode:
    self.have_multimodal_outputs = True
    self.has_postprocess = True
    # For thinker-as-stage-0 in the 3-stage pipeline, additional_information payloads
    # do NOT need cross-chunk accumulation on the producer side; the scheduler emits
    # per-step deltas that the talker (stage 1) will accumulate.
    self.streaming_accumulated_keys: set[tuple[str, str]] = set()
```

Use the `edit` tool with `oldString = "        self.has_preprocess = True"` and `newString = <block above>`.

- [ ] **Step 3: Modify `forward` to return hidden states in omni mode**

The current forward is at lines 1106-1116. Replace it with:

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor | IntermediateTensors | tuple:
    if intermediate_tensors is not None:
        inputs_embeds = None
    hidden = self.language_model.model(
        input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
    )
    if not self._omni_mode:
        return hidden
    if isinstance(hidden, IntermediateTensors):
        return hidden
    # In omni mode, also emit the word embedding at each token position so the
    # downstream talker can key its stream on audio_pad slots without re-embedding.
    if inputs_embeds is not None:
        embed_prefill = inputs_embeds
    elif input_ids is not None:
        embed_prefill = self.language_model.model.embed_input_ids(input_ids)
    else:
        embed_prefill = hidden.new_zeros((0, hidden.shape[-1]))
    captured: dict = {
        "hidden_states": {"output": hidden},
        "embed": {"prefill": embed_prefill},
    }
    return hidden, captured
```

- [ ] **Step 4: Add `make_omni_output`**

Add a module-level import of `OmniOutput` near the other `vllm_omni` imports at the top of the file (immediately after `from vllm_omni.inputs.data import OmniTokensPrompt`):

```python
from vllm_omni.model_executor.models.output_templates import OmniOutput
```

Then immediately after the modified `forward` (before `compute_logits` at what is currently line 1118) insert:

```python
def make_omni_output(self, model_outputs, **kwargs):
    if isinstance(model_outputs, OmniOutput):
        return model_outputs
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        hidden, captured = model_outputs
        return OmniOutput(
            text_hidden_states=hidden.reshape(-1, hidden.shape[-1]),
            multimodal_outputs=captured,
        )
    if not isinstance(model_outputs, torch.Tensor):
        raise TypeError(
            f"AeroRealtime.make_omni_output expected torch.Tensor, OmniOutput, "
            f"or 2-tuple; got {type(model_outputs).__name__}"
        )
    return OmniOutput(
        text_hidden_states=model_outputs.reshape(-1, model_outputs.shape[-1]),
        multimodal_outputs=None,
    )
```

- [ ] **Step 5: Ensure `compute_logits` handles the tuple-shape too**

`compute_logits` (line 1118) currently is:
```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
    return self.language_model.compute_logits(hidden_states)
```

Change to:
```python
def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
    if isinstance(hidden_states, OmniOutput):
        hidden_states = hidden_states.text_hidden_states
    return self.language_model.compute_logits(hidden_states)
```

- [ ] **Step 6: Add `postprocess` no-op stub** (only needed when `_omni_mode=True` since runner checks the attribute)

Add after `make_omni_output`:

```python
def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
    # Thinker stage 0 has no cross-step state to save on the producer side; the
    # talker stage (1) does its own state management from the accumulated payload.
    return {}
```

- [ ] **Step 7: Add `lm_head` prefix rule to `hf_to_vllm_mapper`**

The Qwen3-VL `Qwen3LLMForCausalLM` stores its LM head at `self.language_model.lm_head`,
but the aero checkpoint has `thinker.lm_head.weight` at top level (post-`thinker.`-strip
this becomes `lm_head.weight`). The existing mapper doesn't rewrite this. Add ONE new
entry at the **bottom** of `orig_to_new_prefix` (order matters — `WeightsMapper._map_name`
applies rules in insertion order, so putting it before the `language_model.` rule would
cause a double rewrite):

Locate the mapper (currently ~lines 373-391):

```python
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "language_model.": "language_model.model.",
            "model.vision_tower.": "visual.",
            "vision_tower.": "visual.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.audio_tower.embedder.": "audio_tower.",
            "audio_tower.embedder.": "audio_tower.",
            "model.audio_tower.norm.": "audio_tower.layer_norm.",
            "audio_tower.norm.": "audio_tower.layer_norm.",
            "model.audio_tower.layers.": "audio_tower.layers.",
        },
    )
```

Insert one line before the closing `},`:

```python
            "lm_head.": "language_model.lm_head.",
```

- [ ] **Step 8: Smoke check — thinker still constructible**

Run:
```bash
python -c "
from vllm_omni.model_executor.models.aero_realtime.aero_realtime import AeroRealtimeForConditionalGeneration
print('class ok:', AeroRealtimeForConditionalGeneration.__name__)
print('_omni_mode default:', 'yes' if hasattr(AeroRealtimeForConditionalGeneration, '__init__') else 'no')
"
```

Expected: `class ok: AeroRealtimeForConditionalGeneration` and no import error.

- [ ] **Step 9: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -s -m "feat(aero_realtime): optional hidden-state export in omni mode"
```

---

## Task 3: Create `AeroRealtimeTalkerForConditionalGeneration`

**Files:**
- Create: `vllm_omni/model_executor/models/aero_realtime/aero_realtime_talker.py`

This is the largest task. The talker is a vLLM AR model that:
1. Loads a 28-layer Qwen3 decoder (`vllm.model_executor.models.qwen3.Qwen3Model`).
2. Has `text_projection` MLP (thinker_hidden=2560 → text_hidden=2048 → talker_hidden=1024).
3. Has `codec_head` (Linear talker_hidden=1024 → vocab_size=3072).
4. Has `code_predictor` (reused `Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM` from the qwen3-tts common module).
5. Has a separate `text_embedding` table (`[text_vocab_size=151936, text_hidden_size=2048]`) — loaded from the checkpoint but **unused at inference** for the aero pipeline because we consume thinker hidden states directly. Kept only so `load_weights` doesn't complain about unmapped tensors. (The talker is trained with `text_embedding` as an auxiliary path in the lmms-engine wrapper; at inference we bypass it.)
6. Preprocess builds the trunk prompt `[cond(3) + body(N_total)]` from accumulated `hidden_states.output` (thinker post-norm hidden at each `<|audio_pad|>` position). Body row i = `text_projection(H_i) + codec_embedding(prev_group0_{i-1})`, with `prev_group0_0 = codec_bos_id`.
7. Postprocess runs the 15-step nested code_predictor AR per decode step.

- [ ] **Step 1: Create the file with imports and the class skeleton**

Write `/data/v-kaichen/vllm-omni/vllm_omni/model_executor/models/aero_realtime/aero_realtime_talker.py`:

```python
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
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, WeightsMapper, maybe_prefix
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
    """Two-layer MLP for hidden-dim resize (thinker → talker or similar)."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int, bias: bool = True):
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
        if isinstance(top_config, AeroRealtimeOmniConfig):
            self.config: AeroRealtimeTalkerConfig = top_config.talker_config
        else:
            self.config = top_config
        talker_config: AeroRealtimeTalkerConfig = self.config

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.streaming_accumulated_keys: set[tuple[str, str]] = {
            ("hidden_states", "output"),
            ("codes", "audio"),
            ("codes", "past_group0"),
        }
        # GPU-resident buffer keys (avoid CPU↔GPU round-trips inside the decode loop).
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "last"),
        }

        # Build the trunk from the talker_config, using vllm's Qwen3Model.
        # We wrap it in a Qwen3-compatible vllm_config so Qwen3Model reads the right hidden sizes.
        trunk_vllm_config = vllm_config.with_hf_config(talker_config)
        self.model = Qwen3Model(vllm_config=trunk_vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Codec head: linear projection from talker hidden → codec vocab logits (group 0).
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

        self.text_projection = AeroRealtimeTalkerResizeMLP(
            input_size=talker_config.thinker_hidden_size,
            intermediate_size=talker_config.text_hidden_size,
            output_size=talker_config.hidden_size,
            bias=True,
        )

        # Text embedding: parked here purely so load_weights doesn't drop the tensor.
        # Not consumed at inference for aero-realtime pipeline (we use thinker hidden
        # states directly). If keeping it as a plain nn.Embedding is memory-costly at
        # ~600MB (151936 * 2048 * 2), it can be lazily released after load_weights;
        # for the first cut we keep it in memory.
        self.text_embedding = nn.Embedding(talker_config.text_vocab_size, talker_config.text_hidden_size)

        # Code predictor for groups 1..15. Reuse Qwen3-TTS's CodePredictorWrapper
        # which already handles the nested AR + CUDA graph capture + sampling.
        predictor_compilation = dataclasses.replace(vllm_config.compilation_config)
        predictor_compilation.static_forward_context = {}
        cp_vllm_config = dataclasses.replace(vllm_config, compilation_config=predictor_compilation)
        from vllm.config.vllm import set_current_vllm_config as _set_cfg

        with _set_cfg(cp_vllm_config):
            self.code_predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
                vllm_config=cp_vllm_config,
                config=talker_config.code_predictor_config,
                talker_config=talker_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )
        self._cp_vllm_config = cp_vllm_config

        # Cache tokens we need often as buffers (avoid CPU→GPU per step).
        if not talker_config.speaker_id:
            raise ValueError("talker_config.speaker_id must have at least one entry")
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
        Decode branch (span_len == 1): use the most recent row of accumulated
        ``hidden_states.output`` and the most recent sampled group-0 code from
        ``codes.past_group0`` to build one new body slot.
        """
        # Normalize the additional_information top-level shape.
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

        thinker_hidden_full = hs.get("output")  # accumulated [N_total_frames, thinker_hidden]
        past_group0 = codes.get("past_group0")  # accumulated [N_total_frames], group-0 codes so far

        if not isinstance(thinker_hidden_full, torch.Tensor):
            raise ValueError(
                "AeroRealtimeTalker.preprocess: missing accumulated hidden_states.output "
                "(expected thinker post-norm hidden at every <|audio_pad|> slot)"
            )

        n_total = int(thinker_hidden_full.shape[0])
        if n_total == 0:
            # No audio_pad slots yet — return an empty pass-through.
            return input_ids, self.embed_input_ids(input_ids), {}

        if span_len > 1:
            # Prefill: rebuild the full trunk prompt every chunk.
            thinker_h = thinker_hidden_full.to(device=device, dtype=dtype)  # [N_total, 2560]
            text_h = self.text_projection(thinker_h)  # [N_total, 1024]

            cond_ids = self._cond_ids.to(device=device)
            cond_emb = self.embed_input_ids(cond_ids)  # [3, 1024]

            # prev_group0[i] = codec_bos for i=0; past_group0[i-1] otherwise.
            prev_ids = torch.empty(n_total, dtype=torch.long, device=device)
            prev_ids[0] = talker_cfg.codec_bos_id
            if n_total > 1:
                if isinstance(past_group0, torch.Tensor) and past_group0.numel() >= n_total - 1:
                    prev_ids[1:] = past_group0.to(device=device, dtype=torch.long)[: n_total - 1]
                else:
                    # No prior sampling yet — teacher-force codec_bos for all body slots.
                    prev_ids[1:] = talker_cfg.codec_bos_id
            prev_emb = self.embed_input_ids(prev_ids)  # [N_total, 1024]

            body_emb = text_h + prev_emb  # [N_total, 1024]
            prompt_embeds_full = torch.cat([cond_emb.to(dtype=dtype), body_emb], dim=0)  # [3+N_total, 1024]

            # If span_len < 3+N_total, take the tail slice matching span_len (the
            # scheduler-appended segment). Chunked prefill uses `talker_prefill_offset`
            # to slice into the full prompt across multiple worker steps.
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
                # Pad tail with the last body embedding (defensive; should not usually happen).
                pad_n = span_len - int(take.shape[0])
                pad_row = prompt_embeds_full[-1:].expand(pad_n, -1)
                take = torch.cat([take, pad_row], dim=0)

            input_ids_out = input_ids.clone()
            input_ids_out[:] = int(talker_cfg.codec_pad_id)

            info_update: dict[str, Any] = {
                "meta": {"talker_prefill_offset": offset + span_len},
                "codes": {"audio": torch.zeros((span_len, int(talker_cfg.num_code_groups)), dtype=torch.long, device=device)},
            }
            return input_ids_out, take, info_update

        # Decode branch (span_len == 1): one extra body slot.
        # We rely on the runner to have called postprocess between the previous forward
        # and this one; postprocess stored `hidden_states.last` and appended a new row
        # to `codes.past_group0`. The runner also appended one new `hidden_states.output`
        # row from the thinker delta of the current chunk.
        thinker_h_last = thinker_hidden_full[-1:].to(device=device, dtype=dtype)  # [1, 2560]
        text_h_last = self.text_projection(thinker_h_last)  # [1, 1024]

        if isinstance(past_group0, torch.Tensor) and past_group0.numel() > 0:
            prev_ids = past_group0[-1:].to(device=device, dtype=torch.long)
        else:
            prev_ids = self._codec_bos_id_tensor.to(device=device)
        prev_emb = self.embed_input_ids(prev_ids)  # [1, 1024]

        body_emb = (text_h_last + prev_emb).reshape(1, -1)

        input_ids_out = input_ids.clone()
        input_ids_out[:] = int(talker_cfg.codec_pad_id)

        info_update = {
            "codes": {"audio": torch.zeros((1, int(talker_cfg.num_code_groups)), dtype=torch.long, device=device)},
        }
        return input_ids_out, body_emb, info_update

    def postprocess(self, hidden_states: torch.Tensor, sampled_token_ids: torch.Tensor | None = None, **info_dict: Any) -> dict[str, Any]:
        """Run the nested 15-step code_predictor AR after group-0 is sampled.

        The ``sampled_token_ids is None`` branch is used during profiling / warmup
        when no logits are available; end-to-end streaming always passes the
        sampled group-0 id via the runner's postprocess hook (Task 6 wires this).
        """
        talker_cfg = self.config
        if hidden_states is None or hidden_states.numel() == 0:
            return {}

        last_hidden = hidden_states[-1:, :].detach()  # [1, 1024]
        if sampled_token_ids is None:
            frame = torch.zeros((1, int(talker_cfg.num_code_groups)), dtype=torch.long, device=hidden_states.device)
            return {
                "hidden_states": {"last": last_hidden},
                "codes": {
                    "audio": frame,
                    "past_group0": torch.zeros((1,), dtype=torch.long, device=hidden_states.device),
                },
            }

        # sampled_token_ids: shape [1] on GPU, dtype long. This is group0 for the last frame.
        group0_id = sampled_token_ids.reshape(-1)[-1:].to(dtype=torch.long, device=hidden_states.device)  # [1]

        # Nested 15-step code_predictor AR via CodePredictorWrapper.forward().
        # The wrapper returns [B, num_groups] with layer0 at column 0 and residuals at 1..G-1.
        layer0_embed = self.embed_input_ids(group0_id).reshape(1, 1, -1)  # [1, 1, 1024]
        past_hidden = last_hidden.reshape(1, 1, -1)                       # [1, 1, 1024]
        audio_codes = self.code_predictor(
            layer0_code=group0_id.reshape(1, 1),
            layer0_embed=layer0_embed,
            last_talker_hidden=past_hidden,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
        )  # [1, num_code_groups]
        frame = audio_codes.reshape(1, -1).to(dtype=torch.long)  # [1, 16]

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
        loader that expects prefix `code_predictor.*` after our mapper; we split the
        weight stream so the code_predictor sees its own weights and everything else
        goes through the standard `AutoWeightsLoader`.
        """
        cp_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            mapped = self.hf_to_vllm_mapper._map_name(name)
            if mapped is None:
                continue
            if mapped.startswith("code_predictor."):
                # Strip the "code_predictor." prefix; the wrapper's load_weights expects
                # its inner names (e.g. "model.layers.0..." and "lm_head.0.weight").
                cp_weights.append((mapped[len("code_predictor.") :], w))
            else:
                other_weights.append((name, w))

        loader = AutoWeightsLoader(self, skip_prefixes=["code_predictor."])
        loaded = loader.load_weights(other_weights, mapper=self.hf_to_vllm_mapper)
        cp_loaded = self.code_predictor.load_weights(iter(cp_weights))
        return loaded | {f"code_predictor.{n}" for n in cp_loaded}


__all__ = ["AeroRealtimeTalkerForConditionalGeneration", "AeroRealtimeTalkerResizeMLP"]
```

- [ ] **Step 2: Verify the code_predictor wrapper API**

`CodePredictorWrapper.forward(layer0_code, layer0_embed, last_talker_hidden, do_sample, temperature, top_k, top_p) -> Tensor[B, num_groups]` is the confirmed public entry point (matches `qwen3_tts_talker.py:1693-1701`). The postprocess in Step 1 calls it as a plain callable (`self.code_predictor(...)`).

Sanity check that the wrapper exposes the expected signature:

```bash
python -c "
import inspect
from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper
print(inspect.signature(CodePredictorWrapper.forward))
"
```

Expected: signature contains `layer0_code`, `layer0_embed`, `last_talker_hidden`.

- [ ] **Step 3: Verify import + instantiation (no checkpoint yet)**

Run:
```bash
python -c "
from vllm_omni.model_executor.models.aero_realtime.aero_realtime_talker import (
    AeroRealtimeTalkerForConditionalGeneration,
)
print('class:', AeroRealtimeTalkerForConditionalGeneration.__name__)
print('hf_to_vllm_mapper:', AeroRealtimeTalkerForConditionalGeneration.hf_to_vllm_mapper)
"
```

Expected: prints the class name and mapper; no import error.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime_talker.py
git commit -s -m "feat(aero_realtime_omni): add talker (trunk + text_projection + code_predictor)"
```

---

## Task 4: Create top-level dispatcher `AeroRealtimeOmniForConditionalGeneration`

**Files:**
- Create: `vllm_omni/model_executor/models/aero_realtime/aero_realtime_omni.py`

This class is the entry point registered with vLLM. Depending on `vllm_config.model_config.model_stage` it constructs either the thinker, the talker, or the code2wav submodule. It's a thin dispatcher mirroring `Qwen3OmniMoeForConditionalGeneration.__init__` at `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py:103`.

- [ ] **Step 1: Write the dispatcher class**

Write `/data/v-kaichen/vllm-omni/vllm_omni/model_executor/models/aero_realtime/aero_realtime_omni.py`:

```python
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

        # Register submodules at stage-specific attribute names so PyTorch's
        # state_dict layout matches the checkpoint prefixes (thinker.* / talker.*
        # / code2wav.*). Also alias self.model to the active stage; qwen3_omni.py
        # uses the same pattern.
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
            # a required-but-unused placeholder.
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
        """Route checkpoint weights to the appropriate submodule by top-level prefix.

        - thinker.* → strip the prefix (thinker's hf_to_vllm_mapper uses unprefixed keys)
        - talker.*  → keep the prefix (talker's hf_to_vllm_mapper has "talker.*" rules)
        - code2wav.* → keep the prefix; Qwen3TTSCode2Wav loads separately anyway
        """
        loaded: set[str] = set()
        thinker_weights: list[tuple[str, torch.Tensor]] = []
        talker_weights: list[tuple[str, torch.Tensor]] = []
        code2wav_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if name.startswith("thinker."):
                thinker_weights.append((name[len("thinker."):], w))
            elif name.startswith("talker."):
                talker_weights.append((name, w))
            elif name.startswith("code2wav."):
                code2wav_weights.append((name, w))

        if self.thinker is not None and thinker_weights:
            loaded |= {f"thinker.{n}" for n in self.thinker.load_weights(thinker_weights)}
        if self.talker is not None and talker_weights:
            loaded |= {f"talker.{n}" for n in self.talker.load_weights(talker_weights)}
        if self.code2wav is not None and code2wav_weights:
            loaded |= {f"code2wav.{n}" for n in self.code2wav.load_weights(code2wav_weights)}
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
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "
from vllm_omni.model_executor.models.aero_realtime.aero_realtime_omni import (
    AeroRealtimeOmniForConditionalGeneration,
)
print('class:', AeroRealtimeOmniForConditionalGeneration.__name__)
"
```

Expected: prints class name; no import error.

- [ ] **Step 3: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime_omni.py
git commit -s -m "feat(aero_realtime_omni): top-level stage dispatcher"
```

---

## Task 5: Register architectures in the model registry

**Files:**
- Modify: `vllm_omni/model_executor/models/registry.py:204-208` (add adjacent to existing aero_realtime entry)
- Modify: `vllm_omni/model_executor/models/aero_realtime/__init__.py`

- [ ] **Step 1: Add three new registry entries in `_OMNI_MODELS`**

Locate the current entry at `registry.py:204-208`:

```python
    "AeroRealtimeForConditionalGeneration": (
        "aero_realtime",
        "aero_realtime",
        "AeroRealtimeForConditionalGeneration",
    ),
```

Insert immediately after it (still inside `_OMNI_MODELS`):

```python
    "AeroRealtimeOmniForConditionalGeneration": (
        "aero_realtime",
        "aero_realtime_omni",
        "AeroRealtimeOmniForConditionalGeneration",
    ),
    "AeroRealtimeTalkerForConditionalGeneration": (
        "aero_realtime",
        "aero_realtime_talker",
        "AeroRealtimeTalkerForConditionalGeneration",
    ),
    "AeroRealtimeCode2Wav": (
        "qwen3_tts",
        "qwen3_tts_code2wav",
        "Qwen3TTSCode2Wav",
    ),
```

Use `edit` tool with `oldString` capturing the existing block ending line 208 and `newString` = existing block + new entries.

- [ ] **Step 2: Export the new classes from `aero_realtime/__init__.py`**

Current file:
```python
from .aero_realtime import AeroRealtimeForConditionalGeneration

__all__ = ["AeroRealtimeForConditionalGeneration"]
```

Replace with:
```python
from .aero_realtime import AeroRealtimeForConditionalGeneration
from .aero_realtime_omni import AeroRealtimeOmniForConditionalGeneration
from .aero_realtime_talker import AeroRealtimeTalkerForConditionalGeneration

__all__ = [
    "AeroRealtimeForConditionalGeneration",
    "AeroRealtimeOmniForConditionalGeneration",
    "AeroRealtimeTalkerForConditionalGeneration",
]
```

- [ ] **Step 3: Verify registry lookup works**

Run:
```bash
python -c "
from vllm_omni.model_executor.models.registry import OmniModelRegistry
print('aero omni:', OmniModelRegistry.inspect_model_cls('AeroRealtimeOmniForConditionalGeneration'))
print('aero talker:', OmniModelRegistry.inspect_model_cls('AeroRealtimeTalkerForConditionalGeneration'))
print('aero code2wav:', OmniModelRegistry.inspect_model_cls('AeroRealtimeCode2Wav'))
"
```

Expected: three lines, each printing a ModelInfo tuple; no LookupError.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/models/registry.py \
        vllm_omni/model_executor/models/aero_realtime/__init__.py
git commit -s -m "feat(aero_realtime_omni): register omni + talker + code2wav architectures"
```

---

## Task 6: Add stage input processors (`thinker2talker_async_chunk`, `talker2code2wav_async_chunk`)

**Files:**
- Create: `vllm_omni/model_executor/stage_input_processors/aero_realtime_omni.py`

The producer hooks translate the thinker's pooling output into a talker payload, and the talker's pooling output into a code2wav payload. We adapt `qwen3_omni.py` (thinker→talker) and `qwen3_omni.py:501` (talker→code2wav) but simplify: aero has no tts_bos/tts_eos/tts_pad thinker special tokens, no PD prefill merging, and no speaker/language extraction (single speaker is baked into the talker).

Key design point: on the thinker side, we only forward hidden states + word embeddings at the `<|audio_pad|>` slots (not at every thinker input token). The thinker's per-step output has shape `[num_tokens_in_step, hidden]`; we filter to audio_pad positions using the `text_stream_ids` scaffold that the buffer already builds (each `<|audio_pad|>` in the prompt gets replaced by either the previously-sampled token or `<|rt_pad|>`), which means audio_pad positions are the ones where the raw `input_ids == audio_token_id`.

- [ ] **Step 1: Write the processor file**

Write `/data/v-kaichen/vllm-omni/vllm_omni/model_executor/stage_input_processors/aero_realtime_omni.py`:

```python
"""Stage input processors for aero_realtime_omni: thinker→talker→code2wav."""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


# The thinker config carries audio_token_index=151671 (aero_realtime.py). Since
# the request object doesn't expose the hf_config in current vLLM-Omni, we
# hardcode the aero-realtime <|audio_pad|> id at module scope, matching how
# qwen3_omni.py:32-35 hardcodes im_start / user / assistant token ids.
_AUDIO_PAD_TOKEN_ID = 151671


def _ensure_list(x):
    if hasattr(x, "_x"):
        return list(x._x)
    if not isinstance(x, list):
        return list(x) if x is not None else []
    return list(x)


def _codec_chunk_config(transfer_manager: Any) -> tuple[int, int]:
    """Read (codec_chunk_frames, codec_left_context_frames) from the connector config.

    Matches the pattern in qwen3_omni.py:515-519, fish_speech.py:81-85, cosyvoice3.py.
    """
    extra = {}
    connector = getattr(transfer_manager, "connector", None)
    if connector is not None:
        cfg = getattr(connector, "config", None) or {}
        extra = cfg.get("extra", {}) or {}
    return (
        int(extra.get("codec_chunk_frames", 25)),
        int(extra.get("codec_left_context_frames", 25)),
    )


def _filter_audio_pad_rows(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    step_input_ids: list[int],
    audio_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter [T, H] hidden and embed to only the rows at audio_pad positions in this step."""
    if hidden is None or hidden.numel() == 0 or not step_input_ids:
        return hidden, embed
    if hidden.shape[0] != len(step_input_ids):
        # Defensive: shape mismatch → skip the filter rather than IndexError.
        return hidden, embed
    ids = torch.tensor(step_input_ids, dtype=torch.long, device=hidden.device)
    mask = ids == audio_token_id
    if int(mask.sum().item()) == 0:
        return hidden.new_zeros((0, hidden.shape[-1])), embed.new_zeros((0, embed.shape[-1]))
    return hidden[mask], embed[mask]


# ---- thinker → talker -------------------------------------------------------


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Per-step producer hook: extract new audio_pad hidden states from the thinker.

    `pooling_output` is already the flat multimodal-outputs dict (not wrapped
    under a `multimodal_outputs` key) — matches how qwen3_omni.py:303-305 reads it.
    """
    if not isinstance(pooling_output, dict):
        return None

    hs = (pooling_output.get("hidden_states") or {}).get("output")
    embed = (pooling_output.get("embed") or {}).get("prefill")
    if not isinstance(hs, torch.Tensor) or not isinstance(embed, torch.Tensor):
        return None

    # Derive per-step token ids from the tail of request.all_token_ids
    # (request has no `last_step_input_ids` attribute in current vLLM-Omni).
    all_ids = _ensure_list(request.all_token_ids)
    n = int(hs.shape[0])
    step_input_ids = all_ids[-n:] if n > 0 else []

    hs_filtered, emb_filtered = _filter_audio_pad_rows(hs, embed, step_input_ids, _AUDIO_PAD_TOKEN_ID)
    if hs_filtered.shape[0] == 0 and not is_finished:
        return None

    payload: OmniPayload = {
        "embed": {"prefill": emb_filtered.detach().to("cpu").contiguous()},
        "hidden_states": {"output": hs_filtered.detach().to("cpu").contiguous()},
        "meta": {"finished": torch.tensor(is_finished, dtype=torch.bool)},
    }
    return payload


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Non-async fallback: assemble a single OmniTokensPrompt for the talker from
    the completed thinker stage output. Used when async_chunk=False.

    For a realtime streaming pipeline this path is mostly a safety net; the primary
    path is the async_chunk hook above.
    """
    stage_id = engine_input_source[0]
    thinker_outputs = stage_list[stage_id].engine_outputs
    talker_inputs: list[OmniTokensPrompt] = []
    device = torch.device(current_platform.device_type)

    for i, out in enumerate(thinker_outputs):
        top = out.outputs[0]
        mm = top.multimodal_output or {}
        hs = (mm.get("hidden_states") or {}).get("output")
        embed = (mm.get("embed") or {}).get("prefill")
        all_ids = _ensure_list(out.prompt_token_ids) + _ensure_list(top.cumulative_token_ids)

        if isinstance(hs, torch.Tensor) and isinstance(embed, torch.Tensor) and all_ids:
            n = int(hs.shape[0])
            ids_tail = all_ids[-n:]
            hs, embed = _filter_audio_pad_rows(hs.to(device), embed.to(device), ids_tail, _AUDIO_PAD_TOKEN_ID)

        payload: OmniPayload = {
            "embed": {"prefill": embed.detach().to("cpu") if isinstance(embed, torch.Tensor) else torch.empty(0)},
            "hidden_states": {"output": hs.detach().to("cpu") if isinstance(hs, torch.Tensor) else torch.empty(0)},
        }

        n_frames = int(hs.shape[0]) if isinstance(hs, torch.Tensor) else 0
        placeholder_len = 3 + n_frames
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * placeholder_len,
                additional_information=payload,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# ---- talker → code2wav ------------------------------------------------------


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Per-step producer hook: buffer the talker's new codec frames until we have
    enough for a code2wav chunk (with left-context overlap for smooth boundaries).
    """
    if not isinstance(pooling_output, dict):
        return None
    codes = (pooling_output.get("codes") or {}).get("audio")
    if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
        return None

    request_id = request.external_req_id
    frame_buf: list[torch.Tensor] = transfer_manager.code_prompt_token_ids[request_id]
    # `codes` is [1, 16] per decode step; append to the buffer.
    frame_buf.append(codes.detach().to(device="cpu", dtype=torch.long).reshape(-1))

    finished = bool(is_finished or request.is_finished())
    chunk_frames, left_context = _codec_chunk_config(transfer_manager)

    total = len(frame_buf)
    already_sent = int(transfer_manager.put_req_chunk[request_id]) * chunk_frames
    pending = total - already_sent
    if pending == 0:
        return None
    if pending < chunk_frames and not finished:
        return None

    end = min(already_sent + chunk_frames, total)
    left = max(0, already_sent - left_context)
    window = frame_buf[left:end]

    # Codebook-major flat layout expected by code2wav: [G=16, W] → flatten row-major.
    stacked = torch.stack(window, dim=0)  # [W, 16]
    codebook_major = stacked.transpose(0, 1).reshape(-1).tolist()  # [16*W]

    return {
        "codes": {"audio": codebook_major},
        "meta": {
            "left_context_size": int(already_sent - left),
            "finished": torch.tensor(finished, dtype=torch.bool),
        },
    }


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Non-async fallback: single OmniTokensPrompt with the full flat codec code stream."""
    stage_id = engine_input_source[0]
    talker_outputs = stage_list[stage_id].engine_outputs
    inputs: list[OmniTokensPrompt] = []
    for out in talker_outputs:
        top = out.outputs[0]
        mm = top.multimodal_output or {}
        codes = (mm.get("codes") or {}).get("audio")
        if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
            inputs.append(OmniTokensPrompt(prompt_token_ids=[], additional_information={}))
            continue
        # codes is [T, 16]; codebook-major flat is [16, T] → flatten.
        codebook_major = codes.reshape(-1, 16).transpose(0, 1).reshape(-1).tolist()
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codebook_major,
                additional_information={"meta": {"left_context_size": 0}},
            )
        )
    return inputs


__all__ = [
    "thinker2talker_async_chunk",
    "thinker2talker",
    "talker2code2wav_async_chunk",
    "talker2code2wav",
]
```

- [ ] **Step 2: Verify the module imports**

Run:
```bash
python -c "
from vllm_omni.model_executor.stage_input_processors import aero_realtime_omni as m
print('funcs:', [k for k in dir(m) if not k.startswith('_')])
"
```

Expected: prints `['OmniEngineCoreRequest', 'OmniPayload', ...]` including `talker2code2wav`, `talker2code2wav_async_chunk`, `thinker2talker`, `thinker2talker_async_chunk`.

- [ ] **Step 3: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/stage_input_processors/aero_realtime_omni.py
git commit -s -m "feat(aero_realtime_omni): stage input processors (thinker→talker→code2wav)"
```

---

## Task 7: Define the 3-stage pipeline + deploy config

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/pipeline.py`
- Create: `vllm_omni/deploy/aero_realtime_omni.yaml`

- [ ] **Step 1: Extend `pipeline.py` with `AERO_REALTIME_OMNI_PIPELINE`**

Open `/data/v-kaichen/vllm-omni/vllm_omni/model_executor/models/aero_realtime/pipeline.py`. Below the existing `AERO_REALTIME_PIPELINE` block, add:

```python

_OMNI_PROC = "vllm_omni.model_executor.stage_input_processors.aero_realtime_omni"

AERO_REALTIME_OMNI_PIPELINE = PipelineConfig(
    model_type="aero_realtime_omni",
    model_arch="AeroRealtimeOmniForConditionalGeneration",
    hf_architectures=("AeroRealtimeOmniForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_OMNI_PROC}.thinker2talker_async_chunk",
            custom_process_next_stage_input_func=f"{_OMNI_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_OMNI_PROC}.thinker2talker",
            async_chunk_process_next_stage_input_func=f"{_OMNI_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_OMNI_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],  # codec_eos_id
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",  # code2wav reuses thinker_config as a stub
            engine_output_type="audio",
            model_arch="AeroRealtimeCode2Wav",
            custom_process_input_func=f"{_OMNI_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


__all__ = ["AERO_REALTIME_PIPELINE", "AERO_REALTIME_OMNI_PIPELINE"]
```

Also update the existing `__all__ = ["AERO_REALTIME_PIPELINE"]` line — replace it as shown above.

- [ ] **Step 2: Register the new pipeline in `pipeline_registry.py`**

Pipeline registration is NOT automatic in vllm-omni. `vllm_omni/config/pipeline_registry.py`
holds an explicit `_OMNI_PIPELINES: dict[str, tuple[str, str]]` map keyed by `model_type`;
every existing pipeline (qwen3_omni_moe, aero_realtime, etc.) has an entry there. Add:

```python
"aero_realtime_omni": (
    "vllm_omni.model_executor.models.aero_realtime.pipeline",
    "AERO_REALTIME_OMNI_PIPELINE",
),
```

Insert it alongside the existing `"aero_realtime"` entry so the two variants sit together.

- [ ] **Step 3: Create the deploy yaml**

Write `/data/v-kaichen/vllm-omni/vllm_omni/deploy/aero_realtime_omni.yaml`. This default
layout splits the 3 stages across 2 GPUs (stage 0 on cuda:0, stages 1+2 on cuda:1); on
single-GPU hosts, override `devices:` and `gpu_memory_utilization:` per stage via a
per-machine yaml or CLI. `max_model_len` is intentionally NOT pinned here — the model's
config-declared context (262144 for Qwen3-VL) applies unless the caller overrides it at
`AsyncOmni(max_model_len=...)`.

```yaml
# aero_realtime_omni: 3-stage streaming pipeline
# Stage 0 = thinker (Qwen3-VL 4B), stages 1/2 = talker (0.6B) + code2wav (~few MB)
#
# Default layout (multi-GPU): stage 0 on cuda:0, stage 1 on cuda:1, stage 2 on cuda:1.
# For single-GPU deployments override devices + gpu_memory_utilization via CLI or a
# per-machine yaml.
async_chunk: true
dtype: bfloat16

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_chunk_frames: 25
      codec_left_context_frames: 25

stages:
  - stage_id: 0
    max_num_seqs: 1
    gpu_memory_utilization: 0.8
    enforce_eager: true
    mm_processor_cache_gb: 0
    devices: "0"
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 1
      seed: 42

  - stage_id: 1
    max_num_seqs: 1
    gpu_memory_utilization: 0.6
    enforce_eager: true
    devices: "1"
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 4096
      seed: 42

  - stage_id: 2
    max_num_seqs: 1
    gpu_memory_utilization: 0.15
    enforce_eager: true
    async_scheduling: false
    max_num_batched_tokens: 51200
    devices: "1"
    input_connectors:
      from_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 65536
      seed: 42
```

- [ ] **Step 4: Verify pipeline resolves**

Run:
```bash
python -c "
from vllm_omni.model_executor.models.aero_realtime.pipeline import AERO_REALTIME_OMNI_PIPELINE
print('model_type:', AERO_REALTIME_OMNI_PIPELINE.model_type)
print('stages:', [(s.stage_id, s.model_stage) for s in AERO_REALTIME_OMNI_PIPELINE.stages])
"
```

Expected:
```
model_type: aero_realtime_omni
stages: [(0, 'thinker'), (1, 'talker'), (2, 'code2wav')]
```

- [ ] **Step 5: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add vllm_omni/model_executor/models/aero_realtime/pipeline.py \
        vllm_omni/config/pipeline_registry.py \
        vllm_omni/deploy/aero_realtime_omni.yaml
git commit -s -m "feat(aero_realtime_omni): 3-stage pipeline + deploy config"
```

---

## Task 8: Weight loading smoke test (per-stage)

**Files:** none created; verification only.

- [ ] **Step 1: Copy the speech_tokenizer folder into the aero checkpoint (one-time setup)**

Run:
```bash
cp -r /data/v-kaichen/azure_blob/pretrained_models/huggingface/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
      /data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine/
```

Verify:
```bash
ls /data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine/speech_tokenizer/
```

Expected: `config.json`, `configuration.json`, `model.safetensors`, `preprocessor_config.json`.

- [ ] **Step 2: Instantiate the omni model config via HF `AutoConfig` end-to-end**

```bash
python -c "
from transformers import AutoConfig
import vllm_omni.transformers_utils.configs.aero_realtime_omni  # trigger AutoConfig.register
cfg = AutoConfig.from_pretrained(
    '/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine'
)
print('model_type:', cfg.model_type)
print('talker.num_hidden_layers:', cfg.talker_config.num_hidden_layers)
"
```

Expected:
```
model_type: aero_realtime_omni
talker.num_hidden_layers: 28
```

- [ ] **Step 3: Try instantiating `AsyncOmni` with the deploy config (real weight load)**

This will attempt to load ALL 3 stages with real weights, so it exercises `load_weights`
for thinker, talker, and code2wav.

**Important**: write the smoke test to a real file (not a `python - <<'PY'` heredoc).
vllm-omni's stage init unconditionally sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and
the spawn child re-imports the caller's file. A heredoc'd script becomes `<stdin>` on
disk and the child fails with `FileNotFoundError`. Use the pattern below:

```bash
cat > /tmp/smoke_task8.py <<'PY'
import asyncio
import warnings
warnings.filterwarnings("ignore")

from vllm_omni.entrypoints.async_omni import AsyncOmni

async def main():
    try:
        omni = AsyncOmni(
            model="/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine",
            deploy_config="vllm_omni/deploy/aero_realtime_omni.yaml",
            log_stats=False,
            gpu_memory_utilization=0.9,
            skip_mm_profiling=True,
        )
        print("OK — all 3 stages loaded")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())
PY

cd /data/v-kaichen/vllm-omni
python /tmp/smoke_task8.py
```

Expected: `OK — all 3 stages loaded` and no exception.

Notes:
- Requires ≥2 GPUs by default (see Task 7 deploy yaml). For single-GPU dev, override
  `devices` and `gpu_memory_utilization` in a per-machine yaml.
- `AsyncOmni` does not accept an `only_stage` kwarg; there's no built-in way to load
  only a subset of stages. If a stage fails, isolate the traceback by inspecting logs.

Debug tips if this fails:
- Missing tensors: check the corresponding stage's `hf_to_vllm_mapper` prefixes against
  the actual checkpoint keys via `python -c "from safetensors import safe_open; ..."`
- Unexpected tensor shape: check that the stage sub-config values (num_hidden_layers,
  hidden_size, etc.) match `config.json`.
- KV cache OOM: reduce `max_model_len` at the `AsyncOmni(...)` call site (do NOT pin
  it in the yaml — that's host-specific tuning).

- [ ] **Step 4: (No commit) — this is a validation-only task**

---

## Task 9: End-to-end offline example with WAV output

**Files:**
- Create: `examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py`
- Create: `examples/offline_inference/aero_realtime_omni/README.md`

Adapt the existing `examples/offline_inference/aero_realtime/offline_realtime_debug.py` (313 lines) with two changes:
1. Use `deploy_config="vllm_omni/deploy/aero_realtime_omni.yaml"` (3-stage) instead of the single-stage default.
2. In the async loop, in addition to printing text tokens, consume `output.outputs[0].multimodal_output["audio"]` bytes, concatenate, and write to a WAV file at 24 kHz.

- [ ] **Step 1: Copy the existing offline script**

```bash
mkdir -p /data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime_omni
cp /data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime/offline_realtime_debug.py \
   /data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py
```

- [ ] **Step 2: Edit the copy — change deploy_config default and add WAV writer**

Open `/data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py` and apply the following surgical edits (use `edit` tool with unique context):

**Edit A:** change the module docstring (line 1) to reflect omni output.

`oldString`:
```python
"""Offline streaming demo for AeroRealtime.

Feeds 80 ms audio chunks (and interleaved video frames) into the model in
real-time order and prints decoded tokens as they are generated.
```

`newString`:
```python
"""Offline streaming demo for AeroRealtime Omni (3-stage: thinker + talker + code2wav).

Feeds 80 ms audio chunks (and interleaved video frames) into the model in
real-time order, prints decoded text tokens, and writes the generated
audio delta stream to a 24 kHz WAV file.
```

**Edit B:** change the default deploy_config path.

`oldString`:
```python
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/aero_realtime.yaml")
```

`newString`:
```python
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/aero_realtime_omni.yaml")
    parser.add_argument("--output-wav", default=None,
                        help="If set, write concatenated 24 kHz audio to this WAV path.")
```

**Edit C:** in `main()` (the async body), after the `async for output in omni.generate(...)` loop begins, add audio-delta collection. Locate the current loop body (around line 300-307):

`oldString`:
```python
            if not output.outputs:
                continue
            token_ids = list(output.outputs[0].token_ids)
            if not token_ids:
                continue
            input_stream.put_nowait(token_ids)
            decoded = [tokenizer.decode([t]) for t in token_ids]
            print(f"[token] ids={token_ids} text={decoded!r}")
```

`newString`:
```python
            if not output.outputs:
                continue
            out0 = output.outputs[0]
            token_ids = list(out0.token_ids)
            if token_ids:
                input_stream.put_nowait(token_ids)
                decoded = [tokenizer.decode([t]) for t in token_ids]
                print(f"[token] ids={token_ids} text={decoded!r}")
            # Collect audio deltas produced by stage-2 code2wav.
            mm = getattr(out0, "multimodal_output", None) or {}
            audio_delta = None
            if isinstance(mm, dict):
                if "audio" in mm:
                    audio_delta = mm["audio"]
                elif "model_outputs" in mm:
                    audio_delta = mm["model_outputs"]
            if audio_delta is not None:
                if isinstance(audio_delta, list):
                    audio_delta = torch.cat([t.reshape(-1) for t in audio_delta], dim=0)
                audio_np = audio_delta.detach().cpu().to(torch.float32).numpy() if hasattr(audio_delta, "detach") else np.asarray(audio_delta, dtype=np.float32)
                _audio_chunks.append(audio_np.reshape(-1))
                print(f"[audio] delta samples={audio_np.reshape(-1).shape[0]}")
```

**Edit D:** immediately before the `try:` block in `main()`, initialize `_audio_chunks`:

`oldString`:
```python
    request_id = f"aero-rt-debug-{uuid.uuid4()}"
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

    try:
```

`newString`:
```python
    request_id = f"aero-rt-omni-debug-{uuid.uuid4()}"
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
    _audio_chunks: list[np.ndarray] = []

    try:
```

**Edit E:** after the `finally: await omni.abort(request_id)` line, add WAV writing.

`oldString`:
```python
    try:
        async for output in omni.generate(
            build_streaming_inputs(
                omni, sample, input_stream,
                audio_chunk_ms=args.audio_chunk_ms,
                inter_chunk_delay_s=args.inter_chunk_delay_s,
                ask_second=args.ask_second,
                ask_text=args.ask_text,
                verbose=args.verbose,
            ),
            request_id=request_id,
            sampling_params_list=[sampling_params],
        ):
```

Keep the try body but after the `finally: await omni.abort(request_id)` add:

Locate:
```python
    finally:
        await omni.abort(request_id)
```

`newString`:
```python
    finally:
        await omni.abort(request_id)

        if _audio_chunks:
            import soundfile as sf
            audio_full = np.concatenate(_audio_chunks, axis=0).astype(np.float32)
            out_wav = args.output_wav or f"output_aero_omni_{int(__import__('time').time())}.wav"
            sf.write(out_wav, audio_full, 24000)
            print(f"[audio] wrote {audio_full.shape[0]} samples ({audio_full.shape[0] / 24000.0:.2f}s) → {out_wav}")
```

**Edit F:** add `torch` import at the top (already imported implicitly via `vllm.sampling_params` but let's be explicit). Locate the imports block near line 20-30:

`oldString`:
```python
import librosa
import numpy as np
```

`newString`:
```python
import librosa
import numpy as np
import torch
```

- [ ] **Step 3: Write the README**

Write `/data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime_omni/README.md`:

```markdown
# Aero Realtime Omni — offline demo

3-stage streaming pipeline: **thinker** (Qwen3-VL 4B multimodal) → **talker**
(Qwen3-TTS-style AR codec generator) → **code2wav** (24 kHz waveform decoder).

## One-time setup

The trained aero_realtime_omni checkpoint does not ship the speech tokenizer
(the codec decoder). Copy it from the Qwen3-TTS-Base repo:

```bash
cp -r /path/to/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
      /path/to/aero_realtime_omni_ckpt/
```

## Run

```bash
export AERO_REALTIME_OMNI_MODEL=/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine

python offline_realtime_omni_debug.py \
    --model "$AERO_REALTIME_OMNI_MODEL" \
    --video-path /path/to/clip.mp4 \
    --ask-second 4 --ask-text "What is happening now? " \
    --output-wav aero_omni_out.wav
```

Output:
- `[token] ids=... text=[...]` — thinker's streamed text tokens (subtitle).
- `[audio] delta samples=N` — code2wav's PCM delta for each chunk.
- `[audio] wrote M samples (X.XXs) → aero_omni_out.wav` — final WAV file at 24 kHz.
```

- [ ] **Step 4: Environment variable rename inside the script**

In the copy at `examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py`, change:

`oldString`:
```python
MODEL_ENV = "AERO_REALTIME_MODEL"
MODEL_PLACEHOLDER = "<aero-realtime-checkpoint>"
```

`newString`:
```python
MODEL_ENV = "AERO_REALTIME_OMNI_MODEL"
MODEL_PLACEHOLDER = "<aero-realtime-omni-checkpoint>"
```

- [ ] **Step 5: Smoke test — script prints help without crashing**

```bash
python /data/v-kaichen/vllm-omni/examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py --help
```

Expected: argparse help text ending with `--output-wav` entry; no `ImportError`.

- [ ] **Step 6: End-to-end run on a real video**

```bash
export AERO_REALTIME_OMNI_MODEL=/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine

python examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py \
    --video-path <PATH-TO-A-SHORT-MP4> \
    --ask-second 2 --ask-text "Describe this scene. " \
    --output-wav /tmp/aero_omni_test.wav
```

Expected:
- Stream of `[token]` lines while the video plays through.
- One or more `[audio] delta samples=...` lines once the model starts speaking.
- Final line: `[audio] wrote N samples (X.XXs) → /tmp/aero_omni_test.wav`.
- Play `/tmp/aero_omni_test.wav` and verify it contains intelligible speech.

If audio is silent or garbled, the most likely culprits (in order):
1. `text_projection` weight loading mismatch (verify `linear_fc1.weight` shape is `[2048, 2560]`).
2. `codec_head` weight mismatch (verify `weight` shape is `[3072, 1024]`).
3. Wrong `codec_bos_id` / `codec_nothink_id` / `speaker_id=3061` values in `AeroRealtimeTalkerConfig`.
4. Group-0 sampling stop condition (verify `stop_token_ids=[2150]` = codec_eos_id in pipeline).
5. `code_predictor(...)` (CodePredictorWrapper.forward) returning wrong-shaped codes.

- [ ] **Step 7: Commit**

```bash
cd /data/v-kaichen/vllm-omni
git add examples/offline_inference/aero_realtime_omni/
git commit -s -m "feat(aero_realtime_omni): offline end-to-end example with WAV output"
```

---

## Self-review — spec coverage and consistency

| Spec item | Covered by task(s) |
|---|---|
| §2 goals: 3-stage pipeline | Task 7 |
| §2 goals: load reference checkpoint | Task 8 |
| §2 goals: e2e offline example | Task 9 |
| §2 non-goals: no voice clone / language_id | Talker preprocess (Task 3) has no such branches |
| §4 D1 per-chunk re-prefill | Task 3 preprocess span_len>1 branch |
| §4 D2 code predictor in postprocess | Task 3 postprocess |
| §4 D3 new model_type `aero_realtime_omni` | Task 1 (config), Task 5 (registry), Task 7 (pipeline) |
| §4 D4 reuse Qwen3TTSCode2Wav | Task 5 alias, Task 7 stage 2 |
| §4 D5 additional_information contract | Task 6 processors, Task 3 preprocess read paths |
| §7 group-0 accumulator sync (past_group0) | Task 3 postprocess writes past_group0; Task 3 preprocess reads it; both in `streaming_accumulated_keys` |
| §8 validation | Task 8 (config+weight loading), Task 9 (e2e), Task 3 (concurrent smoke — optional) |
| §9 test harness | Task 9 |

## Post-task fallbacks

If Task 3 Step 2 discovers the `CodePredictorWrapper` API doesn't expose the exact method used above, the recovery is documented inline in Task 3 Step 2. Read `common/qwen3_code_predictor.py` end-to-end, identify the residual-AR entry point, and either call it directly or write a small wrapper inside the talker.

If Task 8 Step 3 fails with a missing/extra weight tensor, run this diagnostic:

```bash
python - <<'PY'
from safetensors import safe_open
from vllm_omni.transformers_utils.configs.aero_realtime_omni import AeroRealtimeOmniConfig

ckpt = "/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine"
with safe_open(f"{ckpt}/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
prefixes = sorted(set(".".join(k.split(".")[:3]) for k in keys))
for p in prefixes:
    print(p)
PY
```

Compare to the prefixes in each stage's `hf_to_vllm_mapper` and adjust one at a time.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-aero-realtime-omni.md`.

Recommended next step: use **subagent-driven-development** — dispatch one fresh subagent per task, review between tasks. This isolates each task's context and keeps commits atomic. Alternative: **executing-plans** for inline batched execution with checkpoints.
