# Aero Realtime Omni — Design Spec

**Status:** Draft, awaiting user review
**Date:** 2026-07-05
**Author:** OpenCode
**Reference implementation:** `/data/v-kaichen/lmms-engine/src/lmms_engine/models/aero_realtime_omni/`
**Reference checkpoint:** `/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine`

## 1. Problem

The current `aero_realtime` model in vLLM-Omni is a **single-stage realtime AR thinker** that consumes streaming
audio (+ optional video / text) chunks and emits text tokens. In `lmms-engine` the same thinker has been
paired with a **Qwen3-TTS-style talker** (16-codebook AR + code predictor), producing per-frame codec codes
that Qwen3-TTS's Code2Wav can render into 24 kHz audio. The task is to port that thinker+talker+code2wav
combination into vLLM-Omni as a **new 3-stage streaming pipeline**, without touching the existing
single-stage `aero_realtime` deployment.

## 2. Goals & non-goals

**Goals:**

- Add a new `aero_realtime_omni` model type registered under a distinct architecture name so that
  existing `aero_realtime` deployments continue to work unchanged.
- Ship a 3-stage `PipelineConfig`: `thinker → talker → code2wav` (streaming, async_chunk).
- Load the checkpoint at
  `/data/v-kaichen/azure_blob/output/aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b_1x4_a100_80g_lr1e_4_cosine`
  with correct `hf_to_vllm_mapper` prefixes.
- End-to-end offline example (adapted from the existing `offline_realtime_debug.py`) that streams
  audio+video chunks in and dumps a 24 kHz WAV out.

**Non-goals for this iteration (YAGNI):**

- Voice cloning / speaker encoder — the model has one baked-in speaker (`{"ryan": 3061}`).
- Language ID / dialect / think-mode / instruct — Aero-realtime talker does not implement any of these.
- CUDA-graph acceleration of the talker or code predictor.
- Classifier-free-guidance (CFG).
- Tensor / sequence parallelism.
- OpenAI-compatible online serving (`/v1/audio/speech`, realtime WebSocket).
- Stateful (append-only KV) talker across chunks — see D1 below.

## 3. Background — reference facts

Fact-finding results referenced elsewhere; the design choices below depend on them.

### 3.1 lmms-engine `aero_realtime_omni` talker

Files: `configuration_aero_realtime_talker.py`, `modeling_aero_realtime_talker.py`,
`modeling_aero_realtime_omni.py`.

- **Trunk prompt (training, per sample):**
  ```
  [ codec_emb(codec_bos),
    codec_emb(codec_nothink),
    codec_emb(speaker_id),
    text_projection(H_1)  + codec_emb(codec_bos),          # body_1
    text_projection(H_2)  + codec_emb(group0_1),           # body_2
    ...
    text_projection(H_N)  + codec_emb(group0_{N-1}) ]      # body_N
  ```
  where `H_i` is the thinker's final-layer post-norm hidden state at the i-th `<|audio_pad|>` position.
- **Group 0** is sampled from the trunk's `codec_head`.
- **Groups 1..15** are produced by `AeroRealtimeTalkerCodePredictorModelForConditionalGeneration`:
  nested 15-step AR seeded with `[trunk_hidden ; codec_emb(group0)]`. See
  `modeling_aero_realtime_talker.py:1279-1288`.
- **Special tokens:** `codec_bos_id=2149`, `codec_nothink_id=2155`, `codec_pad_id=2148`,
  `codec_eos_id=2150`, `speaker_id={"ryan":3061}`. No think-mode tokens.
- **Config attribute divergences from Qwen3-TTS:** `codec_eos_id` (not `codec_eos_token_id`),
  `speaker_id` (dict, not `spk_id`), `thinker_hidden_size` new field.
- No chunk / streaming code path exists in the reference (`forward` is HF-`generate` style).

### 3.2 vLLM-Omni streaming infrastructure

- **Stage 0 (append-only):** `OmniARScheduler._update_request_as_session` at
  `vllm_omni/core/sched/omni_ar_scheduler.py:600-615` preserves KV; only the newly appended
  `input_ids` are prefilled per chunk. Aero-realtime thinker already runs in this mode.
- **Stage > 0 (full re-prefill):** `_replace_session_with_streaming_update` clears KV and rebuilds
  the whole prompt. `additional_information` payloads listed in `model.streaming_accumulated_keys`
  are auto-concatenated across chunks (`vllm_omni/worker/gpu_model_runner.py:1488-1543`).
- **Producer hook:** `async_chunk_process_next_stage_input_func` fires per producer step with
  `transfer_manager.put_req_chunk[request_id]` chunk counter (0-indexed).
- **Consumer hook:** `custom_process_input_func` is called on the downstream stage when it assembles
  its `OmniTokensPrompt` from `stage_list[<producer_id>].engine_outputs`.

### 3.3 Qwen3-Omni-MoE vs Qwen3-TTS vLLM talkers

- **`Qwen3TTSTalkerForConditionalGeneration`** (standalone) tokenizes `additional_information["text"]`
  internally. Not usable for Aero — the input is thinker hidden states, not a text string.
- **`Qwen3OmniMoeTalkerForConditionalGeneration`** accepts thinker hidden states via
  `additional_information.hidden_states.output` and projects them with `hidden_projection`. This is
  the correct reference pattern for Aero-realtime talker.

### 3.4 Checkpoint layout

`safetensors` at the reference checkpoint has:
- `thinker.language_model.*`, `thinker.vision_tower.*`, `thinker.audio_tower.*`,
  `thinker.multi_modal_projector.*`, `thinker.lm_head.weight` (matches existing
  `AeroRealtimeForConditionalGeneration.hf_to_vllm_mapper` if the `thinker.` prefix is stripped).
- `talker.model.text_embedding.weight` `[151936, 2048]`, `talker.model.codec_embedding.weight`
  `[3072, 1024]`, `talker.model.norm.weight` `[1024]`, `talker.model.layers.{0..27}.*`,
  `talker.codec_head.weight` `[3072, 1024]`,
  `talker.text_projection.{linear_fc1,linear_fc2}.{weight,bias}`, `talker.code_predictor.*`.
- Talker config: `num_hidden_layers=28`, `hidden_size=1024`, `num_attention_heads=16`,
  `num_key_value_heads=8`, `intermediate_size=3072`, `head_dim=128`, `vocab_size=3072`,
  `num_code_groups=16`, `text_hidden_size=2048`, `text_vocab_size=151936`,
  `thinker_hidden_size=2560`.
- Code predictor config: `num_hidden_layers=5`, `hidden_size=1024`, `vocab_size=2048`,
  `num_code_groups=16`.
- Speech tokenizer weights: **not** shipped in the aero checkpoint. User will manually copy
  `speech_tokenizer/` from
  `/data/v-kaichen/azure_blob/pretrained_models/huggingface/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/`
  into the aero checkpoint directory before deployment. `Qwen3TTSCode2Wav._ensure_speech_tokenizer_loaded`
  will then find it via its existing `cached_file(self.model_path, "speech_tokenizer/config.json")` call.

## 4. Design decisions

### D1: per-chunk full re-prefill for the talker (not stateful AR)

The talker stage discards KV on every streaming update and rebuilds the trunk prompt from scratch
using the accumulated `hidden_states.output` from the thinker. `streaming_accumulated_keys` handles
the concatenation transparently. Cost is O(N²) with respect to session length, but matches training
semantics exactly and avoids modifying the core scheduler.

### D2: code predictor runs inside the talker's `postprocess` hook

The 15-step residual AR is a nested loop within a single trunk decode step. vLLM's scheduler cannot
express nested AR, so we run it as a plain torch loop inside `AeroRealtimeTalkerForConditionalGeneration.postprocess`,
matching `Qwen3OmniMoeTalkerForConditionalGeneration`'s pattern (`talker_postprocess` in
`qwen3_omni.py:643`). Sampled `group0_id` comes from `additional_information` provided by the sampler;
the postprocess writes the resulting `[group0, ..., group15]` frame back into
`additional_information.codes.audio` and appends to the running per-request codec accumulator.

### D3: new model_type `aero_realtime_omni`, existing `aero_realtime` unchanged

- New top-level `AeroRealtimeOmniForConditionalGeneration` handles multi-stage dispatch by
  `model_stage`.
- Existing `AeroRealtimeForConditionalGeneration` (the thinker) remains unchanged except for
  additive changes that let it export hidden states + word embeddings when `model_stage="thinker"`
  is set. Backwards-compatible: when the extra kwargs are not passed the thinker forward returns
  the plain `hidden_states` as before.

### D4: reuse `Qwen3TTSCode2Wav` as-is via architecture alias

Register `AeroRealtimeCode2Wav` as an alias for `Qwen3TTSCode2Wav`. No new class needed. The
speech_tokenizer path is resolved through the existing `cached_file(self.model_path, ...)` call;
user places the tokenizer subfolder alongside the aero weights.

### D5: talker input additional_information contract

For every streaming update, the producer hook `thinker2talker_async_chunk` emits (accumulate keys
in **bold**):

| key path | value | notes |
|---|---|---|
| **`embed.prefill`** | `Tensor [N_new_frames, hidden_size=2560]` | word embed of each new `<|audio_pad|>` slot; auto-concatenated across chunks |
| **`hidden_states.output`** | `Tensor [N_new_frames, hidden_size=2560]` | thinker post-norm hidden state at each new `<|audio_pad|>` slot |
| `meta.finished` | `torch.Tensor(bool)` | whether the stream is done |

At the talker stage, `preprocess` will see the accumulated tensors of shape `[N_total_frames, 2560]`
in `additional_information.hidden_states.output`.

## 5. Component inventory

### New files

| Path | Purpose |
|---|---|
| `vllm_omni/transformers_utils/configs/aero_realtime_omni.py` | `AeroRealtimeOmniConfig`, `AeroRealtimeTalkerConfig`, `AeroRealtimeTalkerCodePredictorConfig` (port of lmms-engine configs) |
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime_omni.py` | Top-level `AeroRealtimeOmniForConditionalGeneration` — stage dispatcher (thinker | talker | code2wav) modeled on `Qwen3OmniMoeForConditionalGeneration` |
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime_talker.py` | `AeroRealtimeTalkerForConditionalGeneration`: trunk (28-layer Qwen3 decoder), `text_projection` MLP, `codec_head`, `code_predictor` reference. Reuses `vllm.model_executor.models.qwen3.Qwen3Model` for the trunk decoder |
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime_code_predictor.py` | `AeroRealtimeTalkerCodePredictor`: 5-layer Qwen3-style dense predictor with 15 per-group embedding tables + 15 per-group `lm_head`s. Reused inside talker.postprocess |
| `vllm_omni/model_executor/stage_input_processors/aero_realtime_omni.py` | `thinker2talker_async_chunk`, `talker2code2wav_async_chunk`, plus sync `custom_process_input_func` fallbacks. Copy-adapt from `qwen3_omni.py` |
| `vllm_omni/deploy/aero_realtime_omni.yaml` | 3-stage deploy config (mirrors `qwen3_omni_moe.yaml` layout; single-GPU + multi-GPU variants) |
| `examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py` | End-to-end demo. Extends `offline_realtime_debug.py` with WAV writer for the `output.multimodal_output["audio"]` stream |

### Modified files

| Path | Change |
|---|---|
| `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py` | Additive: - Add optional `capture_layer_indices` / `return_hidden_states` args to `forward()` (following the qwen3_omni_moe_thinker pattern). Default = current behavior. - Add `make_omni_output` that packages `(hidden, {"hidden_states":{"output":post_norm_last_hidden}, "embed":{"prefill":word_embeds}})` from the audio_pad positions. - Set `has_postprocess=True` and add an empty `streaming_accumulated_keys` (thinker itself doesn't need accumulation; stage-0 append-only handles delta alignment). |
| `vllm_omni/model_executor/models/aero_realtime/pipeline.py` | Additive: add `AERO_REALTIME_OMNI_PIPELINE` with the 3-stage layout. Keep `AERO_REALTIME_PIPELINE`. |
| `vllm_omni/model_executor/models/registry.py` | Register: `AeroRealtimeOmniForConditionalGeneration`, `AeroRealtimeTalkerForConditionalGeneration`, `AeroRealtimeCode2Wav` (alias to `Qwen3TTSCode2Wav`). |
| `vllm_omni/model_executor/models/aero_realtime/__init__.py` | Export new classes. |
| `vllm_omni/transformers_utils/configs/aero_realtime.py` | Keep unchanged. The new omni config imports/reuses `AeroRealtimeConfig` (thinker-side) as a sub-config. |

### Reused as-is

- `Qwen3TTSCode2Wav` — architecture-alias registration only. Provided the checkpoint bundles a
  `speech_tokenizer/` folder (user-supplied via manual `cp`).
- vLLM's `Qwen3Model` (imported from `vllm.model_executor.models.qwen3`) — used as the talker trunk backbone.

## 6. Runtime flow (per streaming chunk)

Setup: user starts `AsyncOmni(model="/data/.../aero_realtime_omni_qwen3vl_4b_qwen3tts_0_6b...",
deploy_config="vllm_omni/deploy/aero_realtime_omni.yaml")`. Stage 0 loads the ~4B thinker on GPU 0;
stage 1 loads the ~0.6B talker on GPU 1; stage 2 loads the code2wav on GPU 1.

For each incoming chunk (80 ms audio + optional video/text):

1. **Stage 0 thinker.** `buffer_realtime_omni` yields an `OmniTokensPrompt` with prompt_token_ids
   ending in `[..., <|audio_pad|>×4]`, plus audio mm data. Scheduler appends the 4 pad tokens to the
   existing KV. Thinker `forward` returns
   `(last_hidden_states, {"hidden_states":{"output":<H at 4 audio_pad positions>},"embed":{"prefill":<word_embeds of the 4 pad ids>}})`.
   Sampler samples 1 text token (`<|rt_pad|>` if user is not speaking yet).

2. **Producer hook `thinker2talker_async_chunk`.** Reads pooling output; emits payload with the 4 new
   frames' hidden_states.output + embed.prefill. Ships to stage 1.

3. **Stage 1 scheduler.** `_replace_session_with_streaming_update` clears the talker KV, replaces
   `prompt_token_ids` with the new full-length placeholder (length = 3 + N_total_frames, where
   N_total_frames is the sum across all chunks so far), and resets `num_computed_tokens = 0`. Runner
   auto-concatenates `hidden_states.output` and `embed.prefill` across chunks.

4. **Stage 1 talker `preprocess()` (prefill branch, span_len > 1).** Constructs the trunk prompt
   from the accumulated hidden states:
   ```python
   cond_emb = codec_embedding(torch.tensor([2149, 2155, 3061]))         # [3, 1024]
   text_h = self.text_projection(accumulated_hidden)                    # [N_total, 1024]
   prev_ids = concat([[codec_bos_id], past_generated_group0[:-1]])      # [N_total]
   body_emb = text_h + codec_embedding(prev_ids)                        # [N_total, 1024]
   prompt_embeds = concat([cond_emb, body_emb], dim=0)                  # [3+N_total, 1024]
   ```
   Returns `(input_ids_placeholder, prompt_embeds, {"codes":{"audio":zeros((3+N_total,16))}})`.
   Note: `past_generated_group0` (length N_total-1 at start of chunk k, then N_total by end) comes
   from `additional_information.codes.audio[:,0]`, so the trunk sees teacher-forced previous group-0.
   At the *last* body slot, prev_group0 is the group-0 that was sampled at the *previous* chunk;
   for the very first chunk it's `codec_bos_id`.

5. **Stage 1 talker `forward()`.** Runs `self.model(inputs_embeds=prompt_embeds)`, gets last hidden
   at index `3+N_total-1`, then `compute_logits → codec_head(last_hidden)` returns group-0 logits.
   Sampler samples `group0_new` and passes it back to the runner.

6. **Stage 1 talker `postprocess(last_hidden, sampled_id=group0_new)`.**
   - Nested loop: `code_predictor.generate(inputs_embeds=[last_hidden; codec_embedding(group0_new)], max_new_tokens=15)`
     returns `[group1, ..., group15]`.
   - Writes `[group0_new, group1, ..., group15]` into `additional_information.codes.audio[N_total-1]`.
   - Saves `last_hidden` under `hidden_states.last` for the next decode step within this chunk.

7. **Steps 4-6 repeat** for each decode step within this chunk (there may be several `<|audio_pad|>`
   slots per streaming chunk — 4 in the current aero timing).

8. **Producer hook `talker2code2wav_async_chunk`.** Reads the newly-emitted codec frames; buffers
   them until `codec_chunk_frames=25` (config in `aero_realtime_omni.yaml`) is reached; flushes 25
   frames + `codec_left_context_frames=25` overlap to stage 2.

9. **Stage 2 code2wav.** Consumes the flat `[16*chunk_size]` code stream, decodes to 24 kHz PCM via
   the Qwen3-TTS speech tokenizer, returns `multimodal_output["audio"]` as the final delta.

The user's `async for output in omni.generate(...):` sees:
- `output.outputs[0].token_ids` = live text token from thinker (subtitle).
- `output.outputs[0].multimodal_output["audio"]` = new 24 kHz audio delta bytes.

## 7. Edge cases and known risks

- **Group-0 accumulator sync:** The talker prefill reads `past_generated_group0` from
  `additional_information.codes.audio[:,0]`, but this buffer is populated by talker.postprocess in
  the previous chunk. Verify that on chunk k's prefill start, the runner has already committed
  chunk (k-1)'s postprocess writes to `additional_information` — otherwise the last body slot's
  prev-group-0 will be stale (should be `group0_{N_total-1}`, might see zeros). Mitigation: talker
  writes past_group0 both to `codes.audio` and to a dedicated `codes.past_group0` field that is in
  `streaming_accumulated_keys`.
- **First chunk's `body_1` prev-code:** should be `codec_bos_id`, not `codec_pad_id=0`. Handled
  explicitly in the `prev_ids` construction (`prev_ids[0] = codec_bos_id`).
- **Speech tokenizer path:** deployment step requires manually copying
  `speech_tokenizer/` into the aero checkpoint. Documented in the README of the offline example.
- **`realtime_max_tokens=1`:** kept from the current aero thinker. The talker is not `SupportsRealtime`
  and does not need this field — it's not scheduled via the vLLM realtime WebSocket path.
- **Concurrent requests:** single-request first. Multi-request talker state isolation is a follow-up.
- **CUDA graph:** disabled in stage 1/2 (enforce_eager=true in yaml) for the first iteration.

## 8. Validation plan

Main validation runs through the offline example (§9). Additionally:

1. **Config loading.** Instantiate `AeroRealtimeOmniConfig.from_pretrained(<ckpt>)` and verify all
   sub-config attributes match `config.json`.
2. **Weight loading.** Instantiate each of the three stages, run `load_weights`, verify:
   - Thinker: all thinker.* keys mapped; no unexpected missing tensors.
   - Talker: `talker.model.layers.*` → `model.layers.*` mapped; `talker.codec_head` →
     `codec_head`; `talker.text_projection.*` → `text_projection.*`;
     `talker.code_predictor.*` → `code_predictor.*`.
   - Code2Wav: speech tokenizer loads from `<ckpt>/speech_tokenizer/`.
3. **Concurrent smoke (optional).** `max_num_seqs=2` with two parallel requests; verify no
   cross-request state leak in the talker's per-request buffers.

## 9. Test harness path

`examples/offline_inference/aero_realtime_omni/offline_realtime_omni_debug.py` will:

- Load a `.mp4` (audio + video),
- Feed chunks into `AeroRealtimeOmniForConditionalGeneration.buffer_realtime_omni`,
- Print sampled text tokens,
- Concatenate all `multimodal_output["audio"]` deltas and dump to
  `output_aero_omni_<timestamp>.wav` at 24 kHz.

## 10. Open questions

None that block writing the plan; enumerated as follow-ups:

- Do we want to eventually persist talker KV across chunks (stateful mode)? Would remove the O(N²)
  cost but requires a scheduler-side change to opt stage-1 into the append-only path
  (`omni_ar_scheduler.py:596`).
- Do we want CUDA-graph capture on the code predictor's 15-step loop? Estimated 3–5× speedup on that
  hot path (Qwen3-TTS already has a `cuda_graph_decoder_wrapper.py` we can adapt).
- OpenAI-compatible realtime WebSocket endpoint (`/v1/realtime`)? Requires wiring into
  `vllm_omni/entrypoints/openai/`. Follow-up.

## 11. Approval checklist

- [ ] D1: per-chunk re-prefill talker
- [ ] D2: code predictor loop in talker.postprocess
- [ ] D3: new `aero_realtime_omni` model_type, existing `aero_realtime` untouched
- [ ] D4: reuse `Qwen3TTSCode2Wav` via architecture alias
- [ ] D5: `additional_information` contract for thinker→talker

If any of the above are wrong, revise this spec before writing the plan.
