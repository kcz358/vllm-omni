# Aero Realtime — True-Delta Streaming Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make aero-realtime runtime emit one new `audio_pad` per chunk with the correct `text_stream_ids` slot rule, so that `_all_token_ids` / `input_ids` / `text_stream_ids` / KV-cache state evolve as specified in `docs/superpowers/specs/2026-05-16-aero-realtime-true-delta.md`.

**Architecture:** Spec invariants I1–I5 are the contract. Three layers cooperate: (a) runtime `_build_realtime_delta` / `buffer_realtime_omni` / `build_audio_realtime_text_stream_ids` emit per-chunk delta tokens + side-channel; (b) scheduler stage-0 append (already in place); (c) model `preprocess` substitutes `text_stream_ids[i]` over `audio_pad` slots. Only the runtime layer is being changed in this plan — scheduler and worker stage-0 fixes from `e66bb311` stay as-is and will be verified end-to-end.

**Tech Stack:** Python 3.12, `vllm==0.19.1`, `vllm-omni==0.19.0` (editable), `transformers==5.6.2`, `torch==2.10.0+cu130`. Repo: `/data/v-kaichen/vllm-omni`. Branch: `integrate-aero-realtime`. Validation script: `./workspace/run_aero_realtime_tp2.sh`.

**Validation philosophy:** No unit tests. Only two acceptance criteria:
1. **Chunk emit logic** — runtime emits each chunk according to spec §2 (chunk 0 envelope + 1 ap; audio-only continuation = 1 ap; video reopen = envelope + 1 ap). Verified by grepping the instrumentation log added in Task 1.
2. **Decode response + KV reuse** — demo produces a grammatical response (no `II / can can / the the` duplicates) AND `num_computed_tokens` advances monotonically across chunks (no full re-prefill).

**Token ids (41k ckpt):**
- `audio_pad = 151671`, `audio_start = 151669`, `audio_end = 151670`
- `rt_pad = 151673`, `rt_speak = 151674`, `rt_start = 151672`, `rt_end = 151675`
- `vision_start = 151652`, `vision_end = 151653`, `video_pad = 151656`
- `im_start = 151644`, `im_end = 151645`

---

## File map

- **Modify** `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py`
  - `AeroRealtimeStreamState` (line 299): drop unused fields (`rt_start_emitted`, `rt_speak_emitted`, `audio_segment_first_pending`); keep `chat_started`, `audio_segment_open`, `last_generated_token_id`, `current_time_seconds`.
  - `_build_realtime_delta` (line 573): emit exactly **one** `audio_pad` per audio-bearing chunk regardless of `num_audio_tokens`; do NOT call `_expand_first_token` for audio anymore. Video tokens still expand.
  - `build_audio_realtime_text_stream_ids` (line 492): slot rule = `last_generated_token_id` if available (consume it), else `rt_pad`. Drop `rt_speak` segment-start branch.
  - `_drain_generated_tokens` (line 529): skip any token whose id is in the Aero placeholder set; only real text tokens update `last_generated_token_id`.
  - `buffer_realtime_omni` (line 732) and `buffer_realtime_audio` (line 683): unchanged in structure, but call sites now produce 1-audio_pad chunks. Audit `num_video_tokens` decision so audio-only continuation chunks skip the video block (set `num_video_tokens=0` unless the chunk really carries a new frame).
- **Verify only** (no edits expected) `vllm_omni/core/sched/omni_ar_scheduler.py`, `vllm_omni/worker/gpu_model_runner.py`, `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::preprocess`.
- **Driver** `workspace/run_aero_realtime_tp2.sh`

---

## Task 1: Capture broken baseline with instrumentation

**Why:** Lock the current state with a log file so we can show before/after. The instrumentation stays on through Tasks 2–5 and is only removed in Task 6.

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py`

- [ ] **Step 1: Add runtime emission log**

In `buffer_realtime_omni`, immediately before `yield OmniTokensPrompt(...)`, insert:

```python
logger.warning(
    "[aero_dbg_emit] chunk_idx=%d prompt_len=%d stream_len=%d "
    "audio_pads=%d new_video=%s prompt_head=%s stream_head=%s "
    "prompt_tail=%s stream_tail=%s",
    getattr(state, "_chunk_idx", 0),
    len(prompt_ids),
    len(chunk_text_stream_ids),
    sum(1 for t in prompt_ids if t == audio_token_id),
    num_video_tokens > 0,
    prompt_ids[:8],
    list(chunk_text_stream_ids[:8]),
    prompt_ids[-8:],
    list(chunk_text_stream_ids[-8:]),
)
state._chunk_idx = getattr(state, "_chunk_idx", 0) + 1
```

- [ ] **Step 2: Add worker preprocess log**

In `preprocess`, immediately after `changed_mask = changed_mask & (req_input_ids == self.config.audio_token_id)`, insert:

```python
if getattr(self, "_aero_dump", 0) < 30:
    n_audio = int((req_input_ids == self.config.audio_token_id).sum().item())
    logger.warning(
        "[aero_dbg_model] span=%d total=%d offset=%d seg=%d "
        "audio_pads=%d changed=%d input_head=%s stream_head=%s "
        "input_tail=%s stream_tail=%s",
        span_len, total, offset, seg_len, n_audio,
        int(changed_mask.sum().item()),
        input_ids[:8].tolist(),
        stream_ids[offset:offset+8].tolist(),
        input_ids[max(0, seg_len-8):seg_len].tolist(),
        stream_ids[max(0, offset+seg_len-8):offset+seg_len].tolist(),
    )
    self._aero_dump = getattr(self, "_aero_dump", 0) + 1
```

- [ ] **Step 3: Run baseline**

```bash
cd /data/v-kaichen/vllm-omni
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
# Pick an idx with ≥30 GiB free:
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
```

- [ ] **Step 4: Verify broken baseline**

```bash
grep "aero_dbg_emit\|aero_dbg_model\|\[response\]" workspace/logs/aero_realtime_tp2_*.log | tail -40
```

Expected (BEFORE fix):
- `aero_dbg_emit` shows `prompt_len ≈ 95` and `audio_pads ≈ 80+` on every chunk.
- `aero_dbg_model` shows `span > total` mismatch on prefill steps.
- `[response]` ends with `... just just ... the the ...` style duplicates.

- [ ] **Step 5: Commit baseline**

```bash
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "wip(aero-realtime): add boundary instrumentation; capture broken baseline"
```

---

## Task 2: `_build_realtime_delta` emits one `audio_pad` per audio-bearing chunk

**Why:** Spec I1+I5. Each runtime chunk is one new audio token slot. Current code expands `<|audio_pad|>` to `num_audio_tokens` copies — wrong.

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::_build_realtime_delta` (line 573)

- [ ] **Step 1: Implement**

In `_build_realtime_delta` (lines ~599–642), find the final expansion block:

```python
prompt_ids = tokenizer.encode("".join(parts))
if num_video_tokens > 0:
    prompt_ids = cls._expand_first_token(prompt_ids, video_token_id, num_video_tokens)
if num_audio_tokens > 0:
    prompt_ids = cls._expand_first_token(prompt_ids, audio_token_id, num_audio_tokens)
return prompt_ids
```

Delete the audio expansion. New block:

```python
prompt_ids = tokenizer.encode("".join(parts))
if num_video_tokens > 0:
    prompt_ids = cls._expand_first_token(prompt_ids, video_token_id, num_video_tokens)
# Spec I1+I5: each chunk carries exactly one audio_pad slot.
# We intentionally do NOT expand audio_pad to num_audio_tokens copies.
return prompt_ids
```

- [ ] **Step 2: Run demo to verify chunk emit pattern**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
grep "aero_dbg_emit" workspace/logs/aero_realtime_tp2_*.log | head -10
```

Expected (after fix):
- Chunk 0: `audio_pads=1`, `prompt_len = (envelope structural tokens) + (video_pad×S) + 1`.
- Chunks 1..N: depend on Task 3 — for now they likely still show `new_video=True audio_pads=1` (since demo may still pass video every chunk).
- Key invariant for THIS task: every row has `audio_pads=1` (no more 80+).

If any chunk still shows `audio_pads > 1` after this change, that's a bug. Investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "fix(aero-realtime): _build_realtime_delta emits one audio_pad per chunk"
```

---

## Task 3: `buffer_realtime_omni` sets `num_video_tokens=0` on audio-only continuation chunks

**Why:** Spec I4. Even after Task 2, the caller may pass `num_video_tokens > 0` for chunks that don't carry a real new video frame, causing the envelope structural tokens (vs/video_pad/ve, `<t s>`) to re-emit and inflate `prompt_len`.

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::buffer_realtime_omni` (line 732)
- Possibly modify: `examples/offline_inference/aero_realtime/offline_realtime_debug.py`

- [ ] **Step 1: Audit caller**

```bash
grep -n "_chunk_video\|video_chunk\|num_video_tokens" \
  vllm_omni/model_executor/models/aero_realtime/aero_realtime.py | head -10
grep -n "video=\|video_frames\|frame_idx\|chunk_stream\|AeroRealtimeChunk" \
  examples/offline_inference/aero_realtime/offline_realtime_debug.py | head -20
```

Locate where the demo feeds video into chunks. If the demo passes a video frame on every audio chunk, that's where the fix goes. If the demo only passes video on real new-frame chunks but `buffer_realtime_omni` mis-decides, fix in `buffer_realtime_omni`.

- [ ] **Step 2: Apply minimal fix**

Show the exact diff of where you gate video emission. Most likely in the demo: only pass `video=<frame>` on the chunk corresponding to a new video timestamp; pass `video=None` on intermediate audio-only chunks.

- [ ] **Step 3: Re-run demo and verify chunk emit pattern**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
grep "aero_dbg_emit" workspace/logs/aero_realtime_tp2_*.log | head -30
```

Expected (per spec §2):
- Chunk 0: full envelope, `new_video=True audio_pads=1 prompt_len ≈ envelope+1`.
- Most subsequent chunks: `new_video=False audio_pads=1 prompt_len == 1`.
- Chunks at new-video-frame boundaries: `new_video=True audio_pads=1 prompt_len ≈ envelope+1` again.

- [ ] **Step 4: Commit**

```bash
git add examples/offline_inference/aero_realtime/offline_realtime_debug.py \
        vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "fix(aero-realtime): audio-only chunks set num_video_tokens=0 (spec I4)"
```

---

## Task 4: Rewrite `build_audio_realtime_text_stream_ids` per spec I5

**Why:** Spec I5: audio_pad slot stream value = `last_generated_token_id` if available (consume once), else `rt_pad`. Drop `rt_speak` segment-start branch and `audio_segment_first_pending` state.

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::build_audio_realtime_text_stream_ids` (line 492)
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::AeroRealtimeStreamState` (line 299)
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::build_video_realtime_text_stream_ids` (line ~466) — drop the same rt_start/rt_speak branches if present

- [ ] **Step 1: Implement the audio stream builder**

Replace `build_audio_realtime_text_stream_ids` (line 492-514) with:

```python
@staticmethod
def build_audio_realtime_text_stream_ids(
    input_ids: list[int],
    state: AeroRealtimeStreamState,
    *,
    audio_token_id: int,
    rt_pad_id: int,
    rt_speak_id: int,  # kept for signature compat; unused
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
```

- [ ] **Step 2: Drop unused state fields**

In `AeroRealtimeStreamState` (line 299), remove `rt_start_emitted`, `rt_speak_emitted`, `audio_segment_first_pending` (and any usages in `build_video_realtime_text_stream_ids` or elsewhere). Keep:
- `chat_started`
- `audio_segment_open`
- `last_generated_token_id`
- `current_time_seconds`
- (any others that are still referenced by code outside this plan's scope)

If `build_video_realtime_text_stream_ids` (line ~466) uses the dropped fields, simplify it analogously: audio_pad slot rule is identical to the audio version (consume `last_generated_token_id` or default to `rt_pad`).

- [ ] **Step 3: Run demo and verify chunk emit pattern**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
grep "aero_dbg_emit" workspace/logs/aero_realtime_tp2_*.log | head -20
```

Expected:
- Chunk 0's `stream_tail` shows `rt_pad` (151673) at the audio_pad position — NOT `rt_speak` (151674).
- Subsequent audio-only chunks: `stream_head/tail` = `[<previously sampled token>]` (whatever the model sampled).

- [ ] **Step 4: Commit**

```bash
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "fix(aero-realtime): text_stream slot rule (rt_pad default, consume prev sampled)"
```

---

## Task 5: Filter placeholder tokens out of generated-token feedback

**Why:** Spec I5 ("real text token only"). Current `_drain_generated_tokens` accepts rt_pad as a "generated token" and propagates it, defeating the teacher-forcing semantic.

**Files:**
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::_drain_generated_tokens` (line 529)
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py::AeroRealtimeGeneration` class body (add constant)

- [ ] **Step 1: Add placeholder id constant near other class attributes in `AeroRealtimeGeneration`**

```python
_AERO_PLACEHOLDER_TOKEN_IDS = frozenset({
    151669,  # audio_start
    151670,  # audio_end
    151671,  # audio_pad
    151672,  # rt_start
    151673,  # rt_pad
    151674,  # rt_speak
    151675,  # rt_end
})
```

- [ ] **Step 2: Rewrite `_drain_generated_tokens`**

```python
@classmethod
def _drain_generated_tokens(
    cls,
    input_stream: asyncio.Queue[list[int]],
    state: AeroRealtimeStreamState,
) -> None:
    while not input_stream.empty():
        token_ids = input_stream.get_nowait()
        if not token_ids:
            continue
        for tid in token_ids:
            if tid in cls._AERO_PLACEHOLDER_TOKEN_IDS:
                continue
            state.last_generated_token_id = tid
```

Note: this is a `@classmethod` change. Existing callers use `cls._drain_generated_tokens(...)` so this is compatible.

- [ ] **Step 3: Run demo**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
grep "\[response\]" workspace/logs/aero_realtime_tp2_*.log | tail -5
```

Expected: response is more coherent. No `II / can can / the the` style duplicates.

- [ ] **Step 4: Commit**

```bash
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "fix(aero-realtime): drain skips placeholder ids; only real text feeds back"
```

---

## Task 6: End-to-end acceptance (chunk logic + decode response + KV reuse) and cleanup

**Why:** Demonstrate the two acceptance criteria hold. Then remove instrumentation.

**Files:**
- Verify: `workspace/run_aero_realtime_tp2.sh`
- Verify: `workspace/logs/aero_realtime_tp2_*.log`
- Modify: `vllm_omni/model_executor/models/aero_realtime/aero_realtime.py` (remove instrumentation)

- [ ] **Step 1: Run with instrumentation still on**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
LOG=$(ls -t workspace/logs/aero_realtime_tp2_*.log | head -1)
echo "LOG=$LOG"
```

- [ ] **Step 2: Acceptance check #1 — chunk emit logic matches spec §2**

```bash
grep "aero_dbg_emit" "$LOG" | head -20
```

Verify by inspection:
- chunk 0: `new_video=True`, `audio_pads=1`, `prompt_len == stream_len`, prompt ends with audio_pad (151671), stream ends with rt_pad (151673).
- Subsequent audio-only chunks: `new_video=False`, `audio_pads=1`, `prompt_len == 1`, `stream_len == 1`, `prompt_head=[151671]`, `stream_head=[<previously sampled token id>]`.
- Video-reopen chunks (whenever a new video frame arrives): `new_video=True`, `audio_pads=1`, prompt ends with audio_pad, stream ends with rt_pad.

If ANY chunk shows `prompt_len != stream_len` or `audio_pads != 1`, fail this step and debug.

- [ ] **Step 3: Acceptance check #2 — decode response is grammatical**

```bash
grep "\[response\]" "$LOG" | tail -5
```

Expected: a grammatical English sentence describing the 30s video clip, no obvious duplicate tokens (`II`, `can can`, `the the`, `stepped came`). Length comparable to what offline `eval_realtime_ckpt.py` produces on the same sample (a single full sentence).

- [ ] **Step 4: Acceptance check #3 — KV reuse (no full re-prefill per chunk)**

```bash
grep "aero_dbg_model" "$LOG" | awk -F 'span=| total=| offset=| seg=' \
  '{print "span="$2" total="$3" offset="$4" seg="$5}' | head -30
```

Expected:
- On every chunk: `span == total == seg` (worker forward slice equals the new delta exactly).
- For audio-only continuation chunks: `span == 1` (only the 1 new audio_pad is forwarded; the prior prompt's KV is reused).
- For video-reopen chunks: `span == (envelope tokens count) + 1`.

If `span` equals the full cumulative prompt length on any non-first chunk, KV is being re-prefilled — fail this step.

- [ ] **Step 5: If any of #1/#2/#3 fails, debug before proceeding**

Do not remove instrumentation if any acceptance check failed.

- [ ] **Step 6: Remove instrumentation**

Delete:
- The `[aero_dbg_emit]` block added in Task 1 Step 1 (in `buffer_realtime_omni`).
- The `[aero_dbg_model]` block added in Task 1 Step 2 (in `preprocess`).

Do not remove any production code added in Tasks 2–5.

- [ ] **Step 7: Final clean run**

```bash
CUDA_VISIBLE_DEVICES=<idx> GPU_MEMORY_UTILIZATION=0.6 ./workspace/run_aero_realtime_tp2.sh
grep "\[response\]" workspace/logs/aero_realtime_tp2_*.log | tail -5
grep "aero_dbg_" workspace/logs/aero_realtime_tp2_*.log | head -3
```

Expected: grammatical response, no `aero_dbg_*` lines in the log.

- [ ] **Step 8: Commit + push**

```bash
git add vllm_omni/model_executor/models/aero_realtime/aero_realtime.py
git commit -m "verify(aero-realtime): demo passes spec §2 chunk logic + KV reuse + clean response"
git push origin integrate-aero-realtime
```
