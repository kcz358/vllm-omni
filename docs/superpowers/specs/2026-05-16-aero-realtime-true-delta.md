# Aero Realtime — True-Delta Streaming Alignment Spec

**Status:** draft (2026-05-16)
**Branch:** `integrate-aero-realtime`
**Owners:** KC (design), opencode (implementation)

---

## 1. Problem

Aero realtime decode currently produces output with 1-token duplicates and
truncation (`II can can`, `just just`, `the the`, `stepped came`).

Root cause from `workspace/findings_alignment_analysis.md`:
- Stage-0 scheduler/worker fix landed → `additional_information` no longer
  stale, decode-step (1-token forward) is aligned.
- But prefill steps show `span_len=178 / total=95 / audio_pads=0 / changed=0`.
- The runtime `_build_realtime_delta` / `buffer_realtime_omni` **re-emits a
  full per-chunk envelope every chunk** (`audio_end, <t s>, vision_start,
  video_pad×S, vision_end, audio_start, audio_pad×N`, ≈95 tokens) instead
  of emitting a true delta of "just the new audio_pad slots added since
  last chunk".
- The scheduler dutifully appends the 95-token block; the audio_pad core
  inside that block never gets its `text_stream_ids` substitution because
  the next-chunk teacher-forcing token X is supposed to land on **one
  specific new audio_pad slot**, not on a re-replayed envelope.

## 2. Desired logic (worked example, ground truth)

Token ids (41k ckpt):
- `audio_pad = 151671`, `rt_pad = 151673`, `rt_speak = 151674`
- `audio_start = 151669`, `audio_end = 151670`
- `vision_start = 151652`, `vision_end = 151653`, `video_pad = 151656`

**Key principle**: each runtime chunk corresponds to **exactly one new
audio token slot** (one audio_pad). The model decides on every step
whether to emit `<|rt_pad|>` (silence / no text this step) or a real
text token. We do not pre-allocate "audio_pad × N" inside a single
chunk — N is always 1 per chunk.

### Chunk 0 — initial prompt + open first audio segment
```
prompt_token_ids = [im_start, user, "<0.0 s>", vs, vp×S, ve, as, ap]
                                                                  └── exactly ONE audio_pad
text_stream_ids  = [...                                  same...        as, rt_pad]
                                                                          ↑ that ap slot → rt_pad
```

Worker forward → sample X (model is free to emit rt_pad, or rt_speak,
or a real text token id).

### Chunk 1 — pure audio frame (no new video, no text)
```
prompt_token_ids = [ap]                       # length 1 — only NEW audio_pad
text_stream_ids  = [X]                        # length 1 — teacher-force X on that slot
```

Scheduler stage-0 path:
- drops the `X` that vLLM auto-appended to `_output_token_ids` (we feed
  it back in via `text_stream_ids[0]` instead — it lives in the new
  audio_pad slot's stream channel, not in the input_ids token stream).
- appends `[ap]` to `_all_token_ids`.
- preserves `num_computed_tokens` (the prior prompt's KV is still valid).

Worker forward slice = `_all_token_ids[num_computed_tokens:] = [ap]` length 1.
Preprocess sees `input_ids=[151671], text_stream=[X]`, replaces audio_pad
embedding with `embed(X) + audio_embed[0]`. Sample → Y.

### Chunk 2 — pure audio frame
```
prompt_token_ids = [ap]
text_stream_ids  = [Y]
```

### Chunk K — new video frame arrives
A new video frame re-opens the envelope. The first `audio_pad` of the
re-opened envelope has no immediately-prior sampled text token to
teacher-force (the previous chunk closed its segment via `audio_end`
and the new envelope's structural tokens — `<t s>`, `vision_start`,
`video_pad×S`, `vision_end`, `audio_start` — are emitted in between),
so its `text_stream_ids` slot is `rt_pad`:
```
prompt_token_ids = [audio_end, "<t s>", vs, vp×S, ve, audio_start, ap]
text_stream_ids  = [audio_end, "<t s>", vs, vp×S, ve, audio_start, rt_pad]
                                                                   ↑ rt_pad — no prior decode to forward
```

Then audio-only chunks resume the `[ap] / [Z]` pattern (where Z is the
token sampled at the rt_pad slot above, if any).

### KV state after a few chunks (audio-only continuation)
```
slot:     0   1    ...   k       k+1   k+2   k+3   ...
content:  vs, vp,  ...,  ap+rt_pad, ap+X, ap+Y, ap+Z, ...
```

Every audio_pad slot in KV stores `embed(audio_pad) + (embed(stream_id)
- embed(audio_pad)) + audio_embed[...]`. The `stream_id` at slot `i`
is the token sampled when forwarding slot `i-1` (teacher forcing one
step delayed, consistent with training). The very first audio_pad slot
(no prior decode) gets `rt_pad`.

## 3. Invariants the runtime/scheduler/worker must jointly maintain

I1. `len(prompt_token_ids_delta) == len(text_stream_ids_delta)` for every
    chunk emitted by `buffer_realtime_omni`. Length is the number of
    **new** tokens to append. A pure-audio chunk has length 1
    (a single `audio_pad`). A video-bearing chunk has length =
    (structural tokens of envelope) + 1 (one `audio_pad`).

I2. Scheduler stage-0 `_update_request_as_session` drops `_output_token_ids`
    (the X) and appends only `prompt_token_ids_delta`. `num_computed_tokens`
    is preserved. (Already done.)

I3. Worker forward slice == `_all_token_ids[num_computed_tokens:]` ==
    exactly the new delta. Preprocess receives `input_ids` of that length
    and `text_stream_ids` of the same length, aligned 1:1.

I4. Runtime never re-emits already-emitted structural tokens
    (`vision_start`, `video_pad`, `vision_end`, `audio_start`,
    `audio_end`, `<t.t seconds>` strings) on subsequent audio-only chunks.
    Structural tokens only re-appear when a new video frame or text
    interjection actually opens a new envelope.

I5. Each chunk carries exactly one new `audio_pad` slot (one audio sample
    per chunk). `text_stream_ids` at that slot is:
    - the previously-sampled assistant token X (consumed once, comes
      from `state.last_generated_token_id`), OR
    - `rt_pad` when no prior sampled text token is waiting (first audio
      slot of the whole session, or after `last_generated_token_id` was
      already consumed by a prior chunk).
    The model itself decides every step whether to emit `rt_pad` (silence)
    or a real text token — runtime never forces this.

## 4. Out of scope for this spec

- Audio output (model generating audio codec tokens). Already explicitly
  deferred — current scope is audio+video input → text output only.
- TP > 1 correctness. Validated separately.
- Realtime serving over websocket. Offline debug script is the validation
  surface here; serving inherits the runtime/scheduler invariants once
  offline works.
- Changes to `additional_information` transport (we keep it; the
  scheduler/worker stage-0 path already routes it correctly).

## 5. Validation target

Run `./workspace/run_aero_realtime_tp2.sh` (default 30s sample, ask at
t=4s "What is happening now?"). Pass criteria:

- No `II can can`-style duplicated tokens in `[response]` output.
- Response is grammatical English of comparable quality to offline
  `eval_realtime_ckpt.py` on the same sample.
- Worker preprocess dump (when re-enabled temporarily) shows
  `span_len == total == seg_len` on every chunk and `audio_pads > 0,
  changed > 0` on chunks containing audio_pad slots.
