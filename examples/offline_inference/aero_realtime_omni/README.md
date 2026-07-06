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
    --output-wav aero_omni_out.wav \
    --max-model-len 32768
```

`--max-model-len` is often needed on smaller GPUs (the config's declared 262144
context requires ~36 GiB of KV cache for the thinker alone; drop it to 32768 for
typical dev boxes).

Output:
- `[token] ids=... text=[...]` — thinker's streamed text tokens (subtitle).
- `[audio] delta samples=N` — code2wav's PCM delta for each chunk.
- `[audio] wrote M samples (X.XXs) → aero_omni_out.wav` — final WAV file at 24 kHz.
