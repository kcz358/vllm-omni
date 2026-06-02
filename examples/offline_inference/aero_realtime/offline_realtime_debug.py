"""Offline streaming demo for AeroRealtime.

Feeds 80 ms audio chunks (and interleaved video frames) into the model in
real-time order and prints decoded tokens as they are generated.

Usage
-----
    python offline_realtime_debug.py \
        --model <hf-id-or-local-path> \
        --video-path /path/to/clip.mp4 \
        --ask-second 4 --ask-text "What is happening now? "

The model path is resolved from (in order):
    1. ``--model`` CLI flag
    2. ``AERO_REALTIME_MODEL`` env var
    3. built-in placeholder ``"<aero-realtime-checkpoint>"`` which will
       fail fast so the user provides one explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

import librosa
import numpy as np
from qwen_vl_utils import fetch_video
from vllm.engine.protocol import StreamingInput
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.model_executor.models.aero_realtime.aero_realtime import (
    AeroRealtimeForConditionalGeneration,
)

SAMPLE_RATE = 16000
MODEL_ENV = "AERO_REALTIME_MODEL"
MODEL_PLACEHOLDER = "<aero-realtime-checkpoint>"


# ---------------------------------------------------------------------------
# Video / audio loading
# ---------------------------------------------------------------------------


@dataclass
class RealtimeSample:
    audio: np.ndarray            # mono float32 @ 16 kHz
    video: np.ndarray            # (T, H, W, 3) uint8
    video_metadata: object       # decord metadata (or dict)
    sample_fps: float


def _load_audio_or_silence(video_path: str, num_frames: int, sample_fps: float) -> np.ndarray:
    try:
        audio, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    except Exception as exc:  # no audio track -> silence matching video duration
        duration_s = num_frames / max(sample_fps, 1e-6)
        print(f"[audio] no audio track ({exc}); using {duration_s:.2f}s silence")
        return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def load_video_sample(video_path: str, *, max_frames: int) -> RealtimeSample:
    video_inputs, sample_fps = fetch_video(
        {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": 1,
            "min_frames": 1,
            "max_frames": max_frames,
            "min_pixels": 28800,
            "max_pixels": 300 * 300,
        },
        return_video_sample_fps=True,
        return_video_metadata=True,
    )
    video, metadata = video_inputs
    audio = _load_audio_or_silence(video_path, video.shape[0], sample_fps)
    print(
        f"[sample] video_shape={tuple(video.shape)} "
        f"sample_fps={sample_fps} audio_shape={audio.shape}"
    )
    return RealtimeSample(audio=audio, video=video, video_metadata=metadata, sample_fps=sample_fps)


# ---------------------------------------------------------------------------
# Realtime chunk generator
# ---------------------------------------------------------------------------


def _meta(metadata, key: str, default=None):
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _slice_video_metadata(metadata, frame_idx: int) -> dict[str, object]:
    frames_indices = list(_meta(metadata, "frames_indices", []))
    frame_index = frames_indices[frame_idx] if frame_idx < len(frames_indices) else frame_idx
    fps = _meta(metadata, "fps", 1.0) or 1.0
    return {
        "fps": fps,
        "duration": 1.0 / float(fps),
        "total_num_frames": 1,
        "frames_indices": [frame_index],
        "video_backend": _meta(metadata, "video_backend", "decord"),
    }


def _frame_time(metadata, frame_idx: int, sample_fps: float) -> float:
    frames_indices = list(_meta(metadata, "frames_indices", []))
    if frame_idx < len(frames_indices):
        fps = _meta(metadata, "fps", None) or sample_fps or 1.0
        return float(frames_indices[frame_idx]) / float(fps)
    return float(frame_idx) / float(sample_fps or 1.0)


def iter_realtime_chunks(
    sample: RealtimeSample,
    *,
    audio_chunk_ms: float,
    ask_second: int | None,
    ask_text: str,
    verbose: bool,
):
    """Yield {audio, video?, text?, timestamp} chunks in real-time order."""
    num_frames = int(sample.video.shape[0])
    samples_per_chunk = max(1, int(round(SAMPLE_RATE * audio_chunk_ms / 1000.0)))
    num_audio_chunks = int(np.ceil(len(sample.audio) / samples_per_chunk))
    frame_times = [_frame_time(sample.video_metadata, i, sample.sample_fps) for i in range(num_frames)]

    next_frame = 0
    for idx in range(num_audio_chunks):
        start = idx * samples_per_chunk
        end = min(start + samples_per_chunk, len(sample.audio))
        audio_chunk = sample.audio[start:end].astype(np.float32, copy=False)
        if audio_chunk.size == 0:
            break
        pad = samples_per_chunk - audio_chunk.shape[0]
        if pad > 0:
            audio_chunk = np.pad(audio_chunk, (0, pad)).astype(np.float32, copy=False)

        t0, t1 = start / SAMPLE_RATE, end / SAMPLE_RATE
        chunk: dict[str, object] = {"audio": audio_chunk, "timestamp": t0}

        if next_frame < num_frames and frame_times[next_frame] < t1:
            video_chunk = sample.video[next_frame : next_frame + 1]
            chunk["video"] = (video_chunk, _slice_video_metadata(sample.video_metadata, next_frame))
            chunk["mm_processor_kwargs"] = {"fps": sample.sample_fps, "do_sample_frames": False}
            if verbose:
                print(f"[input] t={t0:.2f}s attach_frame={next_frame}")
            next_frame += 1

        if ask_second is not None and t0 <= ask_second < t1:
            chunk["text"] = ask_text
            if verbose:
                print(f"[input] t={t0:.2f}s ask={ask_text!r}")

        yield chunk


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------


async def build_streaming_inputs(
    omni: AsyncOmni,
    sample: RealtimeSample,
    input_stream: asyncio.Queue[list[int]],
    *,
    audio_chunk_ms: float,
    inter_chunk_delay_s: float,
    ask_second: int | None,
    ask_text: str,
    verbose: bool,
) -> AsyncGenerator[StreamingInput, None]:
    model_config = omni.model_config
    renderer = omni.renderer
    if model_config is None or renderer is None:
        raise RuntimeError("AsyncOmni did not expose model_config/renderer")

    async def chunks() -> AsyncIterator[dict[str, object]]:
        for chunk in iter_realtime_chunks(
            sample,
            audio_chunk_ms=audio_chunk_ms,
            ask_second=ask_second,
            ask_text=ask_text,
            verbose=verbose,
        ):
            yield chunk
            if inter_chunk_delay_s > 0:
                await asyncio.sleep(inter_chunk_delay_s)

    prompt_iter = AeroRealtimeForConditionalGeneration.buffer_realtime_omni(
        chunks(), input_stream, model_config
    )
    async for prompt in prompt_iter:
        parsed = parse_model_prompt(model_config, prompt)
        (engine_input,) = await renderer.render_cmpl_async([parsed])
        if isinstance(engine_input, dict) and isinstance(prompt, dict):
            extra = prompt.get("additional_information")
            if extra is not None:
                engine_input["additional_information"] = extra
        yield StreamingInput(prompt=engine_input)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline AeroRealtime streaming demo")
    parser.add_argument("--model", default=os.environ.get(MODEL_ENV, MODEL_PLACEHOLDER),
                        help=f"HF id or path. Defaults to ${MODEL_ENV} or {MODEL_PLACEHOLDER}.")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/aero_realtime.yaml")
    parser.add_argument("--video-path", required=True, help="Local video file to stream.")
    parser.add_argument("--video-max-frames", type=int, default=64)
    parser.add_argument("--audio-chunk-ms", type=float, default=80.0)
    parser.add_argument("--inter-chunk-delay-s", type=float, default=0.0)
    parser.add_argument("--ask-second", type=int, default=None,
                        help="Inject --ask-text at this wall-clock second of the stream.")
    parser.add_argument("--ask-text", default="What is happening now? ")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--stage-0-devices", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.model == MODEL_PLACEHOLDER:
        parser.error(f"--model not set; pass --model or export {MODEL_ENV}=...")
    return args


def build_async_omni(args: argparse.Namespace) -> AsyncOmni:
    kwargs: dict[str, object] = {
        "model": args.model,
        "deploy_config": args.deploy_config,
        "log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "skip_mm_profiling": args.skip_mm_profiling,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        kwargs["stage_0_devices"] = args.stage_0_devices or ",".join(
            str(i) for i in range(args.tensor_parallel_size)
        )
    elif args.stage_0_devices is not None:
        kwargs["stage_0_devices"] = args.stage_0_devices
    return AsyncOmni(**kwargs)


async def main() -> None:
    args = parse_args()
    sample = load_video_sample(args.video_path, max_frames=args.video_max_frames)
    omni = build_async_omni(args)
    tokenizer = cached_tokenizer_from_config(omni.model_config)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=1,
        output_kind=RequestOutputKind.DELTA,
        skip_clone=True,
    )

    request_id = f"aero-rt-debug-{uuid.uuid4()}"
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

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
            if not output.outputs:
                continue
            token_ids = list(output.outputs[0].token_ids)
            if not token_ids:
                continue
            input_stream.put_nowait(token_ids)
            decoded = [tokenizer.decode([t]) for t in token_ids]
            print(f"[token] ids={token_ids} text={decoded!r}")
    finally:
        await omni.abort(request_id)


if __name__ == "__main__":
    asyncio.run(main())
