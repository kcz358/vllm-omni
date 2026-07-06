# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for aero_realtime_omni: thinker → talker → code2wav.

Adapted from ``qwen3_omni.py`` (thinker→talker) and its ``talker2code2wav_async_chunk``
with simplifications: aero has no tts_bos/tts_eos/tts_pad thinker special tokens,
no PD prefill merging, and no speaker/language extraction (single speaker is baked
into the talker).

Design point: on the thinker side, we only forward hidden states + word embeddings
at the ``<|audio_pad|>`` slots (not at every thinker input token). The thinker's
per-step output has shape ``[num_tokens_in_step, hidden]``; we filter to
audio_pad positions.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

# aero_realtime_omni fixed vocab id for the ``<|audio_pad|>`` token.
# Source of truth: ``vllm_omni/transformers_utils/configs/aero_realtime.py``
# (``audio_token_index=151671``). Kept as a module constant because the
# per-step ``request`` object does not carry the stage HF config.
_AUDIO_PAD_TOKEN_ID = 151671


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    if not isinstance(x, list):
        return list(x) if x is not None else []
    return list(x)


def _codec_chunk_config(transfer_manager: Any) -> tuple[int, int]:
    """Read ``codec_chunk_frames`` / ``codec_left_context_frames`` from the
    stage connector config, matching the pattern used by the other processors.
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context = int(cfg.get("codec_left_context_frames", 25))
    return chunk_size, left_context


def _filter_audio_pad_rows(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    step_input_ids: list[int],
    audio_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter [T, H] hidden and embed to only the rows at audio_pad positions in this step."""
    if hidden is None or hidden.numel() == 0 or not step_input_ids:
        return hidden, embed
    ids = torch.tensor(step_input_ids, dtype=torch.long, device=hidden.device)
    if ids.shape[0] != hidden.shape[0]:
        # Length mismatch: fall back to a no-op filter rather than crash.
        return hidden, embed
    mask = ids == audio_token_id
    if int(mask.sum().item()) == 0:
        return (
            hidden.new_zeros((0, hidden.shape[-1])),
            embed.new_zeros((0, embed.shape[-1])),
        )
    return hidden[mask], embed[mask]


# ---- thinker → talker -------------------------------------------------------


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Per-step producer hook: extract new audio_pad hidden states from the thinker."""
    if not isinstance(pooling_output, dict):
        return None

    hs = (pooling_output.get("hidden_states") or {}).get("output")
    embed = (pooling_output.get("embed") or {}).get("prefill")
    if not isinstance(hs, torch.Tensor) or not isinstance(embed, torch.Tensor):
        return None

    all_ids = _ensure_list(request.all_token_ids)
    n = int(hs.shape[0])
    step_input_ids = all_ids[-n:] if n > 0 else []

    hs_filtered, emb_filtered = _filter_audio_pad_rows(
        hs, embed, step_input_ids, _AUDIO_PAD_TOKEN_ID
    )
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
    path is the async_chunk hook above. We produce one prompt whose placeholder
    length equals 3 (cond triplet) + N_total_audio_pad_frames.
    """
    stage_id = engine_input_source[0]
    thinker_outputs = stage_list[stage_id].engine_outputs
    talker_inputs: list[OmniTokensPrompt] = []
    device = torch.device(current_platform.device_type)

    for thinker_output in thinker_outputs:
        top = thinker_output.outputs[0]
        mm = top.multimodal_output or {}
        hs = (mm.get("hidden_states") or {}).get("output")
        embed = (mm.get("embed") or {}).get("prefill")
        all_ids = _ensure_list(thinker_output.prompt_token_ids) + _ensure_list(top.cumulative_token_ids)

        if isinstance(hs, torch.Tensor) and isinstance(embed, torch.Tensor) and all_ids:
            n = int(hs.shape[0])
            ids_tail = all_ids[-n:]
            hs, embed = _filter_audio_pad_rows(
                hs.to(device), embed.to(device), ids_tail, _AUDIO_PAD_TOKEN_ID
            )

        payload: OmniPayload = {
            "embed": {
                "prefill": embed.detach().to("cpu")
                if isinstance(embed, torch.Tensor)
                else torch.empty(0)
            },
            "hidden_states": {
                "output": hs.detach().to("cpu")
                if isinstance(hs, torch.Tensor)
                else torch.empty(0)
            },
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
    for talker_output in talker_outputs:
        top = talker_output.outputs[0]
        mm = top.multimodal_output or {}
        codes = (mm.get("codes") or {}).get("audio")
        if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
            inputs.append(OmniTokensPrompt(prompt_token_ids=[], additional_information={}))
            continue
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
