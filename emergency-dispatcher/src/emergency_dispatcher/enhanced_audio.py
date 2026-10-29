from __future__ import annotations

import asyncio
import functools
import re
import wave
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import aic_sdk as aic
import numpy as np
from openai import OpenAI

from .settings import REPO_ROOT
from .settings import Settings


AUDIO_RUN_DIR = REPO_ROOT / "runs" / "audio"
PCM_SAMPLE_RATE = 24_000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2


def _safe_transcription_error(error: Exception) -> tuple[str, bool]:
    """Return allowlisted metadata only; never persist a provider message."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(error, "response", None), "status_code", None)
    status = status if isinstance(status, int) and 100 <= status <= 599 else None
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        body = {}
    nested = body.get("error") if isinstance(body.get("error"), dict) else body
    error_type = nested.get("type") if isinstance(nested, dict) else None
    error_code = nested.get("code") if isinstance(nested, dict) else None
    allowed_types = {"insufficient_quota", "authentication_error", "invalid_request_error", "rate_limit_exceeded", "server_error"}
    allowed_codes = {"insufficient_quota", "credit_balance_exhausted", "rate_limit_exceeded", "invalid_api_key", "model_not_found"}
    if not isinstance(error_type, str) or error_type not in allowed_types:
        error_type = None
    if not isinstance(error_code, str) or error_code not in allowed_codes:
        error_code = None
    quota = error_type == "insufficient_quota" or error_code in {"insufficient_quota", "credit_balance_exhausted"}
    exception_name = type(error).__name__
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", exception_name):
        exception_name = "ProviderError"
    details = [exception_name]
    if status:
        details.append(f"HTTP {status}")
    if error_type:
        details.append(error_type)
    if error_code:
        details.append(error_code)
    return "; ".join(details), quota


def _pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    float32 = int16.astype(np.float32) / 32768.0
    return float32.reshape(1, -1)


def _float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    return int16.tobytes()


@functools.lru_cache(maxsize=1)
def _model_path(model_id: str, model_dir: str) -> str:
    path = aic.Model.download(model_id, model_dir)
    return str(path)


def _process_pcm_bytes(processor: aic.Processor, pcm_bytes: bytes) -> bytes:
    if not pcm_bytes:
        return b""
    samples = _pcm16_bytes_to_float32(pcm_bytes)
    processed = processor.process(samples)
    return _float32_to_pcm16_bytes(processed)


@dataclass
class EnhancerStream:
    processor: aic.Processor
    bytes_per_chunk: int
    pending: bytearray = field(default_factory=bytearray)

    def process_bytes(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""
        self.pending.extend(pcm_bytes)
        chunks: list[bytes] = []
        while len(self.pending) >= self.bytes_per_chunk:
            chunk = bytes(self.pending[: self.bytes_per_chunk])
            del self.pending[: self.bytes_per_chunk]
            chunks.append(_process_pcm_bytes(self.processor, chunk))
        return b"".join(chunks)

    def flush(self) -> bytes:
        if not self.pending:
            return b""
        original_len = len(self.pending)
        pad_len = (-original_len) % self.bytes_per_chunk
        padded = bytes(self.pending) + (b"\x00" * pad_len)
        self.pending.clear()
        processed = _process_pcm_bytes(self.processor, padded)
        return processed[:original_len]


def make_enhancer(settings: Settings) -> EnhancerStream:
    settings.ai_coustics_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(
        settings.ai_coustics_model_id,
        str(settings.ai_coustics_model_dir),
    )
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig.optimal(
        model,
        sample_rate=PCM_SAMPLE_RATE,
        num_channels=PCM_CHANNELS,
        allow_variable_frames=False,
    )
    processor = aic.Processor(model, settings.ai_coustics_api_key, config)
    bytes_per_chunk = config.num_frames * config.num_channels * PCM_BYTES_PER_SAMPLE
    return EnhancerStream(
        processor=processor,
        bytes_per_chunk=bytes_per_chunk,
    )


def write_pcm_wav(path: Path, pcm_bytes: bytes) -> None:
    AUDIO_RUN_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(PCM_CHANNELS)
        handle.setsampwidth(PCM_BYTES_PER_SAMPLE)
        handle.setframerate(PCM_SAMPLE_RATE)
        handle.writeframes(pcm_bytes)


def _transcribe_wav_sync(path: Path, settings: Settings) -> str | None:
    if not settings.openai_api_key:
        return None
    client_kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key,
        "base_url": (
            settings.openai_base_url
            if settings.openai_base_url and settings.openai_base_url.startswith(("http://", "https://"))
            else "https://api.openai.com/v1"
        ),
    }
    client = OpenAI(**client_kwargs)
    with path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model=settings.openai_transcription_model,
            response_format="text",
            prompt=(
                "Emergency dispatch call transcript. Preserve street names, place names, "
                "numbers, location clues, and emergency wording."
            ),
        )
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", "")
    return text.strip() or None


async def finalize_audio_artifacts(
    *,
    session_id: str,
    raw_pcm: bytes,
    clean_pcm: bytes,
    settings: Settings,
) -> dict[str, str | None]:
    AUDIO_RUN_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = AUDIO_RUN_DIR / f"{session_id}_raw.wav"
    clean_path = AUDIO_RUN_DIR / f"{session_id}_clean.wav"
    if raw_pcm:
        await asyncio.to_thread(write_pcm_wav, raw_path, raw_pcm)
    if clean_pcm:
        await asyncio.to_thread(write_pcm_wav, clean_path, clean_pcm)

    raw_reconstruction = None
    clean_reconstruction = None
    raw_reconstruction_error = None
    clean_reconstruction_error = None
    quota_failure = False
    if raw_pcm:
        try:
            raw_reconstruction = await asyncio.to_thread(
                _transcribe_wav_sync,
                raw_path,
                settings,
            )
        except Exception as exc:
            raw_reconstruction_error, quota_failure = _safe_transcription_error(exc)
    if clean_pcm and not quota_failure:
        try:
            clean_reconstruction = await asyncio.to_thread(
                _transcribe_wav_sync,
                clean_path,
                settings,
            )
        except Exception as exc:
            clean_reconstruction_error, quota_failure = _safe_transcription_error(exc)
    elif clean_pcm and quota_failure:
        clean_reconstruction_error = "Skipped after transcription quota failure"

    return {
        "raw_audio_path": str(raw_path.relative_to(REPO_ROOT)) if raw_pcm else None,
        "clean_audio_path": str(clean_path.relative_to(REPO_ROOT)) if clean_pcm else None,
        "raw_reconstruction": raw_reconstruction,
        "clean_reconstruction": clean_reconstruction,
        "raw_reconstruction_error": raw_reconstruction_error,
        "clean_reconstruction_error": clean_reconstruction_error,
    }
