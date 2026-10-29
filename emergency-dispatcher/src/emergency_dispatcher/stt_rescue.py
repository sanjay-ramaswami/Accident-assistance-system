from __future__ import annotations

import asyncio
import json
import re
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict

from .enhanced_audio import AUDIO_RUN_DIR
from .enhanced_audio import PCM_BYTES_PER_SAMPLE
from .enhanced_audio import PCM_CHANNELS
from .enhanced_audio import PCM_SAMPLE_RATE
from .enhanced_audio import write_pcm_wav
from .settings import Settings


MAX_RESCUE_SECONDS = 9
MAX_RESCUE_BYTES = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS * MAX_RESCUE_SECONDS
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
SONIOX_FILES_URL = "https://api.soniox.com/v1/files"
SONIOX_TRANSCRIPTIONS_URL = "https://api.soniox.com/v1/transcriptions"
SUSPICIOUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbe the child\b", re.IGNORECASE), "unlikely_phrase_be_the_child"),
    (re.compile(r"\breading or not\b", re.IGNORECASE), "likely_breathing_misheard"),
    (re.compile(r"\bis pretty\b", re.IGNORECASE), "likely_breathing_misheard"),
    (re.compile(r"\bmouse castle\b", re.IGNORECASE), "location_gibberish"),
    (re.compile(r"\baircar\b", re.IGNORECASE), "location_gibberish"),
    (re.compile(r"\bapproximately between\b", re.IGNORECASE), "unstable_location_phrase"),
)
HIGH_IMPACT_TERMS = ("child", "children", "weapon", "gun", "shooting", "explosion", "drowning")
COMMON_WORDS = {
    "hello",
    "right",
    "close",
    "somewhere",
    "berlin",
    "middle",
    "nowhere",
    "approximately",
    "between",
    "flat",
    "tire",
    "car",
    "stuck",
    "dont",
    "know",
}
EMERGENCY_KEYWORDS: dict[str, float] = {
    "delta campus": 3.0,
    "lake": 2.5,
    "river": 2.5,
    "water": 2.0,
    "drowning": 4.0,
    "sinking": 4.0,
    "breathing": 3.0,
    "bleeding": 3.0,
    "injured": 2.0,
    "trapped": 2.5,
    "accident": 2.0,
    "car": 1.5,
}
PROVIDER_PRIORITY = {
    "deepgram": 0,
    "soniox": 1,
    "gradium": 2,
}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_for_compare(value: str | None) -> str:
    return re.sub(r"[^a-z0-9\s]", "", _normalize_text(value).casefold())


def _state_text(current_state: dict[str, Any] | None) -> str:
    if not current_state:
        return ""
    parts: list[str] = []
    for value in current_state.values():
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, int):
            parts.append(str(value))
    return " ".join(parts).casefold()


def detect_suspicious_transcript(text: str, current_state: dict[str, Any] | None = None) -> list[str]:
    normalized = _normalize_text(text)
    lowered = normalized.casefold()
    reasons: list[str] = []
    for pattern, label in SUSPICIOUS_PATTERNS:
        if pattern.search(normalized):
            reasons.append(label)

    fragments = [item.strip() for item in re.split(r"[.!?]+", normalized) if item.strip()]
    short_fragments = sum(1 for item in fragments if len(item.split()) <= 2)
    if len(fragments) >= 3 and short_fragments >= 2:
        reasons.append("fragmented_short_clauses")

    state_text = _state_text(current_state)
    tokens = re.findall(r"[a-z']{5,}", lowered)
    repeated_rare_tokens = {
        token
        for token in tokens
        if tokens.count(token) >= 2 and token not in COMMON_WORDS and token not in state_text
    }
    if repeated_rare_tokens:
        reasons.append("repeated_rare_token")

    if "child" in lowered and "child" not in state_text and "children" not in state_text:
        reasons.append("unexpected_child_mention")
    if "weapon" in lowered and "weapon" not in state_text and "gun" not in state_text:
        reasons.append("unexpected_weapon_mention")

    token_count = len(re.findall(r"\b\w+\b", lowered))
    if token_count >= 12 and "delta campus" in lowered and not any(term in lowered for term in ("help", "accident", "fire", "bleeding", "breathing", "drowning")):
        reasons.append("location_only_with_low_semantic_signal")

    return list(dict.fromkeys(reasons))


def _score_transcript(
    transcript: str | None,
    *,
    primary_text: str,
    current_state: dict[str, Any] | None = None,
) -> float:
    normalized = _normalize_text(transcript)
    if not normalized:
        return -100.0
    lowered = normalized.casefold()
    score = min(len(re.findall(r"\b\w+\b", lowered)), 40) * 0.15
    state_text = _state_text(current_state)
    for term, weight in EMERGENCY_KEYWORDS.items():
        if term in lowered:
            score += weight
            if term in state_text:
                score += 0.75
    for pattern, _label in SUSPICIOUS_PATTERNS:
        if pattern.search(normalized):
            score -= 6.0
    if "child" in lowered and "child" not in state_text and "children" not in state_text:
        score -= 3.5
    if "weapon" in lowered and "weapon" not in state_text and "gun" not in state_text:
        score -= 3.5
    if _normalize_for_compare(normalized) == _normalize_for_compare(primary_text):
        score += 0.5
    return score


def _should_force_deepgram_turn(primary_text: str, settings: Settings) -> bool:
    if not settings.deepgram_api_key or not settings.stt_rescue_continuous_compare:
        return False
    word_count = len(re.findall(r"\b\w+\b", _normalize_text(primary_text)))
    return word_count >= 1


@dataclass
class LiveAudioWindow:
    raw_pcm: bytearray
    clean_pcm: bytearray
    rescue_count: int = 0

    def append(self, raw_bytes: bytes, clean_bytes: bytes | None = None) -> None:
        if raw_bytes:
            self.raw_pcm.extend(raw_bytes)
            overflow = len(self.raw_pcm) - MAX_RESCUE_BYTES
            if overflow > 0:
                del self.raw_pcm[:overflow]
        if clean_bytes:
            self.clean_pcm.extend(clean_bytes)
            overflow = len(self.clean_pcm) - MAX_RESCUE_BYTES
            if overflow > 0:
                del self.clean_pcm[:overflow]

    def snapshot(self) -> tuple[bytes, bytes]:
        return bytes(self.raw_pcm), bytes(self.clean_pcm)


LIVE_AUDIO_WINDOWS: dict[str, LiveAudioWindow] = {}


def ensure_live_audio_window(session_id: str) -> LiveAudioWindow:
    window = LIVE_AUDIO_WINDOWS.get(session_id)
    if window is None:
        window = LiveAudioWindow(raw_pcm=bytearray(), clean_pcm=bytearray())
        LIVE_AUDIO_WINDOWS[session_id] = window
    return window


def append_live_audio_window(session_id: str, raw_bytes: bytes, clean_bytes: bytes | None = None) -> None:
    ensure_live_audio_window(session_id).append(raw_bytes, clean_bytes)


def clear_live_audio_window(session_id: str) -> None:
    LIVE_AUDIO_WINDOWS.pop(session_id, None)


class RescueCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    transcript: str | None = None
    score: float = 0.0
    latency_s: float | None = None
    usable: bool = False
    error: str | None = None


class STTRescueEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_index: int
    suspicious_reasons: list[str]
    primary_transcript: str
    selected_provider: str
    selected_transcript: str
    corrected_transcript: str
    ask_for_repeat: bool
    location_needs_spelling: bool
    clip_source: str
    clip_path: str | None = None
    primary_score: float = 0.0
    continuous_compare: bool = False
    rescue_elapsed_s: float = 0.0
    shadow_active: bool = False
    shadow_elapsed_s: float = 0.0
    active_provider: str = "gradium"
    provider_scores: dict[str, float] = {}
    candidates: list[RescueCandidate]


def _write_recent_clip(session_id: str, pcm_bytes: bytes, suffix: str) -> Path | None:
    if not pcm_bytes:
        return None
    window = ensure_live_audio_window(session_id)
    window.rescue_count += 1
    AUDIO_RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_RUN_DIR / f"{session_id}_rescue_{window.rescue_count}_{suffix}.wav"
    write_pcm_wav(path, pcm_bytes)
    return path


def _wav_bytes_from_pcm(pcm_bytes: bytes) -> bytes:
    if not pcm_bytes:
        return b""
    from io import BytesIO

    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(PCM_CHANNELS)
        handle.setsampwidth(PCM_BYTES_PER_SAMPLE)
        handle.setframerate(PCM_SAMPLE_RATE)
        handle.writeframes(pcm_bytes)
    return buffer.getvalue()


async def _deepgram_transcribe(wav_bytes: bytes, settings: Settings) -> RescueCandidate:
    if not settings.deepgram_api_key:
        return RescueCandidate(provider="deepgram", error="missing_api_key")
    started_at = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                DEEPGRAM_URL,
                params={
                    "model": "nova-3",
                    "smart_format": "true",
                    "language": "en",
                    "punctuate": "true",
                },
                headers={
                    "Authorization": f"Token {settings.deepgram_api_key}",
                    "Content-Type": "audio/wav",
                },
                content=wav_bytes,
            )
            response.raise_for_status()
            payload = response.json()
        transcript = (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript")
        )
        transcript = _normalize_text(transcript)
        return RescueCandidate(
            provider="deepgram",
            transcript=transcript or None,
            latency_s=time.perf_counter() - started_at,
            usable=bool(transcript),
        )
    except Exception as exc:
        return RescueCandidate(provider="deepgram", error=str(exc), latency_s=time.perf_counter() - started_at)


async def _soniox_transcribe(wav_bytes: bytes, settings: Settings) -> RescueCandidate:
    if not settings.soniox_api_key:
        return RescueCandidate(provider="soniox", error="missing_api_key")
    started_at = time.perf_counter()
    headers = {"Authorization": f"Bearer {settings.soniox_api_key}"}
    file_id: str | None = None
    transcription_id: str | None = None
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            upload = await client.post(
                SONIOX_FILES_URL,
                headers=headers,
                files={"file": ("rescue.wav", wav_bytes, "audio/wav")},
            )
            upload.raise_for_status()
            upload_payload = upload.json()
            file_id = upload_payload.get("id") or upload_payload.get("file_id")
            if not file_id:
                raise RuntimeError("Missing Soniox file id")

            create = await client.post(
                SONIOX_TRANSCRIPTIONS_URL,
                headers={**headers, "Content-Type": "application/json"},
                content=json.dumps(
                    {
                        "file_id": file_id,
                        "model": "stt-async-preview",
                        "language_hints": ["en"],
                    }
                ),
            )
            create.raise_for_status()
            create_payload = create.json()
            transcription_id = create_payload.get("id") or create_payload.get("transcription_id")
            if not transcription_id:
                raise RuntimeError("Missing Soniox transcription id")

            transcript_text = None
            for _ in range(20):
                status_response = await client.get(
                    f"{SONIOX_TRANSCRIPTIONS_URL}/{transcription_id}",
                    headers=headers,
                )
                status_response.raise_for_status()
                status_payload = status_response.json()
                status_value = (status_payload.get("status") or "").casefold()
                if status_value in {"completed", "done", "succeeded"}:
                    transcript_text = (
                        status_payload.get("transcript")
                        or status_payload.get("text")
                        or status_payload.get("result", {}).get("transcript")
                    )
                    if not transcript_text:
                        transcript_response = await client.get(
                            f"{SONIOX_TRANSCRIPTIONS_URL}/{transcription_id}/transcript",
                            headers=headers,
                        )
                        transcript_response.raise_for_status()
                        transcript_payload = transcript_response.json()
                        transcript_text = (
                            transcript_payload.get("transcript")
                            or transcript_payload.get("text")
                        )
                    break
                if status_value in {"failed", "error"}:
                    raise RuntimeError(status_payload.get("error") or "Soniox transcription failed")
                await asyncio.sleep(0.75)

            transcript = _normalize_text(transcript_text)
            candidate = RescueCandidate(
                provider="soniox",
                transcript=transcript or None,
                latency_s=time.perf_counter() - started_at,
                usable=bool(transcript),
            )

            if transcription_id:
                await client.delete(f"{SONIOX_TRANSCRIPTIONS_URL}/{transcription_id}", headers=headers)
            if file_id:
                await client.delete(f"{SONIOX_FILES_URL}/{file_id}", headers=headers)
            return candidate
    except Exception as exc:
        return RescueCandidate(provider="soniox", error=str(exc), latency_s=time.perf_counter() - started_at)


async def _collect_rescue_candidates(wav_bytes: bytes, settings: Settings) -> list[RescueCandidate]:
    return await _collect_rescue_candidates_for_mode(wav_bytes, settings)


async def _collect_rescue_candidates_for_mode(
    wav_bytes: bytes,
    settings: Settings,
    *,
    parallel_shadow: bool = False,
    preferred_provider: str | None = None,
) -> list[RescueCandidate]:
    if parallel_shadow:
        tasks: list[asyncio.Future[RescueCandidate]] = []
        if settings.deepgram_api_key:
            tasks.append(asyncio.create_task(_deepgram_transcribe(wav_bytes, settings)))
        elif settings.soniox_api_key:
            tasks.append(asyncio.create_task(_soniox_transcribe(wav_bytes, settings)))
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    candidates: list[RescueCandidate] = []
    if preferred_provider == "soniox" and settings.soniox_api_key:
        soniox_candidate = await _soniox_transcribe(wav_bytes, settings)
        candidates.append(soniox_candidate)
        if (not soniox_candidate.usable) and settings.deepgram_api_key:
            candidates.append(await _deepgram_transcribe(wav_bytes, settings))
        return candidates

    deepgram_candidate: RescueCandidate | None = None
    if settings.deepgram_api_key:
        deepgram_candidate = await _deepgram_transcribe(wav_bytes, settings)
        candidates.append(deepgram_candidate)
    if settings.soniox_api_key and (deepgram_candidate is None or not deepgram_candidate.usable):
        candidates.append(await _soniox_transcribe(wav_bytes, settings))
    return candidates


def _shadow_elapsed_seconds(session: dict[str, Any]) -> float:
    started_at = (
        session.get("server", {}).get("started_at")
        or session.get("created_at")
    )
    if not started_at:
        return 0.0
    try:
        return max(0.0, (datetime.now().astimezone() - datetime.fromisoformat(str(started_at))).total_seconds())
    except Exception:
        return 0.0


def _best_provider_from_stats(
    provider_stats: dict[str, Any],
    *,
    default_provider: str = "gradium",
) -> str:
    gradium_avg = float(provider_stats.get("gradium", {}).get("avg_score", float("-inf")))
    best_provider = default_provider
    best_avg = gradium_avg
    for provider, payload in provider_stats.items():
        turns = int(payload.get("turns", 0) or 0)
        avg_score = float(payload.get("avg_score", float("-inf")))
        if turns <= 0:
            continue
        if avg_score > best_avg + 0.25:
            best_provider = provider
            best_avg = avg_score
    return best_provider


def _select_rescue_candidate(
    primary_text: str,
    candidates: list[RescueCandidate],
    *,
    suspicious_reasons: list[str],
    current_state: dict[str, Any] | None = None,
    preferred_provider: str | None = None,
    force_preferred: bool = False,
) -> tuple[str, str, list[RescueCandidate], bool]:
    primary_score = _score_transcript(primary_text, primary_text=primary_text, current_state=current_state)
    scored_candidates: list[RescueCandidate] = []
    best_provider = "gradium"
    best_text = _normalize_text(primary_text)
    best_score = primary_score
    deepgram_candidate: RescueCandidate | None = None
    preferred_candidate: RescueCandidate | None = None

    for candidate in candidates:
        transcript = _normalize_text(candidate.transcript)
        score = _score_transcript(transcript, primary_text=primary_text, current_state=current_state)
        updated = candidate.model_copy(update={"transcript": transcript or None, "score": score})
        scored_candidates.append(updated)
        if preferred_provider and updated.provider == preferred_provider and updated.usable:
            preferred_candidate = updated
        if updated.provider == "deepgram" and updated.usable and score >= primary_score + 0.25:
            deepgram_candidate = updated
        if updated.usable and (
            score > best_score + 1.5
            or (
                abs(score - best_score) < 0.01
                and PROVIDER_PRIORITY.get(updated.provider, 99) < PROVIDER_PRIORITY.get(best_provider, 99)
            )
        ):
            best_provider = updated.provider
            best_text = transcript
            best_score = score

    if force_preferred and preferred_candidate is not None:
        best_provider = preferred_candidate.provider
        best_text = _normalize_text(preferred_candidate.transcript)
        best_score = preferred_candidate.score
    elif deepgram_candidate is not None:
        best_provider = deepgram_candidate.provider
        best_text = _normalize_text(deepgram_candidate.transcript)
        best_score = deepgram_candidate.score

    ask_for_repeat = bool(suspicious_reasons)
    return best_provider, best_text or _normalize_text(primary_text), scored_candidates, ask_for_repeat


async def maybe_run_stt_rescue(
    *,
    session_id: str,
    session: dict[str, Any],
    settings: Settings,
) -> STTRescueEvent | None:
    rescue_started_at = time.perf_counter()
    user_items = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    if not user_items:
        return None
    user_index = len(user_items) - 1
    rescue_meta = session.get("stt_rescue_meta") or {}
    if int(rescue_meta.get("last_rescued_user_index", -1)) >= user_index:
        return None

    primary_text = _normalize_text(user_items[user_index].get("text"))
    if not primary_text:
        return None
    current_state = session.get("merged_triage_state") or {}
    suspicious_reasons = detect_suspicious_transcript(primary_text, current_state)
    shadow_elapsed_s = 0.0
    shadow_active = False
    active_provider = str(rescue_meta.get("active_provider") or "gradium")
    continuous_compare = _should_force_deepgram_turn(primary_text, settings)
    always_compare = bool(suspicious_reasons) or continuous_compare
    if not suspicious_reasons and not always_compare:
        return None

    window = LIVE_AUDIO_WINDOWS.get(session_id)
    if window is None:
        return None
    raw_pcm, clean_pcm = window.snapshot()
    clip_pcm = clean_pcm or raw_pcm
    clip_source = "clean" if clean_pcm else "raw"
    if not clip_pcm:
        return None

    clip_path = _write_recent_clip(session_id, clip_pcm, clip_source)
    wav_bytes = _wav_bytes_from_pcm(clip_pcm)
    preferred_provider = active_provider if active_provider != "gradium" else None
    compare_started_at = time.perf_counter()
    candidates = (
        await _collect_rescue_candidates_for_mode(
            wav_bytes,
            settings,
            parallel_shadow=continuous_compare,
            preferred_provider=preferred_provider,
        )
        if (settings.deepgram_api_key or settings.soniox_api_key)
        else []
    )
    shadow_elapsed_s = time.perf_counter() - compare_started_at if continuous_compare else 0.0
    shadow_active = bool(continuous_compare and candidates)
    selected_provider, corrected_text, scored_candidates, ask_for_repeat = _select_rescue_candidate(
        primary_text,
        list(candidates),
        suspicious_reasons=suspicious_reasons,
        current_state=current_state,
        preferred_provider=preferred_provider,
        force_preferred=bool(preferred_provider == "deepgram"),
    )
    primary_score = _score_transcript(primary_text, primary_text=primary_text, current_state=current_state)
    provider_scores = {"gradium": primary_score}
    for candidate in scored_candidates:
        provider_scores[candidate.provider] = candidate.score
    if (
        always_compare
        and not suspicious_reasons
        and selected_provider == "gradium"
        and _normalize_for_compare(corrected_text) == _normalize_for_compare(primary_text)
    ):
        if not continuous_compare:
            return None
    generic_location = _normalize_text(str(current_state.get("location") or "")).casefold()
    location_needs_spelling = (
        "location_gibberish" in suspicious_reasons
        or "unstable_location_phrase" in suspicious_reasons
        or "repeated_rare_token" in suspicious_reasons
        or generic_location in {"", "berlin", "berlin germany", "berlin, germany"}
    )
    return STTRescueEvent(
        user_index=user_index,
        suspicious_reasons=suspicious_reasons,
        primary_transcript=primary_text,
        selected_provider=selected_provider,
        selected_transcript=corrected_text,
        corrected_transcript=corrected_text,
        ask_for_repeat=bool(suspicious_reasons) and ask_for_repeat,
        location_needs_spelling=location_needs_spelling,
        clip_source=clip_source,
        clip_path=str(clip_path) if clip_path else None,
        primary_score=primary_score,
        continuous_compare=continuous_compare,
        rescue_elapsed_s=time.perf_counter() - rescue_started_at,
        shadow_active=shadow_active,
        shadow_elapsed_s=shadow_elapsed_s,
        active_provider=active_provider,
        provider_scores=provider_scores,
        candidates=scored_candidates,
    )


def stt_rescue_session_patch(
    session: dict[str, Any],
    event: STTRescueEvent,
) -> dict[str, Any]:
    rescue_events = list(session.get("stt_rescue_events", []) or [])
    rescue_events.append(event.model_dump())
    overrides = dict(session.get("stt_rescue_overrides", {}) or {})
    if event.selected_transcript and _normalize_for_compare(event.selected_transcript) != _normalize_for_compare(event.primary_transcript):
        overrides[str(event.user_index)] = event.selected_transcript
    prior_meta = dict(session.get("stt_rescue_meta", {}) or {})
    provider_stats = dict(prior_meta.get("provider_stats", {}) or {})
    for provider, score in (event.provider_scores or {}).items():
        stats = dict(provider_stats.get(provider, {}) or {})
        turns = int(stats.get("turns", 0) or 0) + 1
        total_score = float(stats.get("total_score", 0.0) or 0.0) + float(score)
        provider_stats[provider] = {
            "turns": turns,
            "total_score": total_score,
            "avg_score": round(total_score / max(turns, 1), 4),
        }
    active_provider = str(prior_meta.get("active_provider") or event.active_provider or "gradium")
    provider_locked = bool(prior_meta.get("provider_locked", False))
    switch_events = list(prior_meta.get("switch_events", []) or [])
    selected_score = float((event.provider_scores or {}).get(event.selected_provider, float("-inf")))
    primary_score = float((event.provider_scores or {}).get("gradium", event.primary_score))
    should_switch_to_deepgram = (
        event.selected_provider == "deepgram"
        and (
            bool(event.suspicious_reasons)
            or event.ask_for_repeat
            or selected_score >= primary_score + 0.25
        )
    )
    if should_switch_to_deepgram and active_provider != "deepgram":
        switch_events.append(
            {
                "at": datetime.now().astimezone().isoformat(),
                "from": active_provider,
                "to": "deepgram",
                "reason": "continuous_backend_compare_win",
            }
        )
        active_provider = "deepgram"
        provider_locked = True
    elif not provider_locked:
        next_provider = _best_provider_from_stats(provider_stats, default_provider=active_provider)
        if next_provider != active_provider:
            switch_events.append(
                {
                    "at": datetime.now().astimezone().isoformat(),
                    "from": active_provider,
                    "to": next_provider,
                    "reason": "continuous_backend_compare_average",
                }
            )
            active_provider = next_provider
            provider_locked = active_provider != "gradium"
    return {
        "stt_rescue_events": rescue_events,
        "stt_rescue_overrides": overrides,
        "stt_rescue_meta": {
            "last_rescued_user_index": event.user_index,
            "last_selected_provider": event.selected_provider,
            "last_ask_for_repeat": event.ask_for_repeat,
            "active_provider": active_provider,
            "provider_locked": provider_locked,
            "continuous_compare": event.continuous_compare,
            "shadow_active": event.shadow_active,
            "shadow_elapsed_s": round(event.shadow_elapsed_s, 2),
            "provider_stats": provider_stats,
            "switch_events": switch_events,
        },
    }
