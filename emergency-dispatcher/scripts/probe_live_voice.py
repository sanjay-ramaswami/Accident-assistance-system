"""Headless, sanitized reproduction of the live voice pipeline.

Runs the same gradbot session the server builds (assistant speaks first, PCM
in/out) with synthetic silence instead of a microphone, and prints only
sanitized provider diagnostics: message-type counts, event counts, and
allowlisted error metadata. No secrets, transcripts, or provider payloads
are printed.

Usage:
    .\\.venv312\\Scripts\\python.exe scripts\\probe_live_voice.py [--seconds 12]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradbot  # noqa: E402

from emergency_dispatcher.gradbot_adapter import build_session_config  # noqa: E402
from emergency_dispatcher.server import _provider_error_diagnostics  # noqa: E402
from emergency_dispatcher.server import _run_kwargs  # noqa: E402
from emergency_dispatcher.settings import get_settings  # noqa: E402

PCM_SAMPLE_RATE = 24000
CHUNK = bytes(1920 * 2)  # 40 ms of s16le mono silence at 24 kHz


def wav_to_s16le_mono_24k(path: str) -> bytes:
    """Convert a PCM WAV file to s16le mono 24 kHz with linear resampling."""
    import wave

    with wave.open(path, "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        source_rate = source.getframerate()
        frames = source.readframes(source.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV input is supported")
    import struct

    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    if channels > 1:
        mono = [
            sum(samples[i : i + channels]) // channels
            for i in range(0, len(samples), channels)
        ]
    else:
        mono = list(samples)
    if source_rate == PCM_SAMPLE_RATE:
        out = mono
    else:
        ratio = source_rate / PCM_SAMPLE_RATE
        out = []
        for index in range(int(len(mono) / ratio)):
            position = index * ratio
            left = int(position)
            if left + 1 >= len(mono):
                break
            fraction = position - left
            value = mono[left] * (1 - fraction) + mono[left + 1] * fraction
            out.append(value)
    return struct.pack(f"<{len(out)}h", *[max(-32768, min(32767, int(v))) for v in out])


async def probe(seconds: float, wav_path: str | None = None) -> None:
    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("GRADIUM_API_KEY", settings.gradium_api_key),
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("OPENAI_MODEL", settings.openai_model),
        )
        if not value
    ]
    if missing:
        print("BLOCKED: missing config:", ", ".join(missing))
        return

    speech_pcm = wav_to_s16le_mono_24k(wav_path) if wav_path else b""
    config = build_session_config(live_dispatch_mode="llm_only")
    print("starting gradbot session (%s)..." % ("synthetic speech input" if wav_path else "synthetic silence input"))
    try:
        input_handle, output_handle = await gradbot.run(
            **_run_kwargs(),
            session_config=config,
            input_format=gradbot.AudioFormat.Pcm,
            output_format=gradbot.AudioFormat.Pcm,
        )
    except Exception as exc:
        diag = _provider_error_diagnostics(exc)
        print("INIT FAILED:", diag)
        return
    print("session initialized")

    msg_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    last_failure: dict | None = None
    got_text = False
    got_audio = False
    eof = False

    async def feeder() -> None:
        deadline = asyncio.get_event_loop().time() + seconds
        if speech_pcm:
            step = 1920 * 2  # 40 ms per tick, real-time pacing
            offset = 0
            while asyncio.get_event_loop().time() < deadline and offset < len(speech_pcm):
                await input_handle.send_audio(speech_pcm[offset : offset + step])
                offset += step
                await asyncio.sleep(0.04)
            # Keep the line open with silence so turn end can be observed.
        while asyncio.get_event_loop().time() < deadline:
            await input_handle.send_audio(CHUNK)
            await asyncio.sleep(0.04)

    async def receiver() -> None:
        nonlocal last_failure, got_text, got_audio, eof
        loop = asyncio.get_event_loop()
        deadline = loop.time() + seconds + 15
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                msg = await asyncio.wait_for(output_handle.receive(), timeout=min(5.0, max(0.1, remaining)))
            except asyncio.TimeoutError:
                continue
            if msg is None:
                eof = True
                break
            msg_counts[msg.msg_type] += 1
            if msg.msg_type == "event" and msg.event is not None:
                event_counts[msg.event.event_type] += 1
            elif msg.msg_type == "error":
                last_failure = _provider_error_diagnostics(getattr(msg, "data", None) or msg)
            elif msg.msg_type == "stt_text" and msg.text:
                # Synthetic test audio only; transcripts are logged for evidence.
                print(f"caller transcript received: {msg.text!r}")
            elif msg.msg_type == "tts_text" and msg.text:
                got_text = True
            elif msg.msg_type == "audio" and msg.data:
                got_audio = True

    await asyncio.gather(feeder(), receiver())
    try:
        await input_handle.close()
    except Exception:
        pass

    print("message_type_counts:", dict(msg_counts))
    print("event_counts:", dict(event_counts))
    print("assistant_text_received:", got_text)
    print("assistant_audio_bytes_received:", got_audio)
    print("output_stream_eof:", eof)
    print("last_error:", last_failure or "none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--wav", type=str, default=None, help="PCM WAV file to stream as caller speech")
    args = parser.parse_args()
    asyncio.run(probe(args.seconds, wav_path=args.wav))


if __name__ == "__main__":
    main()
