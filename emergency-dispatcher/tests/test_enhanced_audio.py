import asyncio

from emergency_dispatcher import enhanced_audio
from emergency_dispatcher.enhanced_audio import EnhancerStream
from emergency_dispatcher.settings import get_settings


class DummyProcessor:
    def process(self, samples):
        return samples


def test_enhancer_stream_buffers_partial_chunks():
    stream = EnhancerStream(
        processor=DummyProcessor(),
        bytes_per_chunk=8,
    )

    assert stream.process_bytes(b"\x01\x00\x02\x00") == b""
    output = stream.process_bytes(b"\x03\x00\x04\x00")
    assert len(output) == 8
    assert stream.pending == bytearray()


def test_enhancer_stream_flush_preserves_tail_length():
    stream = EnhancerStream(
        processor=DummyProcessor(),
        bytes_per_chunk=8,
    )

    stream.process_bytes(b"\x01\x00\x02")
    output = stream.flush()
    assert len(output) == 3
    assert stream.pending == bytearray()


def test_quota_failure_is_sanitized_and_skips_second_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr(enhanced_audio, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(enhanced_audio, "AUDIO_RUN_DIR", tmp_path / "runs" / "audio")
    calls = []

    class QuotaError(Exception):
        status_code = 429
        body = {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}

    def fail_transcription(_path, _settings):
        calls.append(1)
        raise QuotaError("PRIVATE_CALLER_TRANSCRIPT_AND_CREDENTIAL_SENTINEL")

    monkeypatch.setattr(enhanced_audio, "_transcribe_wav_sync", fail_transcription)
    result = asyncio.run(
        enhanced_audio.finalize_audio_artifacts(
            session_id="safe-quota-diagnostics",
            raw_pcm=b"\x00\x00",
            clean_pcm=b"\x00\x00",
            settings=get_settings(),
        )
    )

    assert len(calls) == 1
    assert result["raw_reconstruction_error"] == "QuotaError; HTTP 429; insufficient_quota; credit_balance_exhausted"
    assert result["clean_reconstruction_error"] == "Skipped after transcription quota failure"
    assert "PRIVATE" not in repr(result)
