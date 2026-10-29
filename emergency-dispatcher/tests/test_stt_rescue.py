import asyncio
from types import SimpleNamespace

from emergency_dispatcher.stt_rescue import RescueCandidate
from emergency_dispatcher.stt_rescue import STTRescueEvent
from emergency_dispatcher.stt_rescue import _select_rescue_candidate
from emergency_dispatcher.stt_rescue import append_live_audio_window
from emergency_dispatcher.stt_rescue import clear_live_audio_window
from emergency_dispatcher.stt_rescue import maybe_run_stt_rescue
from emergency_dispatcher.stt_rescue import stt_rescue_session_patch
from emergency_dispatcher.stt_rescue import detect_suspicious_transcript


def test_detect_suspicious_transcript_flags_unexpected_child_phrase():
    reasons = detect_suspicious_transcript(
        "I am in a car and I am drowning. Delta campus building. I'm trying to be the child.",
        {"location": "Delta Campus", "issue_type": "VEHICLE_ACCIDENT_WITH_INJURY"},
    )

    assert "unlikely_phrase_be_the_child" in reasons
    assert "unexpected_child_mention" in reasons


def test_detect_suspicious_transcript_flags_location_gibberish():
    reasons = detect_suspicious_transcript(
        "I am approximately between aircar strata aircar in Berlin aircar bet it.",
        {"location": "Berlin", "issue_type": "FLAT_TYRE"},
    )

    assert "location_gibberish" in reasons
    assert "unstable_location_phrase" in reasons
    assert "repeated_rare_token" in reasons


def test_select_rescue_candidate_prefers_stronger_backup_transcript():
    provider, corrected, candidates, ask_for_repeat = _select_rescue_candidate(
        "I'm trying to be the child.",
        [
            RescueCandidate(provider="deepgram", transcript="I'm trying to reach the child.", usable=True),
            RescueCandidate(provider="soniox", transcript="I'm trying to leave the car.", usable=True),
        ],
        suspicious_reasons=["unlikely_phrase_be_the_child"],
        current_state={"location": "Delta Campus", "issue_type": "VEHICLE_ACCIDENT_WITH_INJURY"},
    )

    assert provider == "deepgram"
    assert corrected == "I'm trying to reach the child."
    assert any(item.provider == "deepgram" for item in candidates)
    assert ask_for_repeat is True


def test_select_rescue_candidate_can_force_preferred_provider():
    provider, corrected, candidates, ask_for_repeat = _select_rescue_candidate(
        "Berlin Mitte near a car crash.",
        [
            RescueCandidate(provider="deepgram", transcript="Berlin Mitte near a car crash.", usable=True),
            RescueCandidate(provider="soniox", transcript="Barely in middle near a car crash.", usable=True),
        ],
        suspicious_reasons=[],
        current_state={"location": "Berlin", "issue_type": "VEHICLE_ACCIDENT_WITH_INJURY"},
        preferred_provider="deepgram",
        force_preferred=True,
    )

    assert provider == "deepgram"
    assert corrected == "Berlin Mitte near a car crash."
    assert ask_for_repeat is False
    assert len(candidates) == 2


def test_stt_rescue_session_patch_switches_to_deepgram_on_continuous_compare_win():
    base_session = {
        "stt_rescue_events": [],
        "stt_rescue_overrides": {},
        "stt_rescue_meta": {
            "active_provider": "gradium",
            "provider_locked": False,
            "provider_stats": {},
            "switch_events": [],
        },
    }
    event = STTRescueEvent(
        user_index=0,
        suspicious_reasons=["location_gibberish"],
        primary_transcript="Berlin Mitte near a fire.",
        selected_provider="deepgram",
        selected_transcript="Berlin Mitte near a fire.",
        corrected_transcript="Berlin Mitte near a fire.",
        ask_for_repeat=True,
        location_needs_spelling=False,
        clip_source="clean",
        clip_path=None,
        primary_score=1.0,
        continuous_compare=True,
        shadow_active=False,
        shadow_elapsed_s=10.0,
        active_provider="gradium",
        provider_scores={"gradium": 1.0, "deepgram": 3.5},
        candidates=[],
    )
    patch = stt_rescue_session_patch(base_session, event)
    assert patch["stt_rescue_meta"]["active_provider"] == "deepgram"
    assert patch["stt_rescue_meta"]["provider_locked"] is True
    assert patch["stt_rescue_meta"]["continuous_compare"] is True
    assert len(patch["stt_rescue_meta"]["switch_events"]) == 1
    assert patch["stt_rescue_meta"]["switch_events"][0]["reason"] == "continuous_backend_compare_win"


def test_maybe_run_stt_rescue_skips_normal_turn_when_continuous_compare_disabled():
    session = {
        "client": {
            "transcripts": {
                "user": [
                    {"text": "I'm close to Amelia Strasser."},
                ]
            }
        },
        "merged_triage_state": {"location": None, "issue_type": "ACTIVE_THREAT_OR_WEAPON"},
        "stt_rescue_meta": {
            "active_provider": "deepgram",
            "provider_locked": True,
        },
    }
    settings = SimpleNamespace(
        deepgram_api_key="deepgram-key",
        soniox_api_key=None,
        stt_rescue_continuous_compare=False,
    )

    event = asyncio.run(
        maybe_run_stt_rescue(
            session_id="test-session",
            session=session,
            settings=settings,
        )
    )

    assert event is None


def test_maybe_run_stt_rescue_returns_shadow_compare_event_when_enabled(monkeypatch):
    session_id = "shadow-compare-session"
    clear_live_audio_window(session_id)
    append_live_audio_window(session_id, b"\x00\x00" * 320, b"\x00\x00" * 320)

    async def fake_deepgram(wav_bytes, settings):
        return RescueCandidate(
            provider="deepgram",
            transcript="Can you hear me?",
            usable=True,
            latency_s=0.12,
        )

    monkeypatch.setattr("emergency_dispatcher.stt_rescue._deepgram_transcribe", fake_deepgram)

    session = {
        "client": {
            "transcripts": {
                "user": [
                    {"text": "Can you hear me?"},
                ]
            }
        },
        "merged_triage_state": {"location": None, "issue_type": "MEDICAL_EMERGENCY"},
        "stt_rescue_meta": {
            "active_provider": "gradium",
            "provider_locked": False,
        },
    }
    settings = SimpleNamespace(
        deepgram_api_key="deepgram-key",
        soniox_api_key=None,
        stt_rescue_continuous_compare=True,
    )

    try:
        event = asyncio.run(
            maybe_run_stt_rescue(
                session_id=session_id,
                session=session,
                settings=settings,
            )
        )
    finally:
        clear_live_audio_window(session_id)

    assert event is not None
    assert event.continuous_compare is True
    assert event.shadow_active is True
    assert event.selected_provider in {"gradium", "deepgram"}
    assert event.provider_scores["deepgram"] >= event.provider_scores["gradium"]
