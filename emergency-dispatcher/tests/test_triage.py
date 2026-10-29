import json
from types import SimpleNamespace

from emergency_dispatcher import triage
from emergency_dispatcher.settings import get_settings
from emergency_dispatcher.triage import HeuristicTriageEngine
from emergency_dispatcher.triage import SLMTriageEngine
from emergency_dispatcher.triage import TriageResult
from emergency_dispatcher.triage import analyze_session_triage
from emergency_dispatcher.triage import apply_rule_overrides
from emergency_dispatcher.triage import enforce_rule_overrides
from emergency_dispatcher.triage import merge_triage_state
from emergency_dispatcher.triage import replay_triage_session
from emergency_dispatcher.triage import triage_session_patch


def _heuristic_settings():
    return get_settings().model_copy(update={"triage_engine": "heuristic"})


def test_merge_triage_state_overwrites_unknown_but_preserves_partial_location():
    current = TriageResult(
        location="Delta Campus",
        issue_type=None,
        priority=None,
        victim_breathing="unknown",
        caller_safe="unknown",
    )
    delta = TriageResult(
        location=None,
        issue_type="VEHICLE_ACCIDENT_WITH_INJURY",
        priority="CRITICAL",
        victim_breathing="yes",
        caller_safe="no",
    )

    merged = merge_triage_state(current, delta)

    assert merged.location == "Delta Campus"
    assert merged.issue_type == "VEHICLE_ACCIDENT_WITH_INJURY"
    assert merged.priority == "CRITICAL"
    assert merged.victim_breathing == "yes"
    assert merged.caller_safe == "no"


def test_rule_overrides_force_escalation():
    state = TriageResult(
        location="Delta Campus",
        issue_type="VEHICLE_ACCIDENT",
        priority="LOW",
        caller_safe="yes",
    )

    overrides = apply_rule_overrides("There is shooting and people are shot near Delta Campus", state)
    updated = enforce_rule_overrides(state, overrides)

    assert overrides
    assert updated.escalate_to_human is True
    assert updated.priority == "LOW" or updated.priority == "CRITICAL"
    assert updated.escalation_reason is not None


def test_heuristic_triage_marks_bleeding_and_unknown_breathing():
    engine = HeuristicTriageEngine()
    result = engine.analyze_utterance(
        "My partner is bleeding badly. I do not know if she is breathing or not.",
        TriageResult(),
    )

    assert result.victim_bleeding == "yes"
    assert result.victim_breathing == "unknown"
    assert result.escalate_to_human is True
    assert engine.last_artifacts["model_id"] == "heuristic-demo-v1"
    assert engine.last_artifacts["demo_only"] is True
    assert "victim_breathing" in engine.last_artifacts["uncertainty"]["unknown_fields"]
    assert engine.last_artifacts["uncertainty"]["calibrated_confidence"] is None


def test_openai_triage_failure_is_reported_as_a_demo_rule_fallback(monkeypatch):
    class FailingCompletions:
        def create(self, **_kwargs):
            raise RuntimeError("sanitized test failure")

    monkeypatch.setattr(
        triage,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions())),
    )
    settings = get_settings().model_copy(
        update={"openai_api_key": "test-key", "openai_model": "test-model"}
    )
    engine = triage.LLMTriageEngine(settings)

    result = engine.analyze_utterance("A person is bleeding and I cannot tell if they are breathing.", TriageResult())

    assert result.victim_bleeding == "yes"
    assert result.victim_breathing == "unknown"
    assert engine.selected_engine == "llm"
    assert engine.runtime_provider == "heuristic_fallback"
    assert engine.last_artifacts["demo_only"] is True
    assert engine.last_artifacts["inference_status"] == "failed_demo_fallback"
    assert engine.last_artifacts["fallback_error_type"] == "RuntimeError"


def test_openai_triage_success_records_resolved_model_id_without_confidence_claim(monkeypatch):
    response = SimpleNamespace(
        model="test-model-version",
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"issue_type": "VEHICLE_BREAKDOWN"})))],
    )
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    monkeypatch.setattr(
        triage,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    settings = get_settings().model_copy(
        update={"openai_api_key": "test-key", "openai_model": "test-model"}
    )
    engine = triage.LLMTriageEngine(settings)

    engine.analyze_utterance("The car has a flat tire.", TriageResult())

    assert engine.runtime_provider == "openai"
    assert engine.last_artifacts["model_id"] == "test-model-version"
    assert engine.last_artifacts["inference_status"] == "success"
    assert engine.last_artifacts["demo_only"] is False
    assert engine.last_artifacts["uncertainty"]["calibrated_confidence"] is None


def test_heuristic_breathing_stays_unknown_when_caller_cannot_tell():
    engine = HeuristicTriageEngine()

    result = engine.analyze_utterance(
        "I cannot tell if they are breathing.",
        TriageResult(),
    )

    assert result.victim_breathing == "unknown"


def test_heuristic_breathing_stays_unknown_when_absence_is_uncertain():
    engine = HeuristicTriageEngine()

    result = engine.analyze_utterance(
        "I am not sure if she is not breathing.",
        TriageResult(),
    )

    assert result.victim_breathing == "unknown"


def test_analyze_session_triage_returns_prompt_context():
    session = {
        "session_id": "test-session",
        "client": {
            "transcripts": {
                "user": [
                    {"text": "I am stuck at Delta Campus and my partner is bleeding heavily."},
                ],
                "agent": [],
            }
        },
        "triage_turns": [],
        "merged_triage_state": {},
        "triage_meta": {},
    }

    update = analyze_session_triage(session, _heuristic_settings())

    assert update is not None
    assert update.prompt_context["merged_triage_state"]["location"] == "Delta Campus"
    assert update.prompt_context["dispatchable"] is True
    assert update.prompt_context["conversation_focus"]["next_question_field"] == "caller_safe"
    assert update.prompt_context["conversation_focus"]["location_search_allowed"] is False
    assert update.model_artifacts["demo_only"] is True
    persisted = triage_session_patch(session, update)
    assert persisted["triage_turns"][0]["model_artifacts"]["model_id"] == "heuristic-demo-v1"
    assert persisted["triage_meta"]["runtime_provider"] == "heuristic"


def test_replay_triage_session_runs_deterministically():
    session = {
        "session_id": "replay-session",
        "client": {
            "transcripts": {
                "user": [
                    {"text": "I am stuck at Delta Campus."},
                    {"text": "My leg is bleeding heavily."},
                ]
            }
        },
    }

    payload = replay_triage_session(session, _heuristic_settings())

    assert payload["session_id"] == "replay-session"
    assert len(payload["turns"]) == 2
    assert payload["final_state"]["location"] == "Delta Campus"


def test_slm_triage_engine_calls_decoder_and_gliner(monkeypatch):
    def fake_post_json(*, url, api_key, payload, timeout_s):
        assert url.endswith("/chat/completions")
        assert api_key == "pioneer-token"
        assert timeout_s == 5.0
        if payload["model"] == "decoder-model":
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "extracted_fields": {
                                        "location": None,
                                        "issue_type": "BUILDING_FIRE",
                                        "priority": "CRITICAL",
                                        "victim_breathing": "unknown",
                                        "victim_bleeding": "unknown",
                                        "victim_conscious": "unknown",
                                        "victim_count": "unknown",
                                        "caller_safe": "no",
                                    },
                                    "severity_score": 92,
                                    "severity_delta": 50,
                                    "severity_drivers": ["indoor_fire_immediate", "caller_cannot_leave"],
                                    "human_recommended": True,
                                    "human_required": True,
                                    "agent_should_continue": True,
                                    "constraint_mode": "holding_pattern_only",
                                    "address_in_utterance": False,
                                }
                            )
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {"text": "third floor", "label": "LOCATION", "confidence": 0.97},
                                    {"text": "one", "label": "VICTIM_COUNT", "confidence": 0.95},
                                ]
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(triage, "_post_json", fake_post_json)

    settings = SimpleNamespace(
        triage_api_key="pioneer-token",
        openai_api_key=None,
        triage_base_url="https://api.pioneer.ai/v1",
        triage_decoder_model="decoder-model",
        triage_gliner_model="gliner-model",
        triage_request_timeout_s=5.0,
    )
    engine = SLMTriageEngine(settings)

    result = engine.analyze_utterance(
        "My apartment kitchen is filling with smoke. My son is still inside on the third floor and I can't get back in.",
        TriageResult(severity_score=42),
    )

    assert result.issue_type == "BUILDING_FIRE"
    assert result.human_required is True
    assert result.constraint_mode == "holding_pattern_only"
    assert result.location == "third floor"
    assert result.victim_count == 1
    assert engine.last_artifacts["gliner_entities"][0]["label"] == "LOCATION"
    assert engine.last_artifacts["model_id"] == "decoder-model"
    assert engine.last_artifacts["model_kind"] == "remote_slm"
    assert engine.last_artifacts["demo_only"] is False
    assert engine.last_artifacts["uncertainty"]["calibrated_confidence"] is None


def test_slm_triage_engine_skips_gliner_when_live_disabled(monkeypatch):
    calls = []

    def fake_post_json(*, url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "extracted_fields": {
                                    "location": None,
                                    "issue_type": "ACTIVE_THREAT_OR_WEAPON",
                                    "priority": "HIGH",
                                    "victim_breathing": "unknown",
                                    "victim_bleeding": "unknown",
                                    "victim_conscious": "unknown",
                                    "victim_count": "unknown",
                                    "caller_safe": "no",
                                },
                                "severity_score": 86,
                                "severity_delta": 70,
                                "severity_drivers": ["weapon_shooting_explosion"],
                                "human_recommended": True,
                                "human_required": True,
                                "agent_should_continue": True,
                                "constraint_mode": "holding_pattern_only",
                                "address_in_utterance": False,
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(triage, "_post_json", fake_post_json)

    settings = SimpleNamespace(
        triage_api_key="pioneer-token",
        openai_api_key=None,
        triage_base_url="https://api.pioneer.ai/v1",
        triage_decoder_model="decoder-model",
        triage_gliner_model="gliner-model",
        triage_gliner_live_enabled=False,
        triage_gliner_timeout_s=2.0,
        triage_request_timeout_s=5.0,
    )
    engine = SLMTriageEngine(settings)

    result = engine.analyze_utterance(
        "There is a gunman who is trying to shoot at me.",
        TriageResult(severity_score=16),
    )

    assert result.issue_type == "ACTIVE_THREAT_OR_WEAPON"
    assert calls == ["decoder-model"]
    assert engine.last_artifacts["gliner_live_enabled"] is False
    assert engine.last_artifacts["gliner_entities"] == []
