from emergency_dispatcher.slm_guidance import GuidanceSeed
from emergency_dispatcher.slm_guidance import SLM_GUIDANCE_SCHEMA_VERSION
from emergency_dispatcher.slm_guidance import build_guidance_record
from emergency_dispatcher.slm_guidance import has_explicit_location_cue
from emergency_dispatcher.slm_guidance import schema_snapshots


def test_build_guidance_record_escalates_to_human_required():
    seed = GuidanceSeed.model_validate(
        {
            "raw_utterance": "My flat is on fire on Skalitzer Strasse 22 and my brother is not breathing.",
            "extracted_fields": {
                "location": "Skalitzer Strasse 22",
                "issue_type": "BUILDING_FIRE",
                "victim_breathing": "no",
                "victim_bleeding": "unknown",
                "victim_conscious": "no",
                "victim_count": 1,
                "caller_safe": "no",
            },
            "address_in_utterance": True,
            "scoring_signals": {
                "unconscious_or_unresponsive": True,
                "indoor_fire_immediate": True,
                "caller_cannot_leave": True,
            },
        }
    )

    record = build_guidance_record(seed, previous_severity=35)

    assert record["severity_score"] == 90
    assert record["severity_delta"] == 55
    assert record["human_recommended"] is True
    assert record["human_required"] is True
    assert record["constraint_mode"] == "holding_pattern_only"
    assert record["extracted_fields"]["priority"] == "CRITICAL"


def test_build_guidance_record_drops_severity_after_correction():
    seed = GuidanceSeed.model_validate(
        {
            "raw_utterance": "Actually it is not a fire, it is steam from the engine and everyone is safe.",
            "extracted_fields": {
                "location": None,
                "issue_type": "SMOKE_INVESTIGATION",
                "victim_breathing": "yes",
                "victim_bleeding": "no",
                "victim_conscious": "yes",
                "victim_count": 1,
                "caller_safe": "yes",
            },
            "address_in_utterance": False,
            "scoring_signals": {
                "caller_calm_coherent": True,
                "situation_stable_static": True,
                "everyone_explicitly_safe": True,
            },
        }
    )

    record = build_guidance_record(seed, previous_severity=72)

    assert record["severity_score"] == 0
    assert record["severity_delta"] == -72
    assert record["human_recommended"] is False
    assert record["human_required"] is False
    assert record["constraint_mode"] == "normal"


def test_address_guard_requires_explicit_location_cue():
    assert has_explicit_location_cue("I'm on the A100 near Ostkreuz.") is True
    assert has_explicit_location_cue("I'm in Berlin and I need help.") is False


def test_schema_snapshots_include_seed_and_record():
    snapshots = schema_snapshots()

    assert SLM_GUIDANCE_SCHEMA_VERSION == "2026-04-25.v1"
    assert "seed" in snapshots
    assert "record" in snapshots
    assert snapshots["seed"]["title"] == "GuidanceSeed"
    assert snapshots["record"]["title"] == "GuidanceRecord"
