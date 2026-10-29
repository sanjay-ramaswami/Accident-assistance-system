from emergency_dispatcher.gradbot_adapter import build_session_config


def test_session_config_starts_the_call_and_stays_patient():
    config = build_session_config()

    assert config.assistant_speaks_first is True
    assert config.flush_duration_s >= 1.6
    assert config.silence_timeout_s >= 7.0
    assert len(config.instructions) < 4000
    assert "This demo cannot call emergency services" not in config.instructions
    assert "Emergency dispatch. Tell me what happened and where you are." in config.instructions
    assert "Do not interrupt the caller's opening sentence" in config.instructions
    assert "wait for a clear pause before speaking" in config.instructions
    assert "use `checklist_by_incident` before deciding the next hard-fact question" in config.instructions
    assert "use `resolve_location_note` for landmarks, entrances, floors, rooms, and relative clues" in config.instructions
    assert "normalize_place_name" not in config.instructions


def test_session_config_includes_bounded_curated_bystander_guidance():
    config = build_session_config(first_aid_context={
        "scenario": "Road accident scene safety",
        "immediate_actions": ["Stay clear of traffic and smoke.", "Ask someone to call local emergency services."],
        "actions_to_avoid": ["Do not move injured people unless immediate danger."],
    })

    assert "Current bystander-safety guidance from the local curated first-aid rules" in config.instructions
    assert "Stay clear of traffic and smoke." in config.instructions
    assert "ask one essential question" in config.instructions
    assert "if immediate danger requires real-world help" in config.instructions
    dispatch_config = build_session_config(dispatch_context={
        "serious_emergency": True, "human_monitoring": False,
        "human_review_recommended": True, "services_summary": "Ambulance assigned",
        "announcement_line": "A simulated ambulance has been assigned.",
    })
    assert "Never invent a unit ID, ETA, location, or operator" in dispatch_config.instructions


def test_session_config_gets_more_patient_after_interruptions():
    config = build_session_config(interrupt_count=2, interruption_recovery=True)

    assert config.assistant_speaks_first is True
    assert config.flush_duration_s > 1.65
    assert config.silence_timeout_s > 7.0
    assert "Adaptive turn-taking policy:" in config.instructions
    assert "Please repeat that last part." in config.instructions
    assert "Immediate recovery instruction:" in config.instructions


def test_session_config_forces_binary_medical_confirmation_when_ambiguous():
    config = build_session_config()

    assert "Is she breathing right now? Yes or no." in config.instructions
    assert "Is there heavy bleeding? Yes or no." in config.instructions
    assert "do not keep reconfirming life-status" in config.instructions


def test_session_config_embeds_triage_context_for_llm_guidance():
    config = build_session_config(
        triage_context={
            "version": 2,
            "selected_engine": "heuristic",
            "runtime_provider": "heuristic",
            "raw_utterance": "My partner is bleeding near Delta Campus.",
            "latest_triage_delta": {
                "location": "Delta Campus",
                "issue_type": "VEHICLE_ACCIDENT_WITH_INJURY",
                "priority": "CRITICAL",
                "victim_breathing": "unknown",
                "victim_bleeding": "yes",
                "victim_conscious": "unknown",
                "victim_count": "unknown",
                "caller_safe": "unknown",
                "escalate_to_human": True,
                "escalation_reason": "priority_critical",
                "severity_score": 90,
                "severity_delta": 55,
                "severity_drivers": ["heavy_bleeding"],
                "human_recommended": True,
                "human_required": True,
                "constraint_mode": "holding_pattern_only",
                "address_in_utterance": True,
            },
            "merged_triage_state": {
                "location": "Delta Campus",
                "issue_type": "VEHICLE_ACCIDENT_WITH_INJURY",
                "priority": "CRITICAL",
                "victim_breathing": "unknown",
                "victim_bleeding": "yes",
                "victim_conscious": "unknown",
                "victim_count": "unknown",
                "caller_safe": "unknown",
                "escalate_to_human": True,
                "escalation_reason": "priority_critical",
                "severity_score": 90,
                "severity_delta": 55,
                "severity_drivers": ["heavy_bleeding"],
                "human_recommended": True,
                "human_required": True,
                "constraint_mode": "holding_pattern_only",
                "address_in_utterance": True,
            },
            "missing_or_unknown_fields": ["victim_breathing", "victim_conscious"],
            "rule_overrides": [],
            "handoff_packet": {"status": "ready_for_human_handoff"},
            "dispatchable": True,
            "autonomy_allowed": False,
            "gliner_entities": [{"label": "LOCATION", "text": "Delta Campus", "confidence": 0.98}],
            "stt_rescue": {
                "corrected_transcript": "My partner is bleeding near Delta Campus.",
                "ask_for_repeat": True,
                "location_needs_spelling": True,
            },
        }
    )

    assert "Live triage packet:" in config.instructions
    assert "ask the caller to spell the key word" in config.instructions
    assert "respect constraint_mode exactly" in config.instructions
    assert "never call validate_address or lookup_address unless location_search_allowed is true" in config.instructions
    assert "prefer one relevant deterministic tool call before a follow-up" in config.instructions
    assert "do not ask again for a fact already resolved" in config.instructions
    assert "next_question_field" in config.instructions
    assert "location_search_allowed" in config.instructions
    assert "rescue_text" in config.instructions


def test_session_config_embeds_shared_ledger_context_for_slm_mode():
    config = build_session_config(
        live_dispatch_mode="slm",
        ledger_context={
            "version": 3,
            "known": {
                "issue_cues": ["collision", "smoke"],
                "location_candidate": "Delta Campus",
                "sub_location": "fourth floor",
                "victim_count_confirmed": "unknown",
            },
            "missing_fields": ["victim_count_confirmed"],
            "next_question_field": "victim_count_confirmed",
            "next_question_goal": "confirm how many people are involved",
            "question_style": "short_open",
            "location_search_allowed": False,
            "location_followup_kind": "confirm_candidate",
            "location_followup_prompt": "You said Delta Campus. Is that correct? Yes or no.",
            "location_dead_end": False,
            "location_attempts": 2,
            "location_lock_active": True,
            "candidate_needs_confirmation": True,
            "blocked_actions": ["do_not_repeat_resolved_facts", "do_not_geocode_vague_location"],
        },
    )

    tool_names = [tool.name for tool in config.tools]

    assert "Shared fact ledger packet:" in config.instructions
    assert "update_soft_ledger only for soft guesses" in config.instructions
    assert "location_search_allowed" in config.instructions
    assert "location_followup_prompt" in config.instructions
    assert "location_lock_active" in config.instructions
    assert "the location ladder is" in config.instructions
    assert "location is the master priority" in config.instructions
    assert "the master order is: understand the emergency, confirm the caller is stable enough to continue, resolve location" in config.instructions
    assert "if the caller asks you to send help before location is usable" in config.instructions
    assert "if the caller asks where help is being sent while location is still unresolved" in config.instructions
    assert "do not say help is on the way" in config.instructions
    assert "update_soft_ledger" in tool_names


def test_session_config_supports_human_takeover_and_handoff_prompt():
    takeover_config = build_session_config(human_takeover_active=True)
    assert "a human operator has taken over the live call" in takeover_config.instructions
    assert "remain silent until takeover is cleared" in takeover_config.instructions

    handoff_config = build_session_config(handoff_recovery_prompt="What is your status now?")
    assert "takeover has been cleared" in handoff_config.instructions
    assert "What is your status now?" in handoff_config.instructions
