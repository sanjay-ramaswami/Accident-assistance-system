from emergency_dispatcher.slm_fact_ledger import FactLedger
from emergency_dispatcher.slm_fact_ledger import build_fact_delta
from emergency_dispatcher.slm_fact_ledger import build_ledger_prompt_context
from emergency_dispatcher.slm_fact_ledger import build_fact_priority_packet
from emergency_dispatcher.slm_fact_ledger import build_pioneer_fact_ledger_record
from emergency_dispatcher.slm_fact_ledger import empty_shared_ledger
from emergency_dispatcher.slm_fact_ledger import has_dispatchable_location_cue
from emergency_dispatcher.slm_fact_ledger import is_immediate_fact_ledger_trigger
from emergency_dispatcher.slm_fact_ledger import is_trivial_caller_turn


def test_has_dispatchable_location_cue_rejects_generic_home_and_accepts_address():
    assert has_dispatchable_location_cue("I am at home in Berlin.") is False
    assert has_dispatchable_location_cue("There is smoke near Mullerstrasse 128 in Berlin.") is True
    assert has_dispatchable_location_cue("Donnerausstraße") is True


def test_fact_delta_and_priority_packet_focus_on_missing_hard_facts():
    prior = FactLedger(
        location_candidate="Delta Campus building",
        location_kind="place_name",
        inside_building="yes",
        issue_cues=["trapped"],
        trapped_status="yes",
    )
    merged = FactLedger(
        location_candidate="Delta Campus building",
        location_kind="place_name",
        inside_building="yes",
        issue_cues=["trapped", "child_present"],
        trapped_status="yes",
        child_present="yes",
    )

    delta = build_fact_delta(prior, merged)
    packet = build_fact_priority_packet(prior, merged)

    assert delta["child_present"] == "yes"
    assert "child_present" in delta["issue_cues"]
    assert packet["priority_flags"]["next_question_field"] == "victim_count"
    assert packet["priority_flags"]["next_question_goal"] == "confirm how many people are involved"
    assert packet["priority_flags"]["location_search_allowed"] is False
    assert packet["priority_flags"]["best_effort_location_note"] is None


def test_priority_packet_keeps_best_effort_location_note_without_enabling_search():
    prior = FactLedger()
    merged = FactLedger(
        location_kind="none",
        location_note="behind the red brick church",
        address_in_utterance=False,
    )

    packet = build_fact_priority_packet(prior, merged)

    assert packet["priority_flags"]["location_search_allowed"] is False
    assert packet["priority_flags"]["best_effort_location_note"] == "behind the red brick church"


def test_pioneer_record_contains_user_input_and_merged_facts():
    record = build_pioneer_fact_ledger_record(
        {
            "caller_turn": "With my daughter.",
            "prior_facts": FactLedger().model_dump(),
            "merged_facts": FactLedger(
                issue_cues=["child_present"],
                child_present="yes",
            ).model_dump(),
        }
    )

    assert record["messages"][1]["role"] == "user"
    assert "caller_turn" in record["messages"][1]["content"]
    assert record["messages"][2]["role"] == "assistant"
    assert "merged_facts" in record["messages"][2]["content"]


def test_shared_ledger_prompt_context_uses_confirmed_fields_and_gate():
    ledger = empty_shared_ledger()
    ledger["version"] = 4
    ledger["hard_facts"]["issue_cues"] = ["collision", "smoke"]
    ledger["hard_facts"]["sub_location"] = "fourth floor"
    ledger["hard_facts"]["victim_count_confirmed"] = "unknown"
    ledger["priority"]["missing_fields"] = ["victim_count_confirmed", "bleeding_status"]
    ledger["priority"]["next_question_field"] = "victim_count_confirmed"
    ledger["priority"]["next_question_goal"] = "confirm how many people are involved"
    ledger["priority"]["location_followup_kind"] = "confirm_candidate"
    ledger["priority"]["location_followup_prompt"] = "You said Delta Campus. Is that correct? Yes or no."
    ledger["priority"]["location_attempts"] = 2
    ledger["location_gate"]["search_allowed"] = False
    ledger["location_gate"]["candidate_text"] = "Delta Campus"
    ledger["location_gate"]["needs_confirmation"] = True

    packet = build_ledger_prompt_context(ledger)

    assert packet["version"] == 4
    assert packet["known"]["location_candidate"] == "Delta Campus"
    assert packet["missing_fields"] == ["victim_count_confirmed", "bleeding_status"]
    assert packet["location_search_allowed"] is False
    assert packet["location_followup_kind"] == "confirm_candidate"
    assert packet["candidate_needs_confirmation"] is True


def test_fact_ledger_trigger_rules_skip_bare_yes_no_but_run_for_corrections():
    assert is_trivial_caller_turn("yes")
    assert is_trivial_caller_turn("no")
    assert is_immediate_fact_ledger_trigger("yes") is False
    assert is_immediate_fact_ledger_trigger("no") is False
    assert is_immediate_fact_ledger_trigger("no bleeding") is True
    assert is_immediate_fact_ledger_trigger("actually it's Donau Straße") is True
    assert is_immediate_fact_ledger_trigger("there is smoke in the kitchen") is True
