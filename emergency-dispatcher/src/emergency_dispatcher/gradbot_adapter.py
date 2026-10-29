from __future__ import annotations

from pathlib import Path
from typing import Any

import gradbot

from .dispatch_tools import build_gradbot_tool_defs

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "dispatcher_system_prompt.md"


def load_dispatch_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _adaptive_turn_settings(interrupt_count: int) -> tuple[float, float]:
    active_interrupts = max(0, interrupt_count)
    flush_duration_s = min(1.65 + (0.28 * active_interrupts), 2.45)
    silence_timeout_s = min(7.0 + (1.1 * active_interrupts), 11.5)
    return flush_duration_s, silence_timeout_s


def _format_triage_context(triage_context: dict[str, Any] | None) -> list[str]:
    if not triage_context:
        return []
    merged = triage_context.get("merged_triage_state") or {}
    latest_delta = triage_context.get("latest_triage_delta") or {}
    stt_rescue = triage_context.get("stt_rescue")
    conversation_focus = triage_context.get("conversation_focus") or {}
    constraint_mode = merged.get("constraint_mode") or latest_delta.get("constraint_mode")
    lines = [
        "",
        "Live triage packet:",
        f"- issue_type: {merged.get('issue_type')}",
        f"- caller_safe: {merged.get('caller_safe')}",
        f"- location: {merged.get('location')}",
        f"- victim_count: {merged.get('victim_count')}",
        f"- victim_breathing: {merged.get('victim_breathing')}",
        f"- victim_bleeding: {merged.get('victim_bleeding')}",
        f"- victim_conscious: {merged.get('victim_conscious')}",
        f"- severity_score: {merged.get('severity_score')}",
        f"- human_required: {merged.get('human_required')}",
        f"- constraint_mode: {constraint_mode}",
        f"- address_in_utterance: {latest_delta.get('address_in_utterance')}",
        f"- next_question_field: {conversation_focus.get('next_question_field')}",
        f"- next_question_goal: {conversation_focus.get('next_question_goal')}",
        f"- location_search_allowed: {conversation_focus.get('location_search_allowed')}",
    ]
    if stt_rescue:
        lines.append(f"- rescue_text: {stt_rescue.get('corrected_transcript')}")
        lines.append(f"- rescue_repeat: {stt_rescue.get('ask_for_repeat')}")
        lines.append(f"- rescue_location_spelling: {stt_rescue.get('location_needs_spelling')}")
        lines.extend(
        [
            "",
            "Runtime rules:",
            "- use the triage packet as the source of truth unless the caller clearly corrects it",
            "- be brief, calm, and natural",
            "- ask one question only",
            "- if next_question_field is set, ask only for that field",
            "- do not ask a lower-priority question while a higher-priority field is still missing",
            "- if you ask about the caller, use second person like `Are you able to breathe properly?` or `Are you awake right now?`",
            "- if you ask about another person, use third person like `Is she breathing right now?` or `Is he awake right now?`",
            "- never ask `Are you conscious?` and never produce broken starts like `Is Are you`",
            "- prefer one relevant deterministic tool call before a follow-up when a hard fact just changed",
            "- use resolve_location_note first for landmarks, entrances, floors, rooms, or relative clues",
            "- use checklist_by_incident before deciding the next hard-fact question",
            "- use validate_address or lookup_address only after a searchable location clue lands",
            "- use nearby_context only after a stable place or address lands",
            "- use build_handoff_brief when facts stabilize or before ticket creation",
            "- use at most one tool before each spoken follow-up, except location flow may use address + nearby_context back to back",
            "- do not ask again for a fact already resolved by the triage packet or latest tool output",
            "- do not call a tool if it would only restate known information",
            "- before a slow location lookup or validation, say one short holding line so the caller is not left in silence",
            "- approved holding examples: `I'm locating and dispatching the nearest team now. Give me a moment.`, `Okay, I'm checking that location now.`",
            "- use only one holding line for the same wait",
            "- never call validate_address or lookup_address unless location_search_allowed is true",
            "- never geocode vague phrases like `middle of the street`, `inside the building`, `here`, or `at home`",
            "- if a road or landmark name is used in multiple cities (including `Marine Drive`) and no city is confirmed, ask which city or a nearby landmark before geocoding",
            "- when the caller confirms a city, geocode the place together with that city; describe any city-qualified result as approximate until the incident point is confirmed",
            "- if indoors, ask for floor, unit, entrance, stairwell, or building name before outside landmarks",
            "- respect constraint_mode exactly",
        ]
    )
    if constraint_mode == "single_question_only":
        lines.extend(
            [
                "",
                "Mode:",
                "- ask exactly one short direct question",
            ]
        )
    elif constraint_mode == "pre_arrival_priority":
        lines.extend(
            [
                "",
                "Mode:",
                "- stop broad information gathering",
                "- give immediate pre-arrival help or one binary confirmation only if it changes safety actions",
            ]
        )
    elif constraint_mode == "holding_pattern_only":
        lines.extend(
            [
                "",
                "Mode:",
                "- do not ask exploratory questions",
                "- keep the caller on the line",
                "- give at most one short safety instruction",
            ]
        )
    if stt_rescue:
        lines.extend(
            [
                "",
                "STT rescue guidance:",
                "- use rescue_text as the best current reading of the last caller turn",
            ]
        )
        if stt_rescue.get("ask_for_repeat"):
            lines.extend(
                [
                    "- start by saying you did not quite catch the last part",
                    "- ask the caller to repeat or confirm the unstable detail before moving on",
                ]
            )
        if stt_rescue.get("location_needs_spelling"):
            lines.extend(
                [
                    "- the unstable detail is the location",
                    "- ask for one road, landmark, or junction slowly",
                    "- ask the caller to spell the key word",
                ]
            )
    return lines


def _format_dispatch_context(dispatch_context: dict[str, Any] | None) -> list[str]:
    if not dispatch_context:
        return []
    lines = [
        "",
        "Live dispatch operations packet:",
        f"- simulation_only: true",
        f"- serious_emergency: {dispatch_context.get('serious_emergency')}",
        f"- human_review_recommended: {dispatch_context.get('human_review_recommended')}",
        f"- actual_human_monitoring: {dispatch_context.get('human_monitoring')}",
        f"- services_summary: {dispatch_context.get('services_summary')}",
    ]
    announcement_line = dispatch_context.get("announcement_line")
    if announcement_line:
        lines.append(f"- announcement_line: {announcement_line}")
    lines.extend(
        [
            "",
            "Dispatch operations rules:",
            "- if announcement_line is present, weave it into your next reply once, naturally and briefly",
            "- describe assignment and movement as simulated; do not imply real services were contacted",
            "- say assigned, en route, or arrived only when the corresponding persisted service status says so",
            "- never invent an ambulance identifier, operator, location, or ETA; omit absent details",
            "- a human review recommendation is not a monitoring or takeover event",
            "- do not repeat the same status line after it has been delivered once",
            "- keep operations updates to one short sentence, then return to the caller's immediate need",
        ]
    )
    return lines


def _format_takeover_context(
    *,
    human_takeover_active: bool = False,
    handoff_recovery_prompt: str | None = None,
) -> list[str]:
    if not human_takeover_active and not handoff_recovery_prompt:
        return []
    lines = [
        "",
        "Human takeover packet:",
        f"- human_takeover_active: {human_takeover_active}",
        f"- handoff_recovery_prompt: {handoff_recovery_prompt}",
        "",
        "Human takeover rules:",
    ]
    if human_takeover_active:
        lines.extend(
            [
                "- a human operator has taken over the live call",
                "- stop speaking immediately",
                "- do not ask questions, do not give instructions, and do not produce filler speech while takeover is active",
                "- remain silent until takeover is cleared",
            ]
        )
    else:
        lines.extend(
            [
                "- takeover has been cleared",
                "- when handoff_recovery_prompt is present, your next turn must say that exact question and nothing else before resuming normal flow",
            ]
        )
    return lines


def _format_first_aid_context(first_aid_context: dict[str, Any] | None) -> list[str]:
    if not first_aid_context:
        return []
    scenario = first_aid_context.get("scenario")
    if not scenario:
        return []
    actions = first_aid_context.get("immediate_actions") or []
    avoid = first_aid_context.get("actions_to_avoid") or []
    conflicts = first_aid_context.get("conflicting_facts") or []
    lines = [
        "",
        "Current bystander-safety guidance from the local curated first-aid rules:",
        f"- scenario: {str(scenario)[:80]}",
    ]
    lines.extend(f"- safe_action: {str(item)[:220]}" for item in actions[:2])
    lines.extend(f"- avoid: {str(item)[:180]}" for item in avoid[:2])
    if conflicts:
        lines.append(f"- unresolved_conflicts: {conflicts}")
        lines.append("- do not treat either conflicting report as settled; ask one concise clarification while preserving immediate scene safety")
    lines.extend([
        "- give one concise current safety step when relevant, then ask one essential question; do not repeat it, diagnose, prescribe, or claim an unpersisted dispatch state",
        "- if immediate danger requires real-world help, advise contacting local emergency services and following the real call-taker",
    ])
    return lines


def _format_ledger_context(ledger_context: dict[str, Any] | None) -> list[str]:
    if not ledger_context:
        return []
    known = ledger_context.get("known") or {}
    location_followup_prompt = ledger_context.get("location_followup_prompt")
    must_say_next = ledger_context.get("must_say_next")
    lines = [
        "",
        "Shared fact ledger packet:",
        f"- issue_cues: {known.get('issue_cues')}",
        f"- location_candidate: {known.get('location_candidate')}",
        f"- sub_location: {known.get('sub_location')}",
        f"- victim_count_confirmed: {known.get('victim_count_confirmed')}",
        f"- missing_fields: {ledger_context.get('missing_fields')}",
        f"- next_question_field: {ledger_context.get('next_question_field')}",
        f"- next_question_goal: {ledger_context.get('next_question_goal')}",
        f"- question_style: {ledger_context.get('question_style')}",
        f"- location_search_allowed: {ledger_context.get('location_search_allowed')}",
        f"- location_followup_kind: {ledger_context.get('location_followup_kind')}",
        f"- location_dead_end: {ledger_context.get('location_dead_end')}",
        f"- location_attempts: {ledger_context.get('location_attempts')}",
        f"- location_lock_active: {ledger_context.get('location_lock_active')}",
        f"- candidate_needs_confirmation: {ledger_context.get('candidate_needs_confirmation')}",
        f"- blocked_actions: {ledger_context.get('blocked_actions')}",
    ]
    if location_followup_prompt:
        lines.append(f"- location_followup_prompt: {location_followup_prompt}")
    if must_say_next:
        lines.append(f"- must_say_next: {must_say_next}")
    lines.extend(
        [
            "",
            "Shared ledger rules:",
            "- treat the shared fact ledger as the source of truth for confirmed hard facts",
            "- in SLM + LLM mode, the ledger is the master control plane for question priority",
            "- the master order is: understand the emergency, confirm the caller is stable enough to continue, resolve location, then gather secondary details",
            "- if the emergency type is already clear and the caller sounds stable enough to continue, unresolved location outranks people count, injury detail, and other secondary facts",
            "- do not restate or re-ask any hard fact already present in the ledger",
            "- use the next_question_goal as your highest-priority conversational target",
            "- if question_style is yes_no, ask one binary question only",
            "- if question_style is short_open, ask one short natural question only",
            "- if location_followup_prompt is present and location_search_allowed is false, use that exact location question next unless the caller is already answering it",
            "- if candidate_needs_confirmation is true, your next spoken turn must confirm that candidate before any other new question whenever you get a chance to speak",
            "- if must_say_next is present, your next turn must say that exact confirmation sentence before any other question",
            "- when location is unresolved, keep returning to the location ladder after any brief safety instruction until location_search_allowed becomes true or location_dead_end becomes true",
            "- if location_lock_active is true, location is the master priority until it becomes meaningful or reaches a dead end",
            "- while location_lock_active is true, do not switch to people count, injury detail, or other follow-up questions before finishing the current location step",
            "- if the caller reports immediate danger while location_lock_active is true, you may give one short safety instruction, then go straight back to the queued location question",
            "- if the caller asks you to send help before location is usable, say briefly that you need a usable location to get the team to them, then ask the queued location question",
            "- if the caller asks where help is being sent while location is still unresolved, do not invent a destination; say you need to confirm the location first",
            "- the location ladder is: ask where they are, ask for the exact address, repeat and confirm the candidate, ask them to spell it, then ask for the nearest landmark and surroundings",
            "- if candidate_needs_confirmation is true, repeat the candidate clearly and ask for a yes or no confirmation",
            "- do not geocode or validate location unless location_search_allowed is true",
            "- use update_soft_ledger only for soft guesses or useful notes; never use it to write confirmed facts",
            "- never claim a pin, dispatch, or exact address unless tools or dispatch state confirm it",
            "- if location is unresolved, do not say help is on the way or that a team can reach the caller yet",
        ]
    )
    return lines


def _runtime_prompt(
    *,
    interrupt_count: int = 0,
    interruption_recovery: bool = False,
    triage_context: dict[str, Any] | None = None,
    ledger_context: dict[str, Any] | None = None,
    dispatch_context: dict[str, Any] | None = None,
    human_takeover_active: bool = False,
    handoff_recovery_prompt: str | None = None,
    first_aid_context: dict[str, Any] | None = None,
) -> str:
    prompt = load_dispatch_prompt()
    adaptive_lines: list[str] = []
    if interrupt_count > 0 or interruption_recovery:
        adaptive_lines = [
            "",
            "Adaptive turn-taking policy:",
            f"- interruption_count: {interrupt_count}",
            "- if the caller cuts in or you begin speaking over them, stop escalating the exchange",
            "- apologize briefly with: `Sorry, go ahead. Please repeat that last part.`",
            "- after apologizing, ask no new question in that turn",
            "- wait for the caller to finish before speaking again",
            "- after an interruption, prefer a longer pause before taking the next turn",
        ]
    if interruption_recovery:
        adaptive_lines.extend(
            [
                "",
                "Immediate recovery instruction:",
                "- a recent interruption just occurred",
                "- on your next turn, your first job is recovery, not triage",
                "- say: `Sorry, go ahead. Please repeat that last part.`",
                "- then wait for the caller to finish",
            ]
        )
    triage_lines = _format_triage_context(triage_context)
    ledger_lines = _format_ledger_context(ledger_context)
    dispatch_lines = _format_dispatch_context(dispatch_context)
    takeover_lines = _format_takeover_context(
        human_takeover_active=human_takeover_active,
        handoff_recovery_prompt=handoff_recovery_prompt,
    )
    first_aid_lines = _format_first_aid_context(first_aid_context)
    return prompt + "\n" + "\n".join(adaptive_lines + triage_lines + ledger_lines + dispatch_lines + takeover_lines + first_aid_lines)


def build_tool_defs(*, live_dispatch_mode: str = "slm") -> list[gradbot.ToolDef]:
    return [
        gradbot.ToolDef(
            name=name,
            description=description,
            parameters_json=parameters_json,
        )
        for name, description, parameters_json in build_gradbot_tool_defs(
            include_soft_ledger=live_dispatch_mode == "slm"
        )
    ]


def build_session_config(
    *,
    live_dispatch_mode: str = "slm",
    interrupt_count: int = 0,
    interruption_recovery: bool = False,
    triage_context: dict[str, Any] | None = None,
    ledger_context: dict[str, Any] | None = None,
    dispatch_context: dict[str, Any] | None = None,
    human_takeover_active: bool = False,
    handoff_recovery_prompt: str | None = None,
    first_aid_context: dict[str, Any] | None = None,
) -> gradbot.SessionConfig:
    voice = gradbot.flagship_voice("Emma")
    prompt = _runtime_prompt(
        interrupt_count=interrupt_count,
        interruption_recovery=interruption_recovery,
        triage_context=triage_context,
        ledger_context=ledger_context,
        dispatch_context=dispatch_context,
        human_takeover_active=human_takeover_active,
        handoff_recovery_prompt=handoff_recovery_prompt,
        first_aid_context=first_aid_context,
    )
    flush_duration_s, silence_timeout_s = _adaptive_turn_settings(interrupt_count)
    return gradbot.SessionConfig(
        voice_id=voice.voice_id,
        language=voice.language,
        assistant_speaks_first=True,
        instructions=prompt,
        silence_timeout_s=silence_timeout_s,
        flush_duration_s=flush_duration_s,
        tools=build_tool_defs(live_dispatch_mode=live_dispatch_mode),
    )
