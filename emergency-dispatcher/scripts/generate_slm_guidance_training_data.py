#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from emergency_dispatcher.slm_guidance import GUIDANCE_ISSUE_TYPES
from emergency_dispatcher.slm_guidance import SLM_GUIDANCE_SCHEMA_VERSION
from emergency_dispatcher.slm_guidance import GuidanceSeed
from emergency_dispatcher.slm_guidance import build_gliner_sidecar_record
from emergency_dispatcher.slm_guidance import build_guidance_record


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "generated"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_RETRIES = 3
MAX_SLOT_ATTEMPTS = 5

BAND_SEQUENCE = (
    "low",
    "low",
    "low",
    "low",
    "medium",
    "medium",
    "medium",
    "medium",
    "high",
    "high",
    "high",
    "high",
    "critical",
    "critical",
    "critical",
    "critical",
    "delta_spike",
    "delta_spike",
    "contradiction_drop",
    "contradiction_drop",
)


@dataclass
class ScenarioTemplate:
    issue_type: str
    caller_role: str
    prompt_focus: str


@dataclass
class ScenarioPlan:
    band: str
    issue_type: str
    caller_role: str
    language_style: str
    previous_severity: int
    include_location: bool
    contradiction: bool
    prompt_focus: str
    required_signals: tuple[str, ...]


SCENARIOS: dict[str, list[ScenarioTemplate]] = {
    "low": [
        ScenarioTemplate(
            issue_type="VEHICLE_BREAKDOWN",
            caller_role="victim",
            prompt_focus="car will not start, caller is safe, no injury, wants help but situation is stable",
        ),
        ScenarioTemplate(
            issue_type="LOST_OR_STRANDED",
            caller_role="victim",
            prompt_focus="caller is stranded in a safe place, worried but not in immediate danger",
        ),
        ScenarioTemplate(
            issue_type="ROAD_HAZARD",
            caller_role="witness",
            prompt_focus="debris or stalled vehicle is creating a risk, but nobody is visibly injured",
        ),
        ScenarioTemplate(
            issue_type="SMOKE_INVESTIGATION",
            caller_role="witness",
            prompt_focus="odd smell or faint smoke with no visible flames and no one in immediate danger",
        ),
    ],
    "medium": [
        ScenarioTemplate(
            issue_type="CHILD_MISSING_OR_UNACCOUNTED",
            caller_role="witness",
            prompt_focus="child is briefly unaccounted for in a busy area, situation is concerning but not clearly life-threatening",
        ),
        ScenarioTemplate(
            issue_type="VEHICLE_ACCIDENT",
            caller_role="victim",
            prompt_focus="low-speed collision, shaken caller, unclear injury severity, traffic risk present",
        ),
        ScenarioTemplate(
            issue_type="MEDICAL_EMERGENCY",
            caller_role="witness",
            prompt_focus="responsive but vulnerable person is getting weaker, caller is alone with them, and the scene feels increasingly urgent",
        ),
        ScenarioTemplate(
            issue_type="GAS_LEAK_OR_HAZMAT",
            caller_role="witness",
            prompt_focus="gas or chemical smell reported, no collapse yet, building occupants uncertain",
        ),
        ScenarioTemplate(
            issue_type="ROAD_HAZARD",
            caller_role="witness",
            prompt_focus="debris or a stalled vehicle is causing near misses and one person may already be lightly injured",
        ),
        ScenarioTemplate(
            issue_type="SMOKE_INVESTIGATION",
            caller_role="victim",
            prompt_focus="thick smoke or electrical smell is affecting breathing, but there are no visible flames yet",
        ),
        ScenarioTemplate(
            issue_type="MEDICAL_EMERGENCY",
            caller_role="victim",
            prompt_focus="caller reports an elderly or pregnant person feeling faint, responsive but worsening, with no obvious trauma",
        ),
    ],
    "high": [
        ScenarioTemplate(
            issue_type="BUILDING_FIRE",
            caller_role="victim",
            prompt_focus="smoke or flames in a building, caller is not yet trapped but urgency is rising",
        ),
        ScenarioTemplate(
            issue_type="BREATHING_DISTRESS",
            caller_role="victim",
            prompt_focus="caller is having breathing trouble, possibly with smoke exposure, and needs immediate help",
        ),
        ScenarioTemplate(
            issue_type="TRAPPED_PERSON",
            caller_role="victim",
            prompt_focus="caller cannot get out or is pinned, but is still conscious and talking",
        ),
        ScenarioTemplate(
            issue_type="VEHICLE_ACCIDENT_WITH_INJURY",
            caller_role="witness",
            prompt_focus="injury is confirmed after a crash, caller is alone with the victim or scene is worsening",
        ),
    ],
    "critical": [
        ScenarioTemplate(
            issue_type="PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
            caller_role="victim",
            prompt_focus="person is not responding or not breathing, caller is overwhelmed, life threat is explicit",
        ),
        ScenarioTemplate(
            issue_type="BUILDING_FIRE",
            caller_role="victim",
            prompt_focus="active indoor fire close to the caller with immediate danger and possible trapped occupants",
        ),
        ScenarioTemplate(
            issue_type="ACTIVE_THREAT_OR_WEAPON",
            caller_role="witness",
            prompt_focus="weapon, shooting, or explosion is directly mentioned and danger is immediate",
        ),
        ScenarioTemplate(
            issue_type="VEHICLE_FIRE",
            caller_role="victim",
            prompt_focus="vehicle is burning near the caller or someone may still be inside",
        ),
    ],
    "delta_spike": [
        ScenarioTemplate(
            issue_type="PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
            caller_role="witness",
            prompt_focus="earlier turn sounded manageable, but this turn reveals the victim is no longer responding or breathing",
        ),
        ScenarioTemplate(
            issue_type="BUILDING_FIRE",
            caller_role="victim",
            prompt_focus="earlier turn sounded like smoke only, but this turn reveals flames, trapped people, or rapidly worsening danger",
        ),
        ScenarioTemplate(
            issue_type="CHILD_MISSING_OR_UNACCOUNTED",
            caller_role="witness",
            prompt_focus="earlier turn sounded stable, but this turn makes it clear a child is missing and cannot be reached",
        ),
    ],
    "contradiction_drop": [
        ScenarioTemplate(
            issue_type="SMOKE_INVESTIGATION",
            caller_role="victim",
            prompt_focus="caller corrects an earlier overstatement like fire or collapse; the situation is still worth checking but less severe",
        ),
        ScenarioTemplate(
            issue_type="MEDICAL_EMERGENCY",
            caller_role="witness",
            prompt_focus="caller clarifies that the person is breathing or responsive after sounding much worse earlier",
        ),
        ScenarioTemplate(
            issue_type="VEHICLE_BREAKDOWN",
            caller_role="victim",
            prompt_focus="caller retracts the scary part of the story; the turn should clearly de-escalate from the previous severity",
        ),
        ScenarioTemplate(
            issue_type="CHILD_MISSING_OR_UNACCOUNTED",
            caller_role="witness",
            prompt_focus="caller corrects the panic: the child has now been found and is safe, so severity should drop sharply",
        ),
        ScenarioTemplate(
            issue_type="SMOKE_INVESTIGATION",
            caller_role="witness",
            prompt_focus="caller corrects an apparent building fire into burnt food, steam, or a small appliance issue with everyone safe",
        ),
        ScenarioTemplate(
            issue_type="BREATHING_DISTRESS",
            caller_role="victim",
            prompt_focus="caller corrects a frightening breathing report because an inhaler or fresh air helped and the immediate threat has eased",
        ),
    ],
}

LANGUAGE_STYLES = (
    "english",
    "english with short clipped phrases",
    "german",
    "mixed english and german",
)

BAND_DESCRIPTIONS = {
    "low": "Severity must land between 0 and 30.",
    "medium": "Severity must land between 31 and 60.",
    "high": "Severity must land between 61 and 85.",
    "critical": "Severity must land between 86 and 100.",
    "delta_spike": "Severity must jump sharply from the previous turn with severity_delta at least +25.",
    "contradiction_drop": "Severity must drop materially from the previous turn with severity_delta at most -15.",
}


def band_instance(index: int, band: str) -> int:
    cycle_size = len(BAND_SEQUENCE)
    full_cycles, remainder = divmod(index, cycle_size)
    seen = full_cycles * BAND_SEQUENCE.count(band)
    seen += sum(1 for slot in BAND_SEQUENCE[: remainder + 1] if slot == band) - 1
    return seen


def build_plan(index: int) -> ScenarioPlan:
    band = BAND_SEQUENCE[index % len(BAND_SEQUENCE)]
    scenario_index = band_instance(index, band) % len(SCENARIOS[band])
    scenario = SCENARIOS[band][scenario_index]
    language_style = random.choice(LANGUAGE_STYLES)
    if band == "low":
        previous = random.randint(0, 20)
        include_location = random.random() < 0.7
        contradiction = False
    elif band == "medium":
        previous = random.randint(10, 35)
        include_location = random.random() < 0.75
        contradiction = False
    elif band == "high":
        previous = random.randint(25, 55)
        include_location = random.random() < 0.8
        contradiction = False
    elif band == "critical":
        previous = random.randint(45, 80)
        include_location = random.random() < 0.8
        contradiction = False
    elif band == "delta_spike":
        previous = random.randint(10, 40)
        include_location = random.random() < 0.8
        contradiction = False
    else:
        previous = random.randint(55, 90)
        include_location = random.random() < 0.75
        contradiction = True
    return ScenarioPlan(
        band=band,
        issue_type=scenario.issue_type,
        caller_role=scenario.caller_role,
        language_style=language_style,
        previous_severity=previous,
        include_location=include_location,
        contradiction=contradiction,
        prompt_focus=scenario.prompt_focus,
        required_signals=required_signals_for(band, scenario.issue_type),
    )


def required_signals_for(band: str, issue_type: str) -> tuple[str, ...]:
    if band == "low":
        return ()
    if band == "medium":
        if issue_type == "CHILD_MISSING_OR_UNACCOUNTED":
            return ("children_unaccounted_for", "situation_worsening")
        if issue_type == "GAS_LEAK_OR_HAZMAT":
            return ("vulnerable_adult_involved", "caller_cannot_leave", "situation_worsening")
        if issue_type == "VEHICLE_ACCIDENT":
            return ("injury_reported", "bleeding_any", "situation_worsening")
        if issue_type == "MEDICAL_EMERGENCY":
            return ("vulnerable_adult_involved", "caller_alone_with_victim", "caller_cannot_leave", "situation_worsening")
        if issue_type == "ROAD_HAZARD":
            return ("injury_reported", "caller_cannot_leave", "situation_worsening")
        if issue_type == "SMOKE_INVESTIGATION":
            return ("breathing_difficulty_with_smoke", "caller_cannot_leave")
        return ("vulnerable_adult_involved", "breathing_difficulty_with_smoke", "caller_cannot_leave", "situation_worsening")
    if band == "high":
        if issue_type == "BUILDING_FIRE":
            return ("indoor_fire_immediate", "caller_cannot_leave", "situation_worsening")
        if issue_type == "BREATHING_DISTRESS":
            return (
                "breathing_difficulty_with_smoke",
                "vulnerable_adult_involved",
                "caller_cannot_leave",
                "situation_worsening",
            )
        if issue_type == "TRAPPED_PERSON":
            return ("trapped_or_cannot_exit", "injury_reported", "caller_cannot_leave", "situation_worsening")
        return (
            "injury_reported",
            "bleeding_any",
            "caller_alone_with_victim",
            "situation_worsening",
            "trapped_or_cannot_exit",
        )
    if band == "critical":
        if issue_type == "VEHICLE_FIRE":
            return (
                "vehicle_fire_adjacent",
                "trapped_or_cannot_exit",
                "caller_cannot_leave",
                "situation_worsening",
                "caller_alone_with_victim",
            )
        if issue_type == "BUILDING_FIRE":
            return ("indoor_fire_immediate", "trapped_or_cannot_exit", "caller_cannot_leave", "situation_worsening")
        if issue_type == "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE":
            return (
                "unconscious_or_unresponsive",
                "caller_cannot_leave",
                "caller_alone_with_victim",
                "situation_worsening",
                "multiple_casualties",
            )
        return (
            "weapon_shooting_explosion",
            "heavy_bleeding",
            "caller_cannot_leave",
            "multiple_casualties",
            "situation_worsening",
        )
    if band == "delta_spike":
        if issue_type == "CHILD_MISSING_OR_UNACCOUNTED":
            return ("children_unaccounted_for", "child_in_danger", "child_unreachable", "situation_worsening")
        if issue_type == "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE":
            return (
                "unconscious_or_unresponsive",
                "caller_cannot_leave",
                "caller_alone_with_victim",
                "situation_worsening",
            )
        return ("indoor_fire_immediate", "trapped_or_cannot_exit", "caller_cannot_leave", "situation_worsening")
    return ("caller_calm_coherent", "situation_stable_static")


SIGNAL_HINTS = {
    "unconscious_or_unresponsive": "Say the person is not responding, unconscious, or not breathing.",
    "indoor_fire_immediate": "Make the fire active and close to the caller inside a building.",
    "vehicle_fire_adjacent": "Make it clear the vehicle is burning next to the caller or occupant.",
    "heavy_bleeding": "Use explicit heavy bleeding wording.",
    "breathing_difficulty_with_smoke": "Include breathing trouble together with smoke exposure.",
    "trapped_or_cannot_exit": "Say someone cannot get out, is pinned, or is trapped.",
    "weapon_shooting_explosion": "Mention a weapon, shooting, or explosion directly.",
    "bleeding_any": "Mention bleeding plainly.",
    "injury_reported": "State that someone is injured.",
    "children_unaccounted_for": "Make it clear a child is missing or unaccounted for.",
    "child_in_danger": "Frame the child as being in danger, not just temporarily out of sight.",
    "child_unreachable": "Make it clear the child cannot be reached or found.",
    "vulnerable_adult_involved": "Mention asthma, pregnancy, elderly age, or disability.",
    "situation_worsening": "State that the situation is getting worse right now.",
    "caller_cannot_leave": "Say the caller cannot safely leave the scene or person.",
    "caller_alone_with_victim": "Make the caller alone with the victim or alone managing the scene.",
    "multiple_casualties": "Mention more than one person affected.",
    "caller_calm_coherent": "Keep the caller calm and coherent.",
    "situation_stable_static": "State the situation is stable or not escalating.",
    "everyone_explicitly_safe": "Use an explicit everyone-is-safe statement.",
}


def example_prompt(plan: ScenarioPlan, index: int) -> str:
    location_rule = (
        "The caller MUST say a dispatch-usable location clue in this utterance. "
        "Use a street, station, highway, landmark, junction, or similar concrete clue."
        if plan.include_location
        else "The caller MUST NOT say any dispatch-usable location clue in this utterance."
    )
    contradiction_rule = (
        "This turn must explicitly correct or soften something that sounded more severe earlier."
        if plan.contradiction
        else "Do not include a correction of an earlier mistake unless it happens naturally."
    )
    required_signal_lines = "\n".join(f"- {SIGNAL_HINTS[name]} ({name}=true)" for name in plan.required_signals)
    if not required_signal_lines:
        required_signal_lines = "- Keep the turn mild and avoid life-threat cues."
    return f"""
Generate ONE realistic caller turn for a dispatch-guidance SLM training dataset.

Return strict JSON only with this shape:
{{
  "raw_utterance": "string",
  "extracted_fields": {{
    "location": "string or null",
    "issue_type": "one of {', '.join(GUIDANCE_ISSUE_TYPES)}",
    "victim_breathing": "yes|no|unknown",
    "victim_bleeding": "yes|no|unknown",
    "victim_conscious": "yes|no|unknown",
    "victim_count": "integer or \\"unknown\\"",
    "caller_safe": "yes|no|unknown"
  }},
  "address_in_utterance": true,
  "scoring_signals": {{
    "unconscious_or_unresponsive": false,
    "indoor_fire_immediate": false,
    "vehicle_fire_adjacent": false,
    "heavy_bleeding": false,
    "breathing_difficulty_with_smoke": false,
    "trapped_or_cannot_exit": false,
    "weapon_shooting_explosion": false,
    "bleeding_any": false,
    "injury_reported": false,
    "children_unaccounted_for": false,
    "child_in_danger": false,
    "child_unreachable": false,
    "vulnerable_adult_involved": false,
    "situation_worsening": false,
    "caller_cannot_leave": false,
    "caller_alone_with_victim": false,
    "multiple_casualties": false,
    "caller_calm_coherent": false,
    "situation_stable_static": false,
    "everyone_explicitly_safe": false,
    "suspicious_transcript_streak": 0,
    "unresolved_address_attempts": 0
  }}
}}

Hard requirements:
- Caller speech only. No dispatcher lines.
- One turn only, 25 to 120 words.
- Make it operationally plausible for a live emergency or incident call.
- Use this exact target issue_type: {plan.issue_type}
- Current band target: {plan.band}. {BAND_DESCRIPTIONS[plan.band]}
- previous_severity for this conversation was {plan.previous_severity}.
- Caller role: {plan.caller_role}.
- Language style: {plan.language_style}.
- Focus: {plan.prompt_focus}
- {location_rule}
- If you include a location, use something concrete like `A100 near Ostkreuz`, `Skalitzer Strasse 22`, or `outside Zoo station`.
- {contradiction_rule}
- Every scoring signal set to true must be directly supported by the raw_utterance.
- If a fact is not stated, use "unknown" rather than inventing it.
- location may only be non-null if the caller explicitly states the location in this utterance.
- address_in_utterance must be true only if the raw_utterance itself contains the location clue.
- Keep the utterance natural and messy, but do not make it impossible to parse.
- Required signal cues for this plan:
{required_signal_lines}
- Seed id: {index + 1}
""".strip()


def repair_prompt(plan: ScenarioPlan, candidate: dict[str, Any], errors: list[str]) -> str:
    required_signal_lines = "\n".join(f"- {SIGNAL_HINTS[name]} ({name}=true)" for name in plan.required_signals)
    if not required_signal_lines:
        required_signal_lines = "- Keep the turn mild and avoid life-threat cues."
    return f"""
Repair this JSON training example so it satisfies every rule.

Plan:
- target issue_type: {plan.issue_type}
- target band: {plan.band}
- previous_severity: {plan.previous_severity}
- include_location: {str(plan.include_location).lower()}
- contradiction: {str(plan.contradiction).lower()}
- required_signals:
{required_signal_lines}

Problems to fix:
{json.dumps(errors, ensure_ascii=False)}

Current candidate:
{json.dumps(candidate, ensure_ascii=False)}

Return corrected JSON only with the same schema and keep the utterance realistic.
If a location is required, put it directly in the raw_utterance and set address_in_utterance=true.
""".strip()


def call_openai_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create high-quality emergency dispatch training examples. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def validate_record(plan: ScenarioPlan, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    severity = record["severity_score"]
    delta = record["severity_delta"]
    if plan.band == "low" and not (0 <= severity <= 30):
        errors.append(f"severity {severity} not in low band")
    if plan.band == "medium" and not (31 <= severity <= 60):
        errors.append(f"severity {severity} not in medium band")
    if plan.band == "high" and not (61 <= severity <= 85):
        errors.append(f"severity {severity} not in high band")
    if plan.band == "critical" and not (86 <= severity <= 100):
        errors.append(f"severity {severity} not in critical band")
    if plan.band == "delta_spike" and delta < 25:
        errors.append(f"severity_delta {delta} is too small for spike")
    if plan.band == "contradiction_drop" and delta > -15:
        errors.append(f"severity_delta {delta} is not negative enough for contradiction drop")
    if record["extracted_fields"]["issue_type"] != plan.issue_type:
        errors.append(
            f"issue_type {record['extracted_fields']['issue_type']} does not match target {plan.issue_type}"
        )
    if plan.include_location and not record["address_in_utterance"]:
        errors.append("expected explicit location clue in utterance")
    if not plan.include_location and record["address_in_utterance"]:
        errors.append("utterance should not contain a dispatch-usable location clue")
    if plan.band in {"high", "critical", "delta_spike"} and not record["human_recommended"]:
        errors.append("human_recommended should be true for this band")
    if plan.band == "critical" and not record["human_required"]:
        errors.append("human_required should be true for critical band")
    if not record["agent_should_continue"]:
        errors.append("agent_should_continue must stay true")
    return errors


def generate_one(client: OpenAI, model: str, plan: ScenarioPlan, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = call_openai_json(client, model, example_prompt(plan, index))
    raw_candidate = candidate
    for attempt in range(MAX_RETRIES):
        errors: list[str] = []
        try:
            seed = GuidanceSeed.model_validate(candidate)
            record = build_guidance_record(seed, plan.previous_severity)
            errors = validate_record(plan, record)
        except Exception as exc:
            record = None
            errors = [str(exc)]
        if not errors:
            assert record is not None
            return record, raw_candidate
        if attempt == MAX_RETRIES - 1:
            raise ValueError("; ".join(errors))
        candidate = call_openai_json(client, model, repair_prompt(plan, candidate, errors))
        raw_candidate = candidate
    raise ValueError("unreachable")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Generate severity-guidance SLM training data with OpenAI for pilot or bulk runs."
    )
    parser.add_argument("--count", type=int, default=20, help="Number of examples to generate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")

    random.seed(args.seed)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    records: list[dict[str, Any]] = []
    seed_examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index in range(args.count):
        plan = build_plan(index)
        slot_error: str | None = None
        for slot_attempt in range(1, MAX_SLOT_ATTEMPTS + 1):
            try:
                record, seed_candidate = generate_one(client, args.model, plan, index)
                record["_plan"] = plan.__dict__
                seed_examples.append(
                    {
                        "raw_candidate": seed_candidate,
                        "final_record": record,
                        "plan": plan.__dict__,
                    }
                )
                records.append(record)
                print(
                    f"[{index + 1:>3}/{args.count}] ok  "
                    f"{record['extracted_fields']['issue_type']}  "
                    f"severity={record['severity_score']} delta={record['severity_delta']:+d} "
                    f"(slot_try={slot_attempt})"
                )
                slot_error = None
                break
            except Exception as exc:  # pragma: no cover - runtime UX
                slot_error = str(exc)
        if slot_error:
            failures.append({"index": index, "plan": plan.__dict__, "error": slot_error})
            print(f"[{index + 1:>3}/{args.count}] err {plan.band} :: {slot_error}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = args.out_dir / f"slm_guidance_seed_examples_{stamp}.jsonl"
    dataset_path = args.out_dir / f"slm_guidance_training_{stamp}.jsonl"
    gliner_path = args.out_dir / f"gliner_guidance_sidecar_{stamp}.jsonl"
    failures_path = args.out_dir / f"slm_guidance_failures_{stamp}.json"

    write_jsonl(raw_path, seed_examples)
    clean_rows = []
    for item in records:
        clean = dict(item)
        clean.pop("_plan", None)
        clean_rows.append(clean)
    write_jsonl(dataset_path, clean_rows)
    gliner_rows = [row for row in (build_gliner_sidecar_record(item) for item in clean_rows) if row]
    write_jsonl(gliner_path, gliner_rows)
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "count_requested": args.count,
        "count_generated": len(clean_rows),
        "count_failed": len(failures),
        "model": args.model,
        "schema_version": SLM_GUIDANCE_SCHEMA_VERSION,
        "raw_seed_examples_path": str(raw_path),
        "slm_guidance_dataset_path": str(dataset_path),
        "gliner_sidecar_path": str(gliner_path),
        "failures_path": str(failures_path),
        "note": (
            "Use the SLM guidance dataset as the main training set for the severity-guidance model. "
            "Use the GLiNER sidecar only for optional span extraction experiments."
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
