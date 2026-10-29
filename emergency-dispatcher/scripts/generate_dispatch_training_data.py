#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "generated"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

PRIORITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
YES_NO_UNKNOWN = ("yes", "no", "unknown")
ISSUE_TYPES = (
    "VEHICLE_BREAKDOWN",
    "FLAT_TYRE",
    "BATTERY_DEAD",
    "FUEL_EMPTY",
    "WRONG_FUEL",
    "LOCKED_OUT",
    "MINOR_COLLISION",
    "ROAD_HAZARD",
    "VEHICLE_OVERHEATING",
    "LOST_OR_STRANDED",
    "VEHICLE_ACCIDENT_WITH_INJURY",
    "VEHICLE_FIRE",
    "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
)
SERIOUS_TYPES = {
    "VEHICLE_ACCIDENT_WITH_INJURY",
    "VEHICLE_FIRE",
    "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
}
COMMON_TYPES = [
    "VEHICLE_BREAKDOWN",
    "FLAT_TYRE",
    "BATTERY_DEAD",
    "FUEL_EMPTY",
    "WRONG_FUEL",
    "LOCKED_OUT",
    "MINOR_COLLISION",
    "ROAD_HAZARD",
    "VEHICLE_OVERHEATING",
    "LOST_OR_STRANDED",
]
NONCRITICAL_DEFAULT_PRIORITY = {
    "VEHICLE_BREAKDOWN": "LOW",
    "FLAT_TYRE": "LOW",
    "BATTERY_DEAD": "LOW",
    "FUEL_EMPTY": "LOW",
    "WRONG_FUEL": "MEDIUM",
    "LOCKED_OUT": "LOW",
    "MINOR_COLLISION": "MEDIUM",
    "ROAD_HAZARD": "MEDIUM",
    "VEHICLE_OVERHEATING": "HIGH",
    "LOST_OR_STRANDED": "MEDIUM",
}
SERIOUS_DEFAULT_PRIORITY = {
    "VEHICLE_ACCIDENT_WITH_INJURY": "CRITICAL",
    "VEHICLE_FIRE": "CRITICAL",
    "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE": "CRITICAL",
}
ALWAYS_ESCALATE_CUES = (
    "weapon",
    "gun",
    "shooting",
    "explosion",
    "child locked in",
    "person in water",
    "drowning",
    "multiple casualties",
    "my kids",
    "children",
    "i am a child",
)
FIRE_CUES = ("fire", "smoke", "burning", "flames", "brennt", "rauch")
MEDICAL_CUES = (
    "not breathing",
    "unresponsive",
    "unconscious",
    "passed out",
    "pass out",
    "can't breathe",
    "collapsed",
    "atmet nicht",
    "bewusstlos",
)
ACCIDENT_SEVERE_CUES = (
    "trapped",
    "bleeding",
    "blood",
    "overturned",
    "on its side",
    "rolled over",
    "eingeklemmt",
    "auf der seite",
    "blut",
    "unfall",
)
STYLE_AXES = {
    "speaker_style": [
        "panicked and nonlinear",
        "calm but understated",
        "non-native English with short clauses",
        "stressed and repetitive",
        "injured caller speaking in fragments",
        "roadside assistance tone with confusion",
    ],
    "location_style": [
        "exact street address",
        "landmark plus area only",
        "junction or road sign only",
        "approximate area in Berlin only",
        "mixed exact and vague clues",
    ],
    "noise_style": [
        "mild background traffic noise",
        "heavy road noise and clipped speech",
        "caller interrupted by other voices",
        "phone audio with missing words",
        "caller breathing hard and speaking unevenly",
    ],
}


DECODER_SYSTEM_PROMPT = """You convert one caller utterance into structured emergency triage JSON.
Return JSON only.
Do not include markdown.
Use exactly these keys:
- location
- issue_type
- priority
- victim_breathing
- victim_bleeding
- victim_conscious
- victim_count
- caller_safe
- escalate_to_human
- escalation_reason
- dispatcher_next_action

Allowed values:
- issue_type: one of %s
- priority: one of %s
- victim_breathing, victim_bleeding, victim_conscious, caller_safe: yes/no/unknown
- victim_count: integer or "unknown"
- escalate_to_human: true/false
- escalation_reason: string or null
- location: string or null
- dispatcher_next_action: short operational sentence
""" % (", ".join(ISSUE_TYPES), ", ".join(PRIORITIES))


@dataclass
class ScenarioPlan:
    incident_type: str
    priority: str
    escalate: bool
    speaker_style: str
    location_style: str
    noise_style: str
    seed_hint: str


def ensure_ollama(base_url: str) -> None:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - runtime UX
        raise SystemExit(
            "Could not reach Ollama at %s. Start it with:\n\n"
            "  /opt/homebrew/bin/ollama serve\n\n"
            "Then rerun this script.\n\nOriginal error: %s" % (base_url, exc)
        ) from exc


def call_ollama_json(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.5,
    seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=120.0)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    return json.loads(raw)


def choose_incident(index: int) -> str:
    if index % 13 == 0:
        return random.choice(tuple(SERIOUS_TYPES))
    if index % 17 == 0:
        return random.choice(tuple(SERIOUS_TYPES))
    return random.choice(COMMON_TYPES)


def plan_for_index(index: int) -> ScenarioPlan:
    incident_type = choose_incident(index)
    priority = SERIOUS_DEFAULT_PRIORITY.get(incident_type, NONCRITICAL_DEFAULT_PRIORITY.get(incident_type, "MEDIUM"))
    escalate = incident_type in SERIOUS_TYPES
    if index % 11 == 0:
        escalate = True
        priority = "CRITICAL"
        seed_hint = "sounds minor at first but hides an always-escalate condition"
    else:
        seed_hint = "stable if common, urgent if serious"
    return ScenarioPlan(
        incident_type=incident_type,
        priority=priority,
        escalate=escalate,
        speaker_style=random.choice(STYLE_AXES["speaker_style"]),
        location_style=random.choice(STYLE_AXES["location_style"]),
        noise_style=random.choice(STYLE_AXES["noise_style"]),
        seed_hint=seed_hint,
    )


def example_prompt(plan: ScenarioPlan) -> str:
    serious_note = (
        "Always escalate if weapons, shooting, explosion, child alone, person in water, active danger, multiple casualties, or unusable location appears."
    )
    return f"""
Generate ONE synthetic emergency-dispatch triage training example as strict JSON.

Task:
- Simulate a realistic caller utterance, not a dispatcher response.
- The utterance should be messy, emotional, and operationally plausible.
- The model's target job is extraction + classification + escalation.

Scenario requirements:
- incident_type target: {plan.incident_type}
- target priority: {plan.priority}
- target escalate_to_human: {str(plan.escalate).lower()}
- speaker style: {plan.speaker_style}
- location style: {plan.location_style}
- noise style: {plan.noise_style}
- planning hint: {plan.seed_hint}

Rules:
- Use Berlin or Berlin-adjacent place naming often.
- Include realistic partial location clues when exact address is unknown.
- Make the caller utterance sound like one turn from a call, not a perfect report.
- Include ambiguity sometimes, but keep the gold labels internally coherent.
- {serious_note}

Return exactly this JSON object:
{{
  "raw_utterance": "...",
  "extracted_fields": {{
    "location": string or null,
    "issue_type": "{plan.incident_type}" or another allowed issue type if the utterance truly supports it,
    "priority": "{plan.priority}" or another allowed priority if the utterance truly supports it,
    "victim_breathing": "yes"|"no"|"unknown",
    "victim_bleeding": "yes"|"no"|"unknown",
    "victim_conscious": "yes"|"no"|"unknown",
    "victim_count": integer or "unknown",
    "caller_safe": "yes"|"no"|"unknown",
    "escalate_to_human": true|false,
    "escalation_reason": string or null
  }},
  "escalate_to_human": true|false,
  "escalation_reason": string or null,
  "dispatcher_next_action": "..."
}}

No markdown. JSON only.
""".strip()


def repair_prompt(bad_example: dict[str, Any], errors: list[str]) -> str:
    return f"""
Repair this emergency-triage JSON example so it matches the required schema and rules.
Return JSON only.

Errors:
{json.dumps(errors, indent=2)}

Current object:
{json.dumps(bad_example, indent=2, ensure_ascii=False)}
""".strip()


def normalize_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_yes_no_unknown(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in YES_NO_UNKNOWN:
        return lowered
    if lowered in {"true", "y", "breathing", "awake", "safe"}:
        return "yes"
    if lowered in {"false", "n", "not breathing", "unconscious", "unsafe"}:
        return "no"
    return "unknown"


def normalize_priority(value: Any) -> str:
    text = str(value).strip().upper()
    if text in PRIORITIES:
        return text
    if "CRIT" in text:
        return "CRITICAL"
    if "HIGH" in text:
        return "HIGH"
    if "MED" in text:
        return "MEDIUM"
    return "LOW"


def normalize_issue_type(value: Any) -> str:
    text = str(value).strip().upper()
    if text in ISSUE_TYPES:
        return text
    return "LOST_OR_STRANDED"


def normalize_victim_count(value: Any) -> int | str:
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip().lower()
    if text == "unknown":
        return "unknown"
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return "unknown"


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    fields = dict(example.get("extracted_fields") or {})
    location = normalize_text_or_none(fields.get("location"))
    issue_type = normalize_issue_type(fields.get("issue_type"))
    priority = normalize_priority(fields.get("priority"))
    victim_breathing = normalize_yes_no_unknown(fields.get("victim_breathing"))
    victim_bleeding = normalize_yes_no_unknown(fields.get("victim_bleeding"))
    victim_conscious = normalize_yes_no_unknown(fields.get("victim_conscious"))
    victim_count = normalize_victim_count(fields.get("victim_count"))
    caller_safe = normalize_yes_no_unknown(fields.get("caller_safe"))
    escalate_to_human = bool(fields.get("escalate_to_human", example.get("escalate_to_human", False)))
    escalation_reason = normalize_text_or_none(fields.get("escalation_reason") or example.get("escalation_reason"))
    raw_utterance = str(example.get("raw_utterance") or "").strip()
    dispatcher_next_action = str(example.get("dispatcher_next_action") or "").strip()

    normalized = {
        "raw_utterance": raw_utterance,
        "extracted_fields": {
            "location": location,
            "issue_type": issue_type,
            "priority": priority,
            "victim_breathing": victim_breathing,
            "victim_bleeding": victim_bleeding,
            "victim_conscious": victim_conscious,
            "victim_count": victim_count,
            "caller_safe": caller_safe,
            "escalate_to_human": escalate_to_human,
            "escalation_reason": escalation_reason,
        },
        "escalate_to_human": escalate_to_human,
        "escalation_reason": escalation_reason,
        "dispatcher_next_action": dispatcher_next_action,
    }
    return normalized


def implied_force_escalate(example: dict[str, Any]) -> bool:
    fields = example["extracted_fields"]
    raw = example["raw_utterance"].lower()
    if fields["priority"] == "CRITICAL":
        return True
    if fields["issue_type"] in SERIOUS_TYPES:
        return True
    return any(cue in raw for cue in ALWAYS_ESCALATE_CUES)


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not example["raw_utterance"] or len(example["raw_utterance"].split()) < 6:
        errors.append("raw_utterance is too short")
    fields = example["extracted_fields"]
    for field_name in (
        "location",
        "issue_type",
        "priority",
        "victim_breathing",
        "victim_bleeding",
        "victim_conscious",
        "victim_count",
        "caller_safe",
        "escalate_to_human",
        "escalation_reason",
    ):
        if field_name not in fields:
            errors.append(f"missing extracted_fields.{field_name}")
    if fields["issue_type"] not in ISSUE_TYPES:
        errors.append("issue_type invalid")
    if fields["priority"] not in PRIORITIES:
        errors.append("priority invalid")
    for status_key in ("victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe"):
        if fields[status_key] not in YES_NO_UNKNOWN:
            errors.append(f"{status_key} invalid")
    if not (fields["victim_count"] == "unknown" or isinstance(fields["victim_count"], int)):
        errors.append("victim_count invalid")
    if fields["escalate_to_human"] != example["escalate_to_human"]:
        errors.append("top-level and extracted_fields escalate_to_human disagree")
    if normalize_text_or_none(fields["escalation_reason"]) != normalize_text_or_none(example["escalation_reason"]):
        errors.append("top-level and extracted_fields escalation_reason disagree")
    if implied_force_escalate(example) and not example["escalate_to_human"]:
        errors.append("example should force escalation but does not")
    if example["escalate_to_human"] and not normalize_text_or_none(example["escalation_reason"]):
        errors.append("escalating example missing escalation_reason")
    if not example["dispatcher_next_action"]:
        errors.append("dispatcher_next_action missing")
    raw = example["raw_utterance"].lower()
    issue_type = fields["issue_type"]
    priority = fields["priority"]
    if any(cue in raw for cue in FIRE_CUES):
        if issue_type != "VEHICLE_FIRE":
            errors.append("fire-like utterance not labeled VEHICLE_FIRE")
        if priority not in {"HIGH", "CRITICAL"}:
            errors.append("fire-like utterance priority too low")
    if any(cue in raw for cue in MEDICAL_CUES):
        if issue_type not in {"PERSON_UNCONSCIOUS_OR_UNRESPONSIVE", "VEHICLE_ACCIDENT_WITH_INJURY"}:
            errors.append("medical life-risk utterance not labeled serious")
        if not example["escalate_to_human"]:
            errors.append("medical life-risk utterance should escalate")
    if any(cue in raw for cue in ACCIDENT_SEVERE_CUES):
        if issue_type in {"FLAT_TYRE", "VEHICLE_BREAKDOWN", "BATTERY_DEAD", "FUEL_EMPTY", "WRONG_FUEL", "LOCKED_OUT"}:
            errors.append("severe crash-like utterance mislabeled as minor roadside issue")
    if issue_type == "FLAT_TYRE" and not any(token in raw for token in ("flat tire", "flat tyre", "puncture", "blown tire", "blown tyre")):
        errors.append("FLAT_TYRE label without a tyre cue")
    if issue_type == "VEHICLE_BREAKDOWN" and any(cue in raw for cue in FIRE_CUES + MEDICAL_CUES):
        errors.append("VEHICLE_BREAKDOWN label conflicts with fire or medical cue")
    if issue_type == "VEHICLE_OVERHEATING" and any(cue in raw for cue in FIRE_CUES):
        errors.append("VEHICLE_OVERHEATING label conflicts with fire cue")
    if issue_type == "BATTERY_DEAD":
        if "phone" in raw and "car" not in raw and "vehicle" not in raw and "engine" not in raw:
            errors.append("BATTERY_DEAD mislabeled from phone battery context")
        if not any(token in raw for token in ("car", "vehicle", "engine", "won't start", "wont start", "battery", "stopped")):
            errors.append("BATTERY_DEAD label without vehicle battery cue")
    if issue_type == "MINOR_COLLISION" and (
        fields["victim_breathing"] == "no"
        or fields["victim_conscious"] == "no"
        or fields["victim_bleeding"] == "yes"
    ):
        errors.append("MINOR_COLLISION label conflicts with life-risk victim fields")
    if fields["victim_breathing"] == "no" and priority not in {"HIGH", "CRITICAL"}:
        errors.append("priority too low for victim_breathing=no")
    if fields["victim_conscious"] == "no" and not example["escalate_to_human"]:
        errors.append("unconscious victim should escalate")
    if fields["caller_safe"] == "yes" and any(token in raw for token in ("i'm trapped", "i am trapped", "fire in the vehicle", "stuck in the car")):
        errors.append("caller_safe=yes conflicts with trapped or vehicle fire wording")
    return errors


def example_quality_score(example: dict[str, Any]) -> int:
    raw = example["raw_utterance"].lower()
    score = 0
    if len(raw.split()) >= 10:
        score += 20
    if any(token in raw for token in ("uh", "don't know", "right now", "please", "help", "i think")):
        score += 20
    if example["extracted_fields"]["location"]:
        score += 15
    if example["dispatcher_next_action"]:
        score += 15
    if implied_force_escalate(example) == example["escalate_to_human"]:
        score += 20
    if any(place in raw for place in ("berlin", "mitte", "donaustra", "delta campus", "alexanderplatz", "muller")):
        score += 10
    return score


def generate_one(
    *,
    base_url: str,
    model: str,
    plan: ScenarioPlan,
    seed: int,
) -> dict[str, Any]:
    initial = call_ollama_json(
        base_url=base_url,
        model=model,
        prompt=example_prompt(plan),
        temperature=0.7,
        seed=seed,
    )
    normalized = normalize_example(initial)
    errors = validate_example(normalized)
    if errors:
        repaired = call_ollama_json(
            base_url=base_url,
            model=model,
            prompt=repair_prompt(normalized, errors),
            temperature=0.2,
            seed=seed + 999,
        )
        normalized = normalize_example(repaired)
        errors = validate_example(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    normalized["_quality_score"] = example_quality_score(normalized)
    return normalized


def build_decoder_record(example: dict[str, Any]) -> dict[str, Any]:
    assistant_payload = {
        **example["extracted_fields"],
        "dispatcher_next_action": example["dispatcher_next_action"],
    }
    return {
        "messages": [
            {"role": "system", "content": DECODER_SYSTEM_PROMPT},
            {"role": "user", "content": example["raw_utterance"]},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
    }


def _first_matching_span(text: str, candidates: list[str]) -> tuple[str, str] | None:
    lowered = text.lower()
    for candidate in candidates:
        if not candidate:
            continue
        idx = lowered.find(candidate.lower())
        if idx >= 0:
            return text[idx : idx + len(candidate)], candidate
    return None


def build_ner_seed_record(example: dict[str, Any]) -> dict[str, Any] | None:
    text = example["raw_utterance"]
    entities: list[list[str]] = []
    location = example["extracted_fields"]["location"]
    if location and location.lower() in text.lower():
        entities.append([location, "LOCATION"])

    phrase_hints = {
        "ISSUE_TYPE": [
            "car accident",
            "flat tire",
            "flat tyre",
            "won't start",
            "out of fuel",
            "wrong fuel",
            "locked out",
            "smoke",
            "fire",
            "not breathing",
            "unresponsive",
            "overheating",
            "road hazard",
            "debris",
            "stranded",
        ],
        "BREATHING_STATUS": ["breathing", "not breathing", "can't breathe", "unresponsive"],
        "BLEEDING_STATUS": ["bleeding", "bleeding heavily", "heavy bleeding", "blood everywhere"],
        "CONSCIOUSNESS_STATUS": ["awake", "conscious", "unconscious", "passed out", "not responding"],
        "CALLER_SAFETY": ["i am safe", "we are safe", "unsafe", "i am trapped", "i'm trapped"],
    }
    for label, hints in phrase_hints.items():
        match = _first_matching_span(text, hints)
        if match:
            entities.append([match[0], label])

    victim_count = example["extracted_fields"]["victim_count"]
    if isinstance(victim_count, int):
        count_match = re.search(rf"\b{victim_count}\b|\b(one|two|three|four|five)\b", text, flags=re.IGNORECASE)
        if count_match:
            entities.append([count_match.group(0), "VICTIM_COUNT"])

    if not entities:
        return None
    return {"text": text, "entities": entities}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Pioneer-ready emergency dispatch fine-tuning data with Ollama.")
    parser.add_argument("--count", type=int, default=50, help="Number of examples to generate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between generations.")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_ollama(args.base_url)

    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index in range(args.count):
        plan = plan_for_index(index)
        try:
            example = generate_one(
                base_url=args.base_url,
                model=args.model,
                plan=plan,
                seed=args.seed + index,
            )
            example["_plan"] = plan.__dict__
            examples.append(example)
            print(f"[{index + 1:>3}/{args.count}] ok  {example['extracted_fields']['issue_type']}  q={example['_quality_score']}")
        except Exception as exc:  # pragma: no cover - runtime UX
            failures.append({"index": index, "plan": plan.__dict__, "error": str(exc)})
            print(f"[{index + 1:>3}/{args.count}] err {plan.incident_type} :: {exc}", file=sys.stderr)
        if args.sleep:
            time.sleep(args.sleep)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = args.out_dir / f"dispatch_seed_examples_{stamp}.jsonl"
    decoder_path = args.out_dir / f"pioneer_decoder_dispatch_{stamp}.jsonl"
    ner_path = args.out_dir / f"pioneer_gliner_ner_seed_{stamp}.jsonl"
    failures_path = args.out_dir / f"dispatch_seed_failures_{stamp}.json"

    write_jsonl(raw_path, examples)
    write_jsonl(decoder_path, [build_decoder_record(item) for item in examples])
    ner_rows = [row for row in (build_ner_seed_record(item) for item in examples) if row]
    write_jsonl(ner_path, ner_rows)
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    avg_quality = round(sum(item["_quality_score"] for item in examples) / max(len(examples), 1), 2)
    summary = {
        "count_requested": args.count,
        "count_generated": len(examples),
        "count_failed": len(failures),
        "average_quality_score": avg_quality,
        "raw_examples_path": str(raw_path),
        "decoder_dataset_path": str(decoder_path),
        "ner_seed_dataset_path": str(ner_path),
        "failures_path": str(failures_path),
        "note": (
            "Use the decoder dataset as the main Pioneer training input for full structured JSON output. "
            "Use the NER seed file only as an optional GLiNER-style extraction sidecar dataset."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
