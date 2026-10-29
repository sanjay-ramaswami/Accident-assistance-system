#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from emergency_dispatcher.berlin_location_lexicon import DEFAULT_BERLIN_GEOJSON
from emergency_dispatcher.berlin_location_lexicon import DEFAULT_BERLIN_LEXICON_CACHE
from emergency_dispatcher.berlin_location_lexicon import load_or_build_berlin_location_lexicon
from emergency_dispatcher.berlin_location_lexicon import sample_exact_address_seed
from emergency_dispatcher.berlin_location_lexicon import sample_landmark_note
from emergency_dispatcher.berlin_location_lexicon import sample_place_name_seed
from emergency_dispatcher.berlin_location_lexicon import sample_road_or_junction_seed
from emergency_dispatcher.berlin_location_lexicon import sample_sub_location
from emergency_dispatcher.berlin_location_lexicon import sample_vague_location
from emergency_dispatcher.slm_fact_ledger import FACT_LEDGER_SCHEMA_VERSION
from emergency_dispatcher.slm_fact_ledger import FACT_LEDGER_SYSTEM_PROMPT
from emergency_dispatcher.slm_fact_ledger import FACT_ISSUE_CUES
from emergency_dispatcher.slm_fact_ledger import FactLedger
from emergency_dispatcher.slm_fact_ledger import FactLedgerTrainingExample
from emergency_dispatcher.slm_fact_ledger import build_pioneer_fact_ledger_record
from emergency_dispatcher.slm_fact_ledger import has_dispatchable_location_cue


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "generated"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_RETRIES = 3
MAX_SLOT_ATTEMPTS = 5


@dataclass
class ScenarioTemplate:
    kind: str
    prompt_focus: str
    prior_facts: dict[str, Any]
    required_fields: dict[str, Any]
    required_cues: tuple[str, ...] = ()
    forbidden_cues: tuple[str, ...] = ()
    word_min: int = 1
    word_max: int = 40
    location_rule: str = "any"
    preserve_prior_fields: tuple[str, ...] = ()
    required_nonempty_fields: tuple[str, ...] = ()
    seed_payload: dict[str, Any] | None = None
    language_style: str | None = None


def _base_facts(**updates: Any) -> dict[str, Any]:
    return FactLedger(**updates).model_dump()


SCENARIO_TEMPLATES: tuple[ScenarioTemplate, ...] = (
    ScenarioTemplate(
        kind="explicit_exact_address",
        prompt_focus="Caller gives a dispatch-usable street and number in this turn and one explicit hard cue like smoke, trapped, or bleeding.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "exact_address", "address_in_utterance": True},
        word_min=10,
        word_max=32,
        location_rule="dispatchable",
    ),
    ScenarioTemplate(
        kind="explicit_place_name",
        prompt_focus="Caller gives a concrete place name like Delta Campus building or Zoo station in this turn. It is a usable place clue but not a full street address.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "place_name", "address_in_utterance": True},
        word_min=8,
        word_max=24,
        location_rule="dispatchable",
    ),
    ScenarioTemplate(
        kind="road_or_junction",
        prompt_focus="Caller gives a road, exit, station, corner, bridge, or junction clue in this turn.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "road_or_junction", "address_in_utterance": True},
        word_min=8,
        word_max=28,
        location_rule="dispatchable",
    ),
    ScenarioTemplate(
        kind="landmark_note_only",
        prompt_focus="Caller gives only a best-effort landmark or relative-location note like behind the big tower or next to the playground. Capture it in location_note but do not pretend it is a dispatch-usable address.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "none", "address_in_utterance": False, "location_candidate": None},
        required_nonempty_fields=("location_note",),
        word_min=6,
        word_max=16,
        location_rule="no_dispatchable",
    ),
    ScenarioTemplate(
        kind="dispatchable_with_note",
        prompt_focus="Caller gives a usable place or road clue plus a short best-effort landmark or entrance note in the same turn.",
        prior_facts=_base_facts(),
        required_fields={"address_in_utterance": True},
        required_nonempty_fields=("location_candidate", "location_note"),
        word_min=10,
        word_max=24,
        location_rule="dispatchable",
    ),
    ScenarioTemplate(
        kind="vague_home_reject",
        prompt_focus="Caller says a vague location like at home, in the house, or somewhere in Berlin. This must not become a dispatch-usable location.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "vague", "address_in_utterance": False, "location_candidate": None},
        word_min=4,
        word_max=18,
        location_rule="no_dispatchable",
    ),
    ScenarioTemplate(
        kind="sub_location_only",
        prompt_focus="Caller only gives a floor, room, stairwell, entrance, or unit clue with no building or address in this turn.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "sub_location_only", "address_in_utterance": False, "inside_building": "yes"},
        word_min=4,
        word_max=14,
        location_rule="sub_only",
    ),
    ScenarioTemplate(
        kind="confirm_existing_location",
        prompt_focus="Caller gives a short confirmation like yes that is correct. No new address appears in this turn; keep the previous location exactly.",
        prior_facts=_base_facts(
            location_candidate="Delta Campus building",
            location_kind="place_name",
            address_in_utterance=False,
            inside_building="yes",
            issue_cues=["trapped"],
            trapped_status="yes",
        ),
        required_fields={"location_kind": "place_name", "address_in_utterance": False},
        word_min=1,
        word_max=6,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind", "trapped_status", "issue_cues"),
    ),
    ScenarioTemplate(
        kind="reject_existing_location",
        prompt_focus="Caller gives a short rejection like no that is wrong. No new address appears in this turn, so clear the previous location candidate.",
        prior_facts=_base_facts(
            location_candidate="Donaustrasse 44, Berlin",
            location_kind="exact_address",
            address_in_utterance=False,
            issue_cues=["trapped"],
            trapped_status="yes",
        ),
        required_fields={"location_kind": "none", "address_in_utterance": False, "location_candidate": None},
        word_min=1,
        word_max=7,
        location_rule="no_dispatchable",
        preserve_prior_fields=("trapped_status", "issue_cues"),
    ),
    ScenarioTemplate(
        kind="child_fragment",
        prompt_focus="Caller turn is a short fragment like with my daughter or my son is here. Mark child presence, but do not invent anything else.",
        prior_facts=_base_facts(
            location_candidate="Delta Campus building",
            location_kind="place_name",
            inside_building="yes",
            trapped_status="yes",
            issue_cues=["trapped"],
        ),
        required_fields={"child_present": "yes"},
        required_cues=("child_present",),
        word_min=2,
        word_max=6,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind", "trapped_status"),
    ),
    ScenarioTemplate(
        kind="trapped_fragment",
        prompt_focus="Caller turn is a short explicit trapped update like yes I am trapped or we are stuck inside.",
        prior_facts=_base_facts(
            location_candidate="Delta Campus building",
            location_kind="place_name",
            inside_building="yes",
        ),
        required_fields={"trapped_status": "yes"},
        required_cues=("trapped",),
        word_min=2,
        word_max=8,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind", "inside_building"),
    ),
    ScenarioTemplate(
        kind="bleeding_fragment",
        prompt_focus="Caller turn is a short explicit bleeding update like he is bleeding or no bleeding now. It must update bleeding only if the wording makes it explicit.",
        prior_facts=_base_facts(location_candidate="A100 near Ostkreuz", location_kind="road_or_junction"),
        required_fields={},
        word_min=2,
        word_max=8,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind"),
    ),
    ScenarioTemplate(
        kind="breathing_fragment",
        prompt_focus="Caller turn is a short explicit breathing update like not breathing or she is breathing now.",
        prior_facts=_base_facts(location_candidate="A100 near Ostkreuz", location_kind="road_or_junction"),
        required_fields={},
        word_min=2,
        word_max=8,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind"),
    ),
    ScenarioTemplate(
        kind="count_fragment",
        prompt_focus="Caller turn is a short explicit count update like there are two of us or just me and my daughter.",
        prior_facts=_base_facts(location_candidate="Delta Campus building", location_kind="place_name", inside_building="yes"),
        required_fields={},
        word_min=3,
        word_max=10,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind", "inside_building"),
    ),
    ScenarioTemplate(
        kind="no_change_ack",
        prompt_focus="Caller turn is a pure ack like okay or yes. It must not change any facts at all.",
        prior_facts=_base_facts(
            location_candidate="Delta Campus building",
            location_kind="place_name",
            inside_building="yes",
            issue_cues=["trapped", "child_present"],
            trapped_status="yes",
            child_present="yes",
            victim_count=2,
        ),
        required_fields={},
        word_min=1,
        word_max=3,
        location_rule="no_dispatchable",
        preserve_prior_fields=(
            "location_candidate",
            "location_kind",
            "inside_building",
            "issue_cues",
            "trapped_status",
            "child_present",
            "victim_count",
        ),
    ),
    ScenarioTemplate(
        kind="clear_fire_to_smoke",
        prompt_focus="Caller corrects an earlier fire claim into smoke only. Keep smoke if explicit, but remove fire if caller clearly says there are no flames.",
        prior_facts=_base_facts(
            location_candidate="Mullerstrasse 128, Berlin",
            location_kind="exact_address",
            issue_cues=["fire", "smoke"],
        ),
        required_fields={},
        required_cues=("smoke",),
        forbidden_cues=("fire",),
        word_min=6,
        word_max=20,
        location_rule="no_dispatchable",
        preserve_prior_fields=("location_candidate", "location_kind"),
    ),
    ScenarioTemplate(
        kind="indoor_place_with_sub_location",
        prompt_focus="Caller gives a building or place plus a floor or room clue in the same turn.",
        prior_facts=_base_facts(),
        required_fields={"location_kind": "place_name", "address_in_utterance": True, "inside_building": "yes"},
        word_min=8,
        word_max=24,
        location_rule="dispatchable",
    ),
)

LANGUAGE_STYLES = (
    "plain english",
    "short clipped english",
    "mixed english and german street naming",
    "slightly panicked but still explicit",
)

def _materialize_prior_facts(template: ScenarioTemplate, lexicon: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if template.kind == "confirm_existing_location":
        place_seed = sample_place_name_seed(lexicon, rng)
        return _base_facts(
            location_candidate=place_seed["location_candidate"],
            location_kind="place_name",
            address_in_utterance=False,
            inside_building="yes",
            issue_cues=["trapped"],
            trapped_status="yes",
        )
    if template.kind == "reject_existing_location":
        address_seed = sample_exact_address_seed(lexicon, rng)
        return _base_facts(
            location_candidate=address_seed["location_candidate"],
            location_kind="exact_address",
            address_in_utterance=False,
            issue_cues=["trapped"],
            trapped_status="yes",
        )
    if template.kind in {"child_fragment", "trapped_fragment", "no_change_ack"}:
        place_seed = sample_place_name_seed(lexicon, rng)
        base = _base_facts(
            location_candidate=place_seed["location_candidate"],
            location_kind="place_name",
            inside_building="yes",
            trapped_status="yes",
            issue_cues=["trapped"],
        )
        if template.kind == "no_change_ack":
            base.update(
                {
                    "issue_cues": ["trapped", "child_present"],
                    "child_present": "yes",
                    "victim_count": 2,
                }
            )
        return base
    if template.kind in {"bleeding_fragment", "breathing_fragment"}:
        road_seed = sample_road_or_junction_seed(lexicon, rng)
        return _base_facts(
            location_candidate=road_seed["location_candidate"],
            location_kind="road_or_junction",
        )
    if template.kind == "count_fragment":
        place_seed = sample_place_name_seed(lexicon, rng)
        return _base_facts(
            location_candidate=place_seed["location_candidate"],
            location_kind="place_name",
            inside_building="yes",
        )
    if template.kind == "clear_fire_to_smoke":
        address_seed = sample_exact_address_seed(lexicon, rng)
        return _base_facts(
            location_candidate=address_seed["location_candidate"],
            location_kind="exact_address",
            issue_cues=["fire", "smoke"],
        )
    return template.prior_facts


def _build_seed_payload(template: ScenarioTemplate, lexicon: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if template.kind == "explicit_exact_address":
        return sample_exact_address_seed(lexicon, rng)
    if template.kind == "explicit_place_name":
        return sample_place_name_seed(lexicon, rng)
    if template.kind == "road_or_junction":
        return sample_road_or_junction_seed(lexicon, rng)
    if template.kind == "landmark_note_only":
        return {
            "landmark_note": sample_landmark_note(lexicon, rng),
            "issue_seed": rng.choice(("smoke", "injury", "trapped")),
        }
    if template.kind == "dispatchable_with_note":
        base = rng.choice((sample_place_name_seed(lexicon, rng), sample_road_or_junction_seed(lexicon, rng)))
        base["landmark_note"] = sample_landmark_note(lexicon, rng)
        return base
    if template.kind == "vague_home_reject":
        return {"vague_location": sample_vague_location(lexicon, rng)}
    if template.kind == "sub_location_only":
        return {"sub_location_seed": sample_sub_location(lexicon, rng)}
    if template.kind == "indoor_place_with_sub_location":
        base = sample_place_name_seed(lexicon, rng)
        base["sub_location_seed"] = sample_sub_location(lexicon, rng)
        return base
    return {}


def build_plan(index: int, lexicon: dict[str, Any], *, seed: int = 11) -> ScenarioTemplate:
    rng = random.Random(seed + index)
    template = SCENARIO_TEMPLATES[index % len(SCENARIO_TEMPLATES)]
    materialized = replace(
        template,
        prior_facts=_materialize_prior_facts(template, lexicon, rng),
        seed_payload=_build_seed_payload(template, lexicon, rng),
        language_style=rng.choice(LANGUAGE_STYLES),
    )
    if materialized.kind == "dispatchable_with_note":
        location_kind = materialized.seed_payload["location_kind"]
        materialized = replace(materialized, required_fields={**materialized.required_fields, "location_kind": location_kind})
    if materialized.kind == "indoor_place_with_sub_location":
        materialized = replace(
            materialized,
            required_fields={
                **materialized.required_fields,
                "sub_location": materialized.seed_payload["sub_location_seed"],
            },
            required_nonempty_fields=tuple(
                dict.fromkeys((*materialized.required_nonempty_fields, "location_candidate", "sub_location"))
            ),
        )
    if materialized.kind == "sub_location_only":
        materialized = replace(
            materialized,
            required_fields={**materialized.required_fields, "sub_location": materialized.seed_payload["sub_location_seed"]},
        )
    return materialized


def example_prompt(plan: ScenarioTemplate, index: int) -> str:
    bleeding_rule = ""
    breathing_rule = ""
    count_rule = ""
    note_rule = ""
    if plan.kind == "bleeding_fragment":
        bleeding_rule = "- Make bleeding_status become either yes or no explicitly in merged_facts.\n- If bleeding is explicit, include the bleeding issue cue. If caller explicitly says no bleeding, do not include the bleeding cue."
    if plan.kind == "breathing_fragment":
        breathing_rule = "- Make breathing_status become either yes or no explicitly in merged_facts.\n- If caller says not breathing, include breathing_problem or unconscious only if directly stated."
    if plan.kind == "count_fragment":
        count_rule = "- Make victim_count become an explicit integer like 1, 2, or 3 in merged_facts."
    if plan.kind in {"landmark_note_only", "dispatchable_with_note"}:
        note_rule = "- location_note must preserve a short literal landmark or relative clue from caller_turn.\n- Do not copy the whole caller turn into location_note.\n- location_note is support context, not the main geocodable address."
    seed_payload = json.dumps(plan.seed_payload or {}, ensure_ascii=False)
    return f"""
Generate ONE training example for a fact-only dispatch SLM.

Return strict JSON only with this shape:
{{
  "caller_turn": "string",
  "prior_facts": {{
    "location_candidate": "string or null",
    "location_kind": "exact_address|road_or_junction|place_name|sub_location_only|vague|none",
    "address_in_utterance": true,
    "location_note": "string or null",
    "sub_location": "string or null",
    "inside_building": "yes|no|unknown",
    "issue_cues": ["zero or more from {", ".join(FACT_ISSUE_CUES)}"],
    "victim_count": "integer or \\"unknown\\"",
    "child_present": "yes|no|unknown",
    "bleeding_status": "yes|no|unknown",
    "breathing_status": "yes|no|unknown",
    "consciousness_status": "yes|no|unknown",
    "trapped_status": "yes|no|unknown"
  }},
  "merged_facts": {{
    "location_candidate": "string or null",
    "location_kind": "exact_address|road_or_junction|place_name|sub_location_only|vague|none",
    "address_in_utterance": true,
    "location_note": "string or null",
    "sub_location": "string or null",
    "inside_building": "yes|no|unknown",
    "issue_cues": ["zero or more from {", ".join(FACT_ISSUE_CUES)}"],
    "victim_count": "integer or \\"unknown\\"",
    "child_present": "yes|no|unknown",
    "bleeding_status": "yes|no|unknown",
    "breathing_status": "yes|no|unknown",
    "consciousness_status": "yes|no|unknown",
    "trapped_status": "yes|no|unknown"
  }}
}}

Hard rules:
- This is a strict evidence-only fact ledger.
- Do not add safety, severity, threat, dispatch priority, or escalation.
- prior_facts must match exactly what is specified below.
- merged_facts must carry forward prior_facts unless caller_turn explicitly changes a field.
- Caller turn only. No dispatcher text.
- caller_turn can be as short as one word if the plan calls for it.
- address_in_utterance may be true only if caller_turn itself contains a dispatch-usable location clue.
- Generic phrases like `at home`, `inside the building`, `somewhere in Berlin`, and `middle of the street` are not dispatch-usable locations.
- If caller_turn only confirms prior facts, keep them and set address_in_utterance=false unless the caller repeats the address in this turn.
- If caller_turn rejects prior location with no new one, clear location_candidate and use location_kind=none.
- If caller_turn only gives a floor/room/entrance clue, use location_kind=sub_location_only unless a building/place/address clue is also present in this turn.
- Use the seed payload below to anchor realistic Berlin wording when it is provided.
- Keep issue_cues explicit and literal, not inferred.
- Use this language style: {plan.language_style or LANGUAGE_STYLES[0]}
- Turn length target: {plan.word_min} to {plan.word_max} words.
- Scenario kind: {plan.kind}
- Focus: {plan.prompt_focus}
- Seed id: {index + 1}

prior_facts must be exactly:
{json.dumps(plan.prior_facts, ensure_ascii=False)}

seed payload:
{seed_payload}

merged_facts hard requirements:
{json.dumps(plan.required_fields, ensure_ascii=False)}

fields that must be present and non-empty:
{json.dumps(list(plan.required_nonempty_fields), ensure_ascii=False)}

required issue cues:
{json.dumps(list(plan.required_cues), ensure_ascii=False)}

forbidden issue cues:
{json.dumps(list(plan.forbidden_cues), ensure_ascii=False)}

location rule:
- {plan.location_rule}

preserve these fields from prior_facts unless caller_turn explicitly changes them:
{json.dumps(list(plan.preserve_prior_fields), ensure_ascii=False)}

{bleeding_rule}
{breathing_rule}
{count_rule}
{note_rule}
""".strip()


def repair_prompt(plan: ScenarioTemplate, candidate: dict[str, Any], errors: list[str]) -> str:
    return f"""
Repair this JSON training example so it satisfies every rule.

Plan kind: {plan.kind}
Focus: {plan.prompt_focus}
prior_facts must remain exactly:
{json.dumps(plan.prior_facts, ensure_ascii=False)}

seed payload:
{json.dumps(plan.seed_payload or {}, ensure_ascii=False)}

Problems:
{json.dumps(errors, ensure_ascii=False)}

Current candidate:
{json.dumps(candidate, ensure_ascii=False)}

Return corrected JSON only with the same schema.
Keep the caller turn natural and realistic for live speech.
""".strip()


def call_openai_json(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You create strict fact-ledger emergency dispatch training data. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def validate_example(plan: ScenarioTemplate, candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        example = FactLedgerTrainingExample.model_validate(candidate)
    except Exception as exc:
        return [str(exc)]

    prior = example.prior_facts
    merged = example.merged_facts
    word_count = _word_count(example.caller_turn)
    if not (plan.word_min <= word_count <= plan.word_max):
        errors.append(f"word_count {word_count} not in range {plan.word_min}-{plan.word_max}")
    if prior.model_dump() != plan.prior_facts:
        errors.append("prior_facts do not match the requested plan exactly")

    for field, value in plan.required_fields.items():
        if getattr(merged, field) != value:
            errors.append(f"merged_facts.{field} expected {value!r} got {getattr(merged, field)!r}")

    for field in plan.required_nonempty_fields:
        value = getattr(merged, field)
        if value in (None, "", [], "unknown"):
            errors.append(f"merged_facts.{field} must be present and non-empty")

    for cue in plan.required_cues:
        if cue not in merged.issue_cues:
            errors.append(f"missing required issue cue {cue}")
    for cue in plan.forbidden_cues:
        if cue in merged.issue_cues:
            errors.append(f"forbidden issue cue present: {cue}")

    for field in plan.preserve_prior_fields:
        if field in plan.required_fields:
            continue
        if getattr(merged, field) != getattr(prior, field):
            errors.append(f"expected merged_facts.{field} to preserve prior value")

    if plan.location_rule == "dispatchable":
        if not has_dispatchable_location_cue(example.caller_turn):
            errors.append("expected a dispatch-usable location clue in caller_turn")
        if not merged.address_in_utterance:
            errors.append("address_in_utterance should be true for dispatchable location examples")
        if merged.location_candidate is None:
            errors.append("location_candidate should be populated for dispatchable location examples")
    elif plan.location_rule == "no_dispatchable":
        if has_dispatchable_location_cue(example.caller_turn):
            errors.append("caller_turn should not contain a dispatch-usable location clue")
        if merged.address_in_utterance:
            errors.append("address_in_utterance should be false when no dispatchable location is spoken")
    elif plan.location_rule == "sub_only":
        if has_dispatchable_location_cue(example.caller_turn):
            errors.append("sub-location-only turn should not contain a dispatchable location clue")
        if merged.location_kind != "sub_location_only":
            errors.append("sub-location-only plan must keep location_kind=sub_location_only")
        if not merged.sub_location:
            errors.append("sub-location-only plan must populate sub_location")

    if plan.kind == "no_change_ack" and merged.model_dump() != prior.model_dump():
        errors.append("no_change_ack should leave merged_facts identical to prior_facts")
    if plan.kind == "confirm_existing_location" and merged.location_candidate != prior.location_candidate:
        errors.append("confirm_existing_location must preserve prior location_candidate")
    if plan.kind == "reject_existing_location" and merged.location_candidate is not None:
        errors.append("reject_existing_location must clear location_candidate")
    if plan.kind == "bleeding_fragment" and merged.bleeding_status == "unknown":
        errors.append("bleeding_fragment must explicitly resolve bleeding_status")
    if plan.kind == "breathing_fragment" and merged.breathing_status == "unknown":
        errors.append("breathing_fragment must explicitly resolve breathing_status")
    if plan.kind == "count_fragment" and merged.victim_count == "unknown":
        errors.append("count_fragment must explicitly resolve victim_count")
    if plan.kind == "landmark_note_only":
        if merged.location_candidate is not None:
            errors.append("landmark_note_only must not populate location_candidate")
    if plan.kind in {"landmark_note_only", "dispatchable_with_note"} and merged.location_note:
        if merged.location_note.lower() == example.caller_turn.lower():
            errors.append("location_note should be a short extracted clue, not the full caller turn")
    return errors


def generate_one(client: OpenAI, model: str, plan: ScenarioTemplate, index: int) -> dict[str, Any]:
    candidate = call_openai_json(client, model, example_prompt(plan, index))
    raw_candidate = candidate
    for attempt in range(MAX_RETRIES):
        errors = validate_example(plan, candidate)
        if not errors:
            return FactLedgerTrainingExample.model_validate(candidate).model_dump()
        if attempt == MAX_RETRIES - 1:
            raise ValueError("; ".join(errors))
        candidate = call_openai_json(client, model, repair_prompt(plan, raw_candidate, errors))
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
        description="Generate fact-ledger SLM training data focused on hard facts and address certainty."
    )
    parser.add_argument("--count", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=11, help="Base random seed.")
    parser.add_argument("--berlin-geojson", type=Path, default=DEFAULT_BERLIN_GEOJSON, help="Berlin address GeoJSON source.")
    parser.add_argument("--berlin-cache", type=Path, default=DEFAULT_BERLIN_LEXICON_CACHE, help="Cached compact Berlin lexicon JSON.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    lexicon = load_or_build_berlin_location_lexicon(
        args.berlin_geojson,
        cache_path=args.berlin_cache,
        seed=args.seed,
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index in range(args.count):
        plan = build_plan(index, lexicon, seed=args.seed)
        slot_error: str | None = None
        for slot_attempt in range(1, MAX_SLOT_ATTEMPTS + 1):
            try:
                record = generate_one(client, args.model, plan, index)
                records.append(record)
                print(
                    f"[{index + 1:>4}/{args.count}] ok  "
                    f"{plan.kind:<24} "
                    f"loc={record['merged_facts']['location_kind']:<16} "
                    f"(slot_try={slot_attempt})"
                )
                slot_error = None
                break
            except Exception as exc:  # pragma: no cover - runtime UX
                slot_error = str(exc)
        if slot_error:
            failures.append({"index": index, "plan": plan.kind, "error": slot_error})
            print(f"[{index + 1:>4}/{args.count}] err {plan.kind} :: {slot_error}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_path = args.out_dir / f"slm_fact_ledger_training_{stamp}.jsonl"
    pioneer_path = args.out_dir / f"pioneer_decoder_slm_fact_ledger_{stamp}.jsonl"
    failures_path = args.out_dir / f"slm_fact_ledger_failures_{stamp}.json"

    write_jsonl(dataset_path, records)
    pioneer_rows = [build_pioneer_fact_ledger_record(item) for item in records]
    write_jsonl(pioneer_path, pioneer_rows)
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "count_requested": args.count,
        "count_generated": len(records),
        "count_failed": len(failures),
        "model": args.model,
        "schema_version": FACT_LEDGER_SCHEMA_VERSION,
        "berlin_lexicon_cache_path": str(args.berlin_cache),
        "berlin_lexicon_stats": lexicon.get("stats"),
        "fact_ledger_dataset_path": str(dataset_path),
        "pioneer_decoder_dataset_path": str(pioneer_path),
        "failures_path": str(failures_path),
        "system_prompt_preview": FACT_LEDGER_SYSTEM_PROMPT[:240] + "...",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
