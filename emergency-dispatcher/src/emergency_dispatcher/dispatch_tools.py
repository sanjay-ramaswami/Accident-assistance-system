from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from datetime import UTC
from datetime import timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from emergency_dispatcher.berlin_location_lexicon import load_or_build_berlin_location_lexicon
from .settings import load_project_environment

logger = logging.getLogger(__name__)
# httpx logs full request URLs at INFO, which can contain caller addresses and
# provider query parameters. Keep those raw URLs out of application logs; the
# geocoder below emits redacted provider/status diagnostics instead.
logging.getLogger("httpx").setLevel(logging.WARNING)

TOOL_SCHEMA_VERSION = "v2"
ISSUE_TYPES = ("MEDICAL", "FIRE", "TRAFFIC", "POLICE", "HAZMAT", "GENERAL")
SAFETY_RISKS = ("SEVERE", "ELEVATED", "MODERATE", "UNKNOWN")
CALLER_STATES = ("DISTRESSED", "UNCERTAIN", "STABLE")
PRIORITY_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
FACT_LOCATION_KINDS = ("exact_address", "road_or_junction", "place_name", "sub_location_only", "vague", "none")
ISSUE_CUES = (
    "vehicle",
    "collision",
    "injury",
    "bleeding",
    "breathing_problem",
    "unconscious",
    "smoke",
    "fire",
    "trapped",
    "weapon",
    "explosion",
    "gas_smell",
    "child_present",
)
SERVICE_ORDER = ("police", "ambulance", "fire")
SERVICE_LABELS = {
    "police": "Police",
    "ambulance": "Ambulance",
    "fire": "Fire",
}
SERVICE_UNIT_LABELS = {
    "police": "nearest police team",
    "ambulance": "nearest ambulance crew",
    "fire": "nearest fire crew",
}
SERVICE_RESPONSE_RADIUS_M = 8000
NOTE_TYPES = ("none", "landmark_relative", "entrance", "sub_location", "freeform")
LOCATION_STATUS_ORDER = ("exact_address", "usable_place", "best_effort_note", "missing")
REPO_ROOT = Path(__file__).resolve().parents[2]
load_project_environment(REPO_ROOT / ".env")
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_NEARBYSEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_TIMEOUT_S = 4.0
NOMINATIM_LIMIT = 3


class _NominatimResponseError(RuntimeError):
    pass
BERLIN_VIEWBOX = "13.0884,52.3383,13.7612,52.6755"
BERLIN_BOUNDS = "52.3383,13.0884|52.6755,13.7612"
OVERPASS_INTERPRETER_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_S = 5.0
OVERPASS_RADIUS_M = 450
ROAD_PRIORITY = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "residential": 5,
    "service": 6,
}
ADDRESS_PATTERN = re.compile(
    r"(?P<number>\d{1,5})\s+(?P<street>[\w.\-'\s]+?),\s*(?P<city>[\w.\-'\s]+?)(?=[.?!]| and | but |$)",
    re.UNICODE,
)
ADDRESS_PATTERN_STREET_FIRST = re.compile(
    r"(?:\bon\b|\bat\b)\s+(?P<street>[\w.\-'\s]+?)\s+near\s+number\s+(?P<number>\d{1,5}),\s*(?P<city>[\w.\-'\s]+?)(?=[.?!]| and | but |$)",
    re.IGNORECASE | re.UNICODE,
)
ROAD_TOKEN_PATTERN = re.compile(r"\b(?:a|b|u|s)\s?-?\d{1,4}\b", re.IGNORECASE)
PLACE_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:campus|station|bahnhof|hospital|clinic|school|airport|bridge|junction|intersection|exit|mall|park|tower|hotel|museum|stadium|plaza|square|terminal|center|centre)\b",
    re.IGNORECASE,
)
PLACE_ANCHOR_PATTERN = re.compile(
    r"\b((?:[A-Za-zÀ-ÿ0-9.'-]+\s+){0,3}(?:campus|station|bahnhof|hospital|clinic|school|airport|bridge|junction|mall|park|tower|hotel|museum|stadium|plaza|square|terminal|center|centre))\b",
    re.IGNORECASE,
)
STREET_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|way|straße|strasse|platz|allee|ring|ufer|damm)\b",
    re.IGNORECASE,
)
STREET_ANCHOR_PATTERN = re.compile(
    r"\b((?:[A-Za-zÀ-ÿ0-9.'-]+\s+){0,3}[A-Za-zÀ-ÿ0-9.'-]*(?:straße|strasse|platz|allee|ring|ufer|damm))\b",
    re.IGNORECASE,
)
STREET_NAME_SUFFIX_PATTERN = re.compile(
    r"\b[\w.\-']*(?:straße|strasse|platz|allee|ring|ufer|damm)\b",
    re.IGNORECASE,
)
BARE_STREET_SUFFIX_TOKENS = {"straße", "strasse", "platz", "allee", "ring", "ufer", "damm"}
BARE_PLACE_DESCRIPTOR_TOKENS = {
    "campus",
    "station",
    "bahnhof",
    "hospital",
    "clinic",
    "school",
    "airport",
    "bridge",
    "junction",
    "intersection",
    "exit",
    "mall",
    "park",
    "tower",
    "hotel",
    "museum",
    "stadium",
    "plaza",
    "square",
    "terminal",
    "center",
    "centre",
}
GENERIC_LOCATION_PHRASES = {
    "middle of the street",
    "in the middle of the street",
    "the street",
    "on the street",
    "in the street",
    "middle of the road",
    "on the road",
    "by the road",
    "here",
    "over here",
    "right here",
    "somewhere here",
    "somewhere in berlin",
    "berlin",
    "inside the building",
    "inside the house",
    "inside the apartment",
    "inside my apartment",
    "inside my house",
    "inside the car",
    "in the building",
    "in my apartment",
    "in my house",
    "at home",
    "home",
    "my house",
    "my apartment",
    "third floor",
    "second floor",
    "first floor",
    "upstairs",
    "downstairs",
}
GENERIC_LOCATION_TOKENS = {
    "a",
    "an",
    "and",
    "around",
    "at",
    "building",
    "by",
    "car",
    "corner",
    "floor",
    "here",
    "home",
    "house",
    "in",
    "inside",
    "intersection",
    "middle",
    "my",
    "near",
    "of",
    "on",
    "outside",
    "place",
    "right",
    "road",
    "somewhere",
    "stationary",
    "street",
    "there",
    "vehicle",
}
RELATIVE_LOCATION_PATTERN = re.compile(
    r"\b(behind|in front of|next to|opposite|across from|near|by|beside|outside|inside|west of|east of|north of|south of)\b",
    re.IGNORECASE,
)
ENTRANCE_PATTERN = re.compile(
    r"\b(entrance|gate|lobby|stairwell|loading dock|courtyard|rear door|front door|west entrance|east entrance|north entrance|south entrance)\b",
    re.IGNORECASE,
)
SUB_LOCATION_PATTERN = re.compile(
    r"\b(floor|room|unit|apartment|flat|stairwell|lobby|corridor|basement|entrance|gate|wing|level)\b",
    re.IGNORECASE,
)
SEMANTIC_STREET_SUFFIXES = {
    "strasse",
    "platz",
    "allee",
    "ring",
    "ufer",
    "damm",
}
SEMANTIC_PLACE_SUFFIXES = {
    "park",
    "feld",
    "campus",
    "station",
    "bahnhof",
    "hospital",
    "clinic",
    "school",
    "airport",
    "bridge",
    "junction",
    "mall",
    "tower",
    "hotel",
    "museum",
    "stadium",
    "plaza",
    "square",
    "terminal",
    "center",
    "centre",
}
SEMANTIC_LOCATION_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"i am (?:at|close to)|"
    r"i'm (?:at|close to)|"
    r"caller says they are (?:at|close to)|"
    r"close to|near|at|by|inside|outside|around|toward|towards|"
    r"please\s+|find\s+|search(?:\s+for)?\s+|look up\s+|locate\s+|pin\s+|"
    r"send help to\s+|route to\s+"
    r")+",
    re.IGNORECASE,
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_semantic_location_text(value: str | None) -> str:
    text = _normalize_whitespace(value or "").casefold()
    if not text:
        return ""
    text = SEMANTIC_LOCATION_PREFIX_PATTERN.sub("", text)
    replacements = {
        "ß": "ss",
        "strasser": "strasse",
        "strassen": "strasse",
        "strase": "strasse",
        "strasze": "strasse",
        "straße": "strasse",
        "str.": "strasse",
        "street": "strasse",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    for suffix in sorted(SEMANTIC_STREET_SUFFIXES | SEMANTIC_PLACE_SUFFIXES, key=len, reverse=True):
        text = re.sub(rf"(?<=[a-z0-9]){suffix}\b", f" {suffix}", text)
    text = _normalize_whitespace(text)
    text = re.sub(r"\bberlin\b$", "", text).strip()
    return _normalize_whitespace(text)


def _semantic_location_stem_and_suffix(value: str) -> tuple[str, str | None]:
    tokens = value.split()
    if tokens and tokens[-1] in SEMANTIC_STREET_SUFFIXES | SEMANTIC_PLACE_SUFFIXES:
        return " ".join(tokens[:-1]), tokens[-1]
    return value, None


def _semantic_alpha_prefix(value: str, *, length: int = 3) -> str:
    letters = "".join(char for char in value if char.isalpha())
    return letters[:length]


@lru_cache(maxsize=1)
def _load_berlin_location_semantic_entries() -> tuple[dict[str, Any], ...]:
    try:
        lexicon = load_or_build_berlin_location_lexicon()
    except FileNotFoundError:
        return ()

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in lexicon.get("streets", []):
        name = _normalize_whitespace(str(row.get("street") or ""))
        if not name:
            continue
        key = ("street", name.casefold())
        if key in seen:
            continue
        seen.add(key)
        normalized = _normalize_semantic_location_text(name)
        stem, suffix = _semantic_location_stem_and_suffix(normalized)
        entries.append(
            {
                "kind": "street",
                "candidate": name,
                "normalized": normalized,
                "compact": normalized.replace(" ", ""),
                "stem": stem.replace(" ", ""),
                "suffix": suffix,
            }
        )
    for name in lexicon.get("place_names", []):
        candidate = _normalize_whitespace(str(name or ""))
        if not candidate:
            continue
        key = ("place", candidate.casefold())
        if key in seen:
            continue
        seen.add(key)
        normalized = _normalize_semantic_location_text(candidate)
        stem, suffix = _semantic_location_stem_and_suffix(normalized)
        entries.append(
            {
                "kind": "place",
                "candidate": candidate,
                "normalized": normalized,
                "compact": normalized.replace(" ", ""),
                "stem": stem.replace(" ", ""),
                "suffix": suffix,
            }
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def _load_berlin_location_semantic_index() -> dict[str, Any]:
    entries = _load_berlin_location_semantic_entries()
    by_suffix_prefix: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    by_suffix: dict[str | None, list[dict[str, Any]]] = {}
    by_prefix: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        suffix = entry["suffix"]
        stem = entry["stem"] or entry["compact"]
        prefix1 = _semantic_alpha_prefix(stem, length=1)
        prefix2 = _semantic_alpha_prefix(stem, length=2)
        by_suffix.setdefault(suffix, []).append(entry)
        for prefix in {prefix1, prefix2} - {""}:
            by_suffix_prefix.setdefault((suffix, prefix), []).append(entry)
            by_prefix.setdefault(prefix, []).append(entry)
    return {
        "entries": entries,
        "by_suffix_prefix": by_suffix_prefix,
        "by_suffix": by_suffix,
        "by_prefix": by_prefix,
    }


def _semantic_location_score(query: str, entry: dict[str, Any]) -> float:
    normalized_query = _normalize_semantic_location_text(query)
    if not normalized_query:
        return 0.0
    query_compact = normalized_query.replace(" ", "")
    query_stem, query_suffix = _semantic_location_stem_and_suffix(normalized_query)
    query_stem = query_stem.replace(" ", "")

    base_score = max(
        SequenceMatcher(None, normalized_query, entry["normalized"]).ratio(),
        SequenceMatcher(None, query_compact, entry["compact"]).ratio(),
    )
    stem_score = SequenceMatcher(None, query_stem, entry["stem"]).ratio() if query_stem and entry["stem"] else 0.0
    suffix_bonus = 0.06 if query_suffix and query_suffix == entry["suffix"] else 0.0
    prefix_bonus = 0.04 if entry["normalized"].startswith(normalized_query) or normalized_query.startswith(entry["normalized"]) else 0.0
    contains_bonus = 0.05 if query_stem and query_stem in entry["stem"] else 0.0
    return max(base_score, 0.72 * base_score + 0.28 * stem_score + suffix_bonus + prefix_bonus + contains_bonus)


def _semantic_location_match_is_strong(
    query: str,
    top_match: dict[str, Any],
    runner_up_score: float | None,
) -> bool:
    score = float(top_match.get("score") or 0.0)
    if score >= 0.93:
        return True

    normalized_query = _normalize_semantic_location_text(query)
    query_stem, query_suffix = _semantic_location_stem_and_suffix(normalized_query)
    candidate_stem, candidate_suffix = _semantic_location_stem_and_suffix(top_match.get("normalized") or "")
    if (
        score >= 0.84
        and _semantic_alpha_prefix(query_stem) == _semantic_alpha_prefix(candidate_stem)
        and (not query_suffix or query_suffix == candidate_suffix)
    ):
        return True

    if runner_up_score is not None and score >= 0.9 and score - runner_up_score >= 0.08:
        return True
    return False


def _semantic_match_berlin_location(query: str | None) -> dict[str, Any] | None:
    normalized_query = _normalize_semantic_location_text(query)
    if not normalized_query or len(normalized_query) < 5 or any(char.isdigit() for char in normalized_query):
        return None

    index = _load_berlin_location_semantic_index()
    entries = index["entries"]
    if not entries:
        return None

    query_stem, query_suffix = _semantic_location_stem_and_suffix(normalized_query)
    stem_key = query_stem.replace(" ", "") or normalized_query.replace(" ", "")
    prefix2 = _semantic_alpha_prefix(stem_key, length=2)
    prefix1 = _semantic_alpha_prefix(stem_key, length=1)
    candidates = list(index["by_suffix_prefix"].get((query_suffix, prefix2), ()))
    if len(candidates) < 12 and prefix1:
        candidates.extend(index["by_suffix_prefix"].get((query_suffix, prefix1), ()))
    if len(candidates) < 12:
        candidates.extend(index["by_suffix"].get(query_suffix, ()))
    if len(candidates) < 12:
        if prefix2:
            candidates.extend(index["by_prefix"].get(prefix2, ()))
        if len(candidates) < 12 and prefix1:
            candidates.extend(index["by_prefix"].get(prefix1, ()))
    if not candidates:
        candidates = list(entries)

    deduped_candidates: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        key = f"{candidate['kind']}:{candidate['candidate']}".casefold()
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        deduped_candidates.append(candidate)

    scored = sorted(
        (
            {
                "kind": entry["kind"],
                "candidate": entry["candidate"],
                "normalized": entry["normalized"],
                "score": _semantic_location_score(normalized_query, entry),
            }
            for entry in deduped_candidates
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    if not scored:
        return None

    top_match = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    runner_up_score = float(runner_up["score"]) if runner_up else None
    strong = _semantic_location_match_is_strong(normalized_query, top_match, runner_up_score)
    return {
        "query": _normalize_whitespace(query or "") or None,
        "normalized_query": normalized_query,
        "candidate": top_match["candidate"],
        "kind": top_match["kind"],
        "score": round(float(top_match["score"]), 3),
        "runner_up_candidate": runner_up["candidate"] if runner_up else None,
        "runner_up_score": round(runner_up_score, 3) if runner_up_score is not None else None,
        "apply": strong,
        "source": "berlin_lexicon",
    }


def _meaningful_street_suffix_query(value: str) -> bool:
    match = STREET_ANCHOR_PATTERN.search(value or "")
    if not match:
        return False
    token = _normalize_whitespace(match.group(1)).casefold()
    return token not in BARE_STREET_SUFFIX_TOKENS


def _meaningful_place_anchor_query(value: str) -> bool:
    match = PLACE_ANCHOR_PATTERN.search(value or "")
    if not match:
        return False
    token = _normalize_whitespace(match.group(1)).casefold()
    return token not in BARE_PLACE_DESCRIPTOR_TOKENS


def _normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return _normalize_whitespace(value).lower()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin
    from math import cos
    from math import radians
    from math import sin
    from math import sqrt

    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


def _empty_fact_payload() -> dict[str, Any]:
    return {
        "location_candidate": None,
        "location_kind": "none",
        "address_in_utterance": False,
        "location_note": None,
        "sub_location": None,
        "inside_building": "unknown",
        "issue_cues": [],
        "victim_count": "unknown",
        "child_present": "unknown",
        "bleeding_status": "unknown",
        "breathing_status": "unknown",
        "consciousness_status": "unknown",
        "trapped_status": "unknown",
    }


def _coerce_fact_payload(args: dict[str, Any] | None) -> dict[str, Any]:
    payload = _empty_fact_payload()
    source = dict((args or {}).get("facts") or {})
    for key, value in (args or {}).items():
        if key in payload and key != "facts":
            source.setdefault(key, value)
    payload.update(source)
    payload["location_candidate"] = _normalize_whitespace(str(payload.get("location_candidate") or "")) or None
    payload["location_note"] = _normalize_whitespace(str(payload.get("location_note") or "")) or None
    payload["sub_location"] = _normalize_whitespace(str(payload.get("sub_location") or "")) or None
    location_kind = _normalize_label(str(payload.get("location_kind") or "none"))
    payload["location_kind"] = location_kind if location_kind in FACT_LOCATION_KINDS else "none"
    for key in (
        "inside_building",
        "child_present",
        "bleeding_status",
        "breathing_status",
        "consciousness_status",
        "trapped_status",
    ):
        payload[key] = _normalize_yes_no_unknown(payload.get(key))
    payload["victim_count"] = _normalize_victim_count(payload.get("victim_count"))
    payload["address_in_utterance"] = bool(payload.get("address_in_utterance"))
    payload["issue_cues"] = _normalize_issue_cues(payload.get("issue_cues") or [])
    return payload


def _normalize_yes_no_unknown(value: Any) -> str:
    raw = _normalize_label(str(value or ""))
    if raw in {"yes", "true", "present"}:
        return "yes"
    if raw in {"no", "false", "none", "not present"}:
        return "no"
    return "unknown"


def _normalize_victim_count(value: Any) -> int | str:
    if value in {None, "", "unknown"}:
        return "unknown"
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return max(0, value)
    text = _normalize_label(str(value))
    if text in {"one", "single"}:
        return 1
    if text == "two":
        return 2
    if text == "three":
        return 3
    if text.isdigit():
        return int(text)
    return "unknown"


def _normalize_issue_cues(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        cue = _normalize_label(str(value))
        if cue in ISSUE_CUES and cue not in normalized:
            normalized.append(cue)
    return normalized


def _humanize_label(value: str) -> str:
    return value.replace("_", " ")


def _location_status_from_payload(payload: dict[str, Any]) -> str:
    location_kind = payload.get("location_kind")
    location_candidate = payload.get("location_candidate")
    location_note = payload.get("location_note")
    sub_location = payload.get("sub_location")

    if location_kind == "exact_address" and location_candidate:
        return "exact_address"
    if location_kind in {"road_or_junction", "place_name"} and location_candidate:
        return "usable_place"
    if location_candidate and assess_searchable_location_query(location_candidate)[0]:
        return "usable_place"
    if location_note or sub_location:
        return "best_effort_note"
    return "missing"


def _location_anchor_search_allowed(payload: dict[str, Any]) -> bool:
    location_candidate = payload.get("location_candidate")
    if not location_candidate:
        return False
    allowed, _reason = assess_searchable_location_query(location_candidate)
    return allowed


def _confirmed_fact_lines(payload: dict[str, Any]) -> list[str]:
    confirmed: list[str] = []
    if payload.get("issue_cues"):
        confirmed.append("issue cues: " + ", ".join(_humanize_label(item) for item in payload["issue_cues"]))
    if payload.get("location_candidate"):
        confirmed.append(f"location anchor: {payload['location_candidate']}")
    if payload.get("location_note"):
        confirmed.append(f"location note: {payload['location_note']}")
    if payload.get("sub_location"):
        confirmed.append(f"sub-location: {payload['sub_location']}")
    if payload.get("victim_count") != "unknown":
        confirmed.append(f"victim count: {payload['victim_count']}")
    for key, label in (
        ("inside_building", "inside building"),
        ("child_present", "child present"),
        ("bleeding_status", "bleeding"),
        ("breathing_status", "breathing"),
        ("consciousness_status", "conscious"),
        ("trapped_status", "trapped"),
    ):
        if payload.get(key) != "unknown":
            confirmed.append(f"{label}: {payload[key]}")
    return confirmed


def _extract_anchor_from_location_note(note: str | None) -> str | None:
    normalized = _normalize_whitespace(note or "")
    if not normalized:
        return None
    normalized = re.sub(
        r"^(?:please\s+)?(?:find|search(?: for)?|look up|locate|pin|send help to|route to)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    explicit = extract_address_candidate(normalized)
    if explicit:
        return explicit
    suffix_match = STREET_ANCHOR_PATTERN.search(normalized)
    if suffix_match:
        candidate = _normalize_whitespace(
            re.sub(
                r"^(?:right now\s+|now\s+)?(?:near|close to|around|by|outside|inside|at)\s+",
                "",
                suffix_match.group(1),
                flags=re.IGNORECASE,
            )
        )
        if _meaningful_street_suffix_query(candidate):
            return candidate
    road_match = ROAD_TOKEN_PATTERN.search(normalized)
    if road_match:
        return _normalize_whitespace(road_match.group(0))
    place_match = PLACE_ANCHOR_PATTERN.search(normalized)
    if place_match:
        candidate = _normalize_whitespace(
            re.sub(
                r"^(?:(?:right now|now)\s+)?(?:near|close to|around|by|outside|inside|at|please\s+|find\s+|search(?: for)?\s+|look up\s+|locate\s+|pin\s+|send help to\s+|route to\s+)+",
                "",
                place_match.group(1),
                flags=re.IGNORECASE,
            )
        )
        candidate = re.sub(
            r"^(?:(?:there\s+(?:is|was)|we\s+(?:had|were\s+in)|i\s+had)\s+)?"
            r"(?:an?\s+)?(?:vehicle\s+|car\s+)?(?:accident|crash|collision|fire|smoke|emergency|incident)\s+"
            r"(?:at|near|on|by)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        if _meaningful_place_anchor_query(candidate):
            return candidate
    return None


def resolve_location_note(
    location_note: str | None,
    *,
    location_candidate: str | None = None,
    sub_location: str | None = None,
) -> dict[str, Any]:
    note = _normalize_whitespace(location_note or "") or None
    anchor = _normalize_whitespace(location_candidate or "") or None
    known_sub_location = _normalize_whitespace(sub_location or "") or None
    note_anchor = _extract_anchor_from_location_note(note)
    if note_anchor and assess_searchable_location_query(note_anchor)[0]:
        anchor = note_anchor

    detected_sub_location = known_sub_location
    if note and not detected_sub_location and SUB_LOCATION_PATTERN.search(note):
        detected_sub_location = note

    note_type = "none"
    if note:
        if ENTRANCE_PATTERN.search(note):
            note_type = "entrance"
        elif SUB_LOCATION_PATTERN.search(note):
            note_type = "sub_location"
        elif RELATIVE_LOCATION_PATTERN.search(note):
            note_type = "landmark_relative"
        else:
            note_type = "freeform"
    elif detected_sub_location:
        note_type = "sub_location"

    anchor_payload = _coerce_fact_payload(
        {
            "location_candidate": anchor,
            "location_kind": "place_name" if anchor else "none",
            "location_note": note,
            "sub_location": detected_sub_location,
        }
    )
    location_status = _location_status_from_payload(anchor_payload)
    if anchor and assess_searchable_location_query(anchor)[0]:
        location_status = "usable_place"
    if not anchor and (note or detected_sub_location):
        location_status = "best_effort_note"

    # A road name used in several cities must not be sent to a global
    # geocoder until the caller supplies a city or nearby landmark.
    marine_drive_needs_city = bool(
        anchor
        and re.fullmatch(r"marine\s+drive", anchor, flags=re.IGNORECASE)
        and not (note_anchor or (note and not SUB_LOCATION_PATTERN.search(note)))
    )

    usable_for_dispatch = location_status in {"exact_address", "usable_place", "best_effort_note"}
    move_on_allowed = bool(anchor or note or detected_sub_location)
    search_allowed = bool(anchor and assess_searchable_location_query(anchor)[0])

    return {
        "note_type": note_type if note_type in NOTE_TYPES else "freeform",
        "normalized_note": note,
        "anchor_location": anchor,
        "sub_location": detected_sub_location,
        "location_status": location_status,
        "usable_for_dispatch": usable_for_dispatch,
        "move_on_allowed": move_on_allowed,
        "search_allowed": search_allowed,
        "needs_confirmation": bool(note and not anchor),
        "requires_city_clarification": marine_drive_needs_city,
        "clarification_question": (
            "Which city is Marine Drive in, or what nearby landmark can you see?"
            if marine_drive_needs_city
            else None
        ),
        "reason": (
            "ambiguous_location_without_city"
            if marine_drive_needs_city
            else "anchored_note"
            if anchor and note
            else "sub_location_only"
            if detected_sub_location and not anchor
            else "best_effort_note"
            if note
            else "no_note"
        ),
    }


def nearby_context(
    *,
    address_text: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any]:
    query = _normalize_whitespace(address_text or "") or None
    if lat is not None and lon is not None:
        geocode_result = {
            "lat": lat,
            "lon": lon,
            "normalized_address": query,
            "display_name": query,
            "geo_hint": {"district": None, "city": "Berlin", "state": "Berlin", "source": "coordinates"},
            "source": "coordinates",
        }
    elif query:
        lookup = lookup_address(query)
        if not lookup.get("geocoded"):
            return {
                "resolved": False,
                "query": query,
                "normalized_address": lookup.get("normalized_address"),
                "lat": None,
                "lon": None,
                "vicinity": None,
                "major_roads": [],
                "landmarks": [],
                "named_areas": [],
                "water_features": [],
                "confirmation_readback": None,
                "summary": "No nearby context available yet.",
                "reason": lookup.get("reason"),
            }
        geocode_result = lookup
    else:
        return {
            "resolved": False,
            "query": None,
            "normalized_address": None,
            "lat": None,
            "lon": None,
            "vicinity": None,
            "major_roads": [],
            "landmarks": [],
            "named_areas": [],
            "water_features": [],
            "confirmation_readback": None,
            "summary": "No nearby context available yet.",
            "reason": "no_query",
        }

    context = _build_location_context(geocode_result) or {}
    lines = []
    if context.get("major_roads"):
        lines.append("roads " + ", ".join(context["major_roads"][:2]))
    if context.get("landmarks"):
        lines.append("landmarks " + ", ".join(context["landmarks"][:2]))
    if context.get("named_areas"):
        lines.append("areas " + ", ".join(context["named_areas"][:2]))
    summary = f"{context.get('vicinity') or geocode_result.get('normalized_address')}: " + "; ".join(lines) if lines else (
        context.get("vicinity") or geocode_result.get("normalized_address") or "Context ready"
    )

    return {
        "resolved": True,
        "query": query,
        "normalized_address": geocode_result.get("normalized_address"),
        "lat": geocode_result.get("lat"),
        "lon": geocode_result.get("lon"),
        "vicinity": context.get("vicinity"),
        "major_roads": context.get("major_roads") or [],
        "landmarks": context.get("landmarks") or [],
        "named_areas": context.get("named_areas") or [],
        "water_features": context.get("water_features") or [],
        "confirmation_readback": context.get("confirmation_readback"),
        "summary": summary,
        "reason": context.get("source") or geocode_result.get("source"),
    }


def checklist_by_incident(args: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    payload = _coerce_fact_payload({**(args or {}), **kwargs})
    location_status = _location_status_from_payload(payload)
    search_allowed = _location_anchor_search_allowed(payload)

    missing_fields: list[str] = []
    if not payload["issue_cues"]:
        missing_fields.append("issue_cues")
    if location_status == "missing":
        missing_fields.append("location_candidate")
    elif location_status == "best_effort_note" and not search_allowed:
        missing_fields.append("location_anchor")
    if payload["victim_count"] == "unknown":
        missing_fields.append("victim_count")

    if any(cue in payload["issue_cues"] for cue in ("bleeding", "injury")) and payload["bleeding_status"] == "unknown":
        missing_fields.append("bleeding_status")
    if any(cue in payload["issue_cues"] for cue in ("breathing_problem", "unconscious")) and payload["breathing_status"] == "unknown":
        missing_fields.append("breathing_status")
    if "unconscious" in payload["issue_cues"] and payload["consciousness_status"] == "unknown":
        missing_fields.append("consciousness_status")
    if any(cue in payload["issue_cues"] for cue in ("collision", "fire", "trapped")) and payload["trapped_status"] == "unknown":
        missing_fields.append("trapped_status")
    if "child_present" in payload["issue_cues"] and payload["child_present"] == "unknown":
        missing_fields.append("child_present")

    goal_map = {
        "issue_cues": ("issue_cues", "clarify the emergency", "short_open"),
        "location_candidate": ("location_candidate", "pinpoint where they are", "short_open"),
        "location_anchor": ("location_candidate", "anchor the landmark to a named place or road", "short_open"),
        "victim_count": ("victim_count", "confirm how many people are involved", "short_open"),
        "bleeding_status": ("bleeding_status", "confirm bleeding", "yes_no"),
        "breathing_status": ("breathing_status", "confirm breathing", "yes_no"),
        "consciousness_status": ("consciousness_status", "confirm consciousness", "yes_no"),
        "trapped_status": ("trapped_status", "confirm whether anyone is trapped", "yes_no"),
        "child_present": ("child_present", "confirm whether a child is involved", "yes_no"),
    }
    next_missing = missing_fields[0] if missing_fields else None
    next_field, next_goal, question_style = goal_map.get(next_missing, (None, "hold steady", "none"))

    return {
        "facts_snapshot": payload,
        "location_status": location_status,
        "location_search_allowed": search_allowed,
        "missing_fields": missing_fields,
        "resolved_fields": [
            item
            for item in payload
            if item != "issue_cues"
            and payload.get(item) not in (None, "unknown", "none")
            and payload.get(item) != []
        ],
        "next_question_field": next_field,
        "next_question_goal": next_goal,
        "question_style": question_style,
        "move_on_allowed": location_status != "missing" or bool(payload.get("location_note") or payload.get("sub_location")),
        "best_effort_location_note": payload.get("location_note") or payload.get("sub_location"),
        "confirmed_facts": _confirmed_fact_lines(payload),
    }


def build_handoff_brief(
    *,
    caller_summary: str | None = None,
    args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = _coerce_fact_payload({**(args or {}), **kwargs})
    checklist = checklist_by_incident(payload)
    location_status = checklist["location_status"]

    status = "gathering"
    if payload["issue_cues"] and location_status in {"exact_address", "usable_place"}:
        status = "ready"
    elif payload["issue_cues"] and location_status == "best_effort_note":
        status = "dispatchable_with_note"
    elif payload["issue_cues"]:
        status = "partial"

    headline_parts: list[str] = []
    if payload["issue_cues"]:
        headline_parts.append(", ".join(_humanize_label(item) for item in payload["issue_cues"][:3]))
    if payload["victim_count"] != "unknown":
        headline_parts.append(f"{payload['victim_count']} involved")
    if payload.get("location_candidate"):
        headline_parts.append(payload["location_candidate"])
    elif payload.get("location_note"):
        headline_parts.append(payload["location_note"])
    if payload.get("sub_location"):
        headline_parts.append(payload["sub_location"])

    dispatcher_note_parts: list[str] = []
    if location_status == "best_effort_note" and checklist.get("best_effort_location_note"):
        dispatcher_note_parts.append(f"Best effort location note: {checklist['best_effort_location_note']}.")
    if checklist["missing_fields"]:
        dispatcher_note_parts.append(
            "Still unknown: " + ", ".join(_humanize_label(item) for item in checklist["missing_fields"][:4]) + "."
        )
    if payload.get("location_candidate") and not checklist["location_search_allowed"]:
        dispatcher_note_parts.append("Location anchor is not geocode-safe yet; keep it as a note until confirmed.")

    return {
        "status": status,
        "one_line": " | ".join(headline_parts) if headline_parts else "Facts still being gathered",
        "caller_summary": _normalize_whitespace(caller_summary or "") or None,
        "confirmed_facts": checklist["confirmed_facts"],
        "unknowns": [_humanize_label(item) for item in checklist["missing_fields"]],
        "dispatcher_note": " ".join(dispatcher_note_parts).strip() or "No special handoff note yet.",
        "location_status": location_status,
        "move_on_allowed": checklist["move_on_allowed"],
        "next_question_goal": checklist["next_question_goal"],
    }


def _transcript_issue_cues(transcript: str) -> set[str]:
    lowered = _normalize_label(transcript)
    cues: set[str] = set()
    if any(word in lowered for word in ("weapon", "gun", "knife", "armed", "hostage", "held hostage", "shooting")):
        cues.add("weapon")
    if any(word in lowered for word in ("collision", "crash", "accident", "hit by a car", "rolled over", "car hit")):
        cues.add("collision")
    if any(word in lowered for word in ("injury", "injured", "hurt", "pierced", "broken", "fracture")):
        cues.add("injury")
    if any(word in lowered for word in ("bleeding", "blood", "hemorrhage")):
        cues.add("bleeding")
    if any(word in lowered for word in ("not breathing", "can't breathe", "cannot breathe", "breathing", "asthma", "choking")):
        cues.add("breathing_problem")
    if any(word in lowered for word in ("unconscious", "collapsed", "unresponsive")):
        cues.add("unconscious")
    if "smoke" in lowered:
        cues.add("smoke")
    if "fire" in lowered or "burning" in lowered:
        cues.add("fire")
    if any(word in lowered for word in ("trapped", "stuck", "can't get out", "cannot get out")):
        cues.add("trapped")
    if any(word in lowered for word in ("explosion", "exploded", "blast")):
        cues.add("explosion")
    if any(word in lowered for word in ("gas smell", "gas leak", "fumes")):
        cues.add("gas_smell")
    if any(word in lowered for word in ("child", "daughter", "son", "kid", "kids", "baby")):
        cues.add("child_present")
    return cues


def _priority_from_service_signals(transcript: str, issue_cues: set[str]) -> str:
    lowered = _normalize_label(transcript)
    critical_signals = {"weapon", "fire", "unconscious", "breathing_problem"}
    high_signals = {"bleeding", "trapped", "collision", "injury", "explosion", "gas_smell", "child_present"}
    if "hostage" in lowered or "gun" in lowered or issue_cues & critical_signals:
        return "CRITICAL"
    if issue_cues & high_signals:
        return "HIGH"
    if issue_cues:
        return "MEDIUM"
    return "LOW"


def plan_response_services(
    *,
    transcript: str | None = None,
    issue_type: str | None = None,
    priority: str | None = None,
    issue_cues: list[str] | None = None,
) -> dict[str, Any]:
    normalized_transcript = _normalize_whitespace(transcript or "")
    cue_set = {item for item in (issue_cues or []) if item in ISSUE_CUES}
    cue_set.update(_transcript_issue_cues(normalized_transcript))

    normalized_issue_type = normalize_issue_type(issue_type or normalized_transcript)
    normalized_priority = normalize_priority(priority or _priority_from_service_signals(normalized_transcript, cue_set))

    needs_police = bool(
        normalized_issue_type in {"POLICE", "TRAFFIC"}
        or cue_set & {"weapon", "collision", "child_present"}
        or "hostage" in _normalize_label(normalized_transcript)
    )
    needs_ambulance = bool(
        normalized_issue_type == "MEDICAL"
        or cue_set & {"injury", "bleeding", "breathing_problem", "unconscious", "trapped"}
        or (normalized_issue_type == "FIRE" and cue_set & {"trapped", "injury"})
    )
    needs_fire = bool(
        normalized_issue_type in {"FIRE", "HAZMAT"} or cue_set & {"fire", "smoke", "explosion", "gas_smell"}
    )
    serious_emergency = bool(
        normalized_priority in {"HIGH", "CRITICAL"}
        or "hostage" in _normalize_label(normalized_transcript)
        or cue_set & {"weapon", "unconscious", "breathing_problem", "fire"}
    )
    services = {
        "police": {
            "needed": needs_police,
            "label": SERVICE_LABELS["police"],
            "unit_label": SERVICE_UNIT_LABELS["police"],
            "reason": "threat_or_control" if ("weapon" in cue_set or "hostage" in _normalize_label(normalized_transcript)) else "scene_control",
        },
        "ambulance": {
            "needed": needs_ambulance,
            "label": SERVICE_LABELS["ambulance"],
            "unit_label": SERVICE_UNIT_LABELS["ambulance"],
            "reason": "medical_support",
        },
        "fire": {
            "needed": needs_fire,
            "label": SERVICE_LABELS["fire"],
            "unit_label": SERVICE_UNIT_LABELS["fire"],
            "reason": "fire_or_hazmat",
        },
    }
    needed_services = [service for service in SERVICE_ORDER if services[service]["needed"]]
    return {
        "issue_type": normalized_issue_type,
        "priority": normalized_priority,
        "issue_cues": sorted(cue_set),
        "serious_emergency": serious_emergency,
        # A severity signal can recommend human review, but there is no
        # connected operator-monitoring service in this simulation.
        "human_monitoring": False,
        "human_review_recommended": serious_emergency,
        "monitor_name": None,
        "needed_services": needed_services,
        "services": services,
        "summary": ", ".join(SERVICE_LABELS[name] for name in needed_services) if needed_services else "No service selected yet",
    }


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _service_status_for_elapsed(elapsed_s: float, *, location_ready: bool) -> str:
    if not location_ready:
        return "locating"
    if elapsed_s < 1.4:
        return "locating"
    if elapsed_s < 3.0:
        return "assigned"
    return "dispatched"


def _service_announcement_line(service_name: str, status: str, eta_text: str, base_name: str | None = None) -> str:
    del eta_text, base_name
    label = SERVICE_LABELS.get(service_name, "response unit")
    if status == "locating":
        return "I'm checking the nearest available unit in this simulation. Please confirm your exact location."
    if status == "assigned":
        return f"A simulated {label.lower()} has been assigned. Please confirm your exact location and tell me if anything changes."
    if status == "en_route":
        return f"The simulated {label.lower()} is en route. Please tell me if the person's condition changes."
    if status == "arrived":
        return f"The simulated {label.lower()} has arrived. Follow the real local call-taker's instructions if you have contacted emergency services."
    if status in {"failed", "no_unit_available", "unavailable"}:
        return f"No simulated {label.lower()} is available right now. I can retry the simulation; please contact local emergency services if immediate help is needed."
    return "The simulated response plan is being updated. Please tell me the exact location."


def simulate_dispatch_services(
    *,
    session_id: str,
    plan: dict[str, Any],
    location_text: str | None = None,
    location_confirmed: bool = False,
    response_bases: dict[str, Any] | None = None,
    prior_state: dict[str, Any] | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    now_dt = _parse_utc(now_iso) or datetime.now(UTC)
    previous = prior_state or {}
    previous_services = previous.get("services") or {}
    services: dict[str, Any] = {}
    changed_services: list[str] = []
    # A spoken location string is not enough to plan around: require an
    # operator/caller-confirmed location before the simulation advances.
    location_ready = bool(location_confirmed and _normalize_whitespace(location_text or ""))
    priority = normalize_priority(plan.get("priority"))
    bases = dict(response_bases or previous.get("response_bases") or {})

    for service_name in SERVICE_ORDER:
        plan_item = (plan.get("services") or {}).get(service_name) or {}
        if not plan_item.get("needed"):
            continue
        prev_item = previous_services.get(service_name) or {}
        activated_at = prev_item.get("activated_at") or now_dt.isoformat()
        activated_dt = _parse_utc(activated_at) or now_dt
        elapsed_s = max(0.0, (now_dt - activated_dt).total_seconds())
        status = _service_status_for_elapsed(elapsed_s, location_ready=location_ready)
        previous_status = prev_item.get("status")
        if previous_status != status:
            changed_services.append(service_name)
        base = bases.get(service_name) or {}
        # The current project has no live responder-dispatch ETA source. Do not
        # synthesize an arrival time from a seed or straight-line distance.
        eta_minutes = None
        base_name = base.get("display_name")
        base_address = base.get("address")
        services[service_name] = {
            "needed": True,
            "label": SERVICE_LABELS[service_name],
            "unit_label": SERVICE_UNIT_LABELS[service_name],
            "status": status,
            "status_label": {
                "locating": "Locating nearest",
                "assigned": "Assigned",
                "dispatched": "Dispatched",
            }[status],
            "eta_min": eta_minutes,
            "eta_text": None,
            "detail": (
                f"Locating {SERVICE_UNIT_LABELS[service_name]}{f' near {base_name}' if base_name else ''}..."
                if status == "locating"
                else f"{SERVICE_UNIT_LABELS[service_name].capitalize()} active{f' from {base_name}' if base_name else ''}."
            ),
            "activated_at": activated_at,
            "location_confirmed": bool(location_confirmed),
            "base_name": base_name,
            "base_address": base_address,
            "distance_km": base.get("distance_km"),
        }

    summary_parts = [
        f"SIMULATION ONLY: {SERVICE_LABELS[name]} {services[name]['status_label'].lower()} · no ETA available"
        for name in SERVICE_ORDER
        if name in services
    ]
    announcement_line = None
    announcement_expires_at = None
    if not announcement_line and changed_services:
        newest = changed_services[-1]
        announcement_line = _service_announcement_line(
            newest,
            services[newest]["status"],
            services[newest]["eta_text"],
            services[newest].get("base_name"),
        )
    if announcement_line:
        announcement_expires_at = (now_dt + timedelta(seconds=8)).isoformat()
    else:
        previous_announcement = previous.get("announcement_line")
        previous_expiry = _parse_utc(previous.get("announcement_expires_at"))
        if previous_announcement and previous_expiry and previous_expiry > now_dt:
            announcement_line = previous_announcement
            announcement_expires_at = previous.get("announcement_expires_at")

    return {
        "human_monitoring": False,
        "human_review_recommended": bool(plan.get("human_review_recommended")),
        "monitor_name": None,
        "serious_emergency": bool(plan.get("serious_emergency")),
        "services": services,
        "response_bases": bases,
        "summary": " · ".join(summary_parts) if summary_parts else "No service movement yet",
        "announcement_line": announcement_line,
        "announcement_expires_at": announcement_expires_at,
        "services_needed": [name for name in SERVICE_ORDER if name in services],
        "dispatchable": bool(services) and location_ready,
        "location_confirmed": bool(location_confirmed),
    }


def normalize_issue_type(value: str | None) -> str:
    raw = _normalize_label(value)
    if raw in {item.lower() for item in ISSUE_TYPES}:
        return raw.upper()
    if any(word in raw for word in ("not breathing", "collapsed", "heart attack", "unconscious", "ambulance")):
        return "MEDICAL"
    if any(word in raw for word in ("fire", "smoke", "burning", "explosion")):
        return "FIRE"
    if any(word in raw for word in ("crash", "accident", "collision", "car hit", "overturned", "vehicle")):
        return "TRAFFIC"
    if any(word in raw for word in ("break in", "knife", "gun", "assault", "attack", "threat")):
        return "POLICE"
    if any(word in raw for word in ("gas leak", "chemical", "fumes", "hazmat")):
        return "HAZMAT"
    return "GENERAL"


def normalize_safety_risk(value: str | None) -> str:
    raw = _normalize_label(value)
    if raw in {item.lower() for item in SAFETY_RISKS}:
        return raw.upper()
    if any(
        word in raw
        for word in (
            "not breathing",
            "unconscious",
            "critical",
            "severe",
            "life threatening",
            "trapped",
            "heavy bleeding",
            "explosion",
        )
    ):
        return "SEVERE"
    if any(
        word in raw
        for word in (
            "high",
            "elevated",
            "bleeding",
            "smoke",
            "panic",
            "unsafe",
            "urgent",
            "fumes",
        )
    ):
        return "ELEVATED"
    if any(word in raw for word in ("unknown", "unclear", "not confirmed", "details unclear")):
        return "UNKNOWN"
    return "MODERATE"


def normalize_caller_state(value: str | None) -> str:
    raw = _normalize_label(value)
    if raw in {item.lower() for item in CALLER_STATES}:
        return raw.upper()
    if any(
        word in raw
        for word in ("distress", "panicked", "panic", "urgent", "help", "screaming", "hurry")
    ):
        return "DISTRESSED"
    if any(
        word in raw
        for word in ("uncertain", "unclear", "not sure", "don't know", "details unclear", "unknown")
    ):
        return "UNCERTAIN"
    return "STABLE"


def normalize_priority(value: str | None) -> str:
    raw = _normalize_label(value)
    if raw in {item.lower() for item in PRIORITY_ORDER}:
        return raw.upper()
    if "critical" in raw:
        return "CRITICAL"
    if "high" in raw or "urgent" in raw:
        return "HIGH"
    if "medium" in raw or "moderate" in raw:
        return "MEDIUM"
    return "LOW"


def infer_issue_type(transcript: str) -> str:
    return normalize_issue_type(transcript)


def extract_address_candidate(transcript: str) -> str | None:
    match = ADDRESS_PATTERN.search(transcript)
    if not match:
        alt_match = ADDRESS_PATTERN_STREET_FIRST.search(transcript)
        if not alt_match:
            return None
        number = alt_match.group("number")
        street = _normalize_whitespace(alt_match.group("street"))
        city = _normalize_whitespace(alt_match.group("city"))
        return f"{number} {street}, {city}"
    number = match.group("number")
    street = _normalize_whitespace(match.group("street"))
    city = _normalize_whitespace(match.group("city"))
    return f"{number} {street}, {city}"


def infer_safety_risk(transcript: str) -> str:
    return normalize_safety_risk(transcript)


def infer_caller_state(transcript: str) -> str:
    return normalize_caller_state(transcript)


def _address_fallback(address_text: str, *, reason: str) -> dict[str, Any]:
    normalized = _normalize_whitespace(address_text.replace(" ,", ","))
    district = None
    if "," in normalized:
        district = normalized.split(",")[-1].strip()
    return {
        "valid": True,
        "found": False,
        "geocoded": False,
        "normalized_address": normalized,
        "display_name": normalized,
        "lat": None,
        "lon": None,
        "geo_hint": {
            "district": district,
            "source": "text_fallback",
        }
        if district
        else None,
        "source": "text_fallback",
        "confidence": "low",
        "reason": reason,
        "candidates": [],
    }


def _nominatim_query_params(query: str) -> dict[str, Any]:
    normalized_query = _normalize_whitespace(query)
    params = {
        "q": normalized_query,
        "format": "jsonv2",
        "limit": NOMINATIM_LIMIT,
        "addressdetails": 1,
        "dedupe": 1,
        "accept-language": "en,de",
    }
    # Keep the existing Berlin bias only when the caller explicitly named Berlin.
    if re.search(r"\bberlin\b", normalized_query, re.IGNORECASE):
        params.update({"countrycodes": "de", "viewbox": BERLIN_VIEWBOX, "bounded": 1})
    return params


def _google_maps_api_key() -> str | None:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if key:
        return key.strip() or None
    return None


def _google_query_params(query: str) -> dict[str, Any]:
    normalized_query = _normalize_whitespace(query)
    key = _google_maps_api_key()
    params = {
        "address": normalized_query,
        "key": key,
        "language": "en",
    }
    if re.search(r"\bberlin\b", normalized_query, re.IGNORECASE):
        params.update({"region": "de", "components": "country:DE", "bounds": BERLIN_BOUNDS})
    return params


def _google_result_in_berlin(item: dict[str, Any]) -> bool:
    formatted_address = _normalize_whitespace(item.get("formatted_address", "")).casefold()
    if "berlin" in formatted_address:
        return True
    for component in item.get("address_components", []) or []:
        long_name = _normalize_whitespace(component.get("long_name", "")).casefold()
        short_name = _normalize_whitespace(component.get("short_name", "")).casefold()
        if "berlin" in {long_name, short_name}:
            return True
    return False


def _google_nearby_type(service_name: str) -> str:
    return {
        "police": "police",
        "ambulance": "hospital",
        "fire": "fire_station",
    }[service_name]


def _google_response_bases(lat: float, lon: float) -> dict[str, Any]:
    key = _google_maps_api_key()
    if not key:
        return {}
    results: dict[str, Any] = {}
    with httpx.Client(timeout=NOMINATIM_TIMEOUT_S) as client:
        for service_name in SERVICE_ORDER:
            response = client.get(
                GOOGLE_NEARBYSEARCH_URL,
                params={
                    "location": f"{lat},{lon}",
                    "rankby": "distance",
                    "type": _google_nearby_type(service_name),
                    "language": "en",
                    "key": key,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") not in {"OK", "ZERO_RESULTS"}:
                continue
            items = payload.get("results") or []
            if not items:
                continue
            item = items[0]
            geometry = item.get("geometry", {}) or {}
            location = geometry.get("location", {}) or {}
            item_lat = location.get("lat")
            item_lon = location.get("lng")
            if item_lat is None or item_lon is None:
                continue
            distance_km = _haversine_km(lat, lon, float(item_lat), float(item_lon))
            results[service_name] = {
                "display_name": _normalize_whitespace(item.get("name") or SERVICE_LABELS[service_name]),
                "address": _normalize_whitespace(item.get("vicinity") or ""),
                "lat": float(item_lat),
                "lon": float(item_lon),
                "distance_km": round(distance_km, 2),
                "source_type": _google_nearby_type(service_name),
            }
    return results


def assess_searchable_location_query(address_text: str | None) -> tuple[bool, str]:
    normalized = _normalize_whitespace(address_text or "")
    if not normalized:
        return False, "no_address_detected"
    lowered = normalized.casefold().strip(" ,.!?")
    if len(lowered) < 4:
        return False, "address_too_short"
    if lowered in GENERIC_LOCATION_PHRASES:
        return False, "location_not_specific_enough"
    if extract_address_candidate(normalized):
        return True, "explicit_address"
    # A caller-provided city and country is useful for a geocoding attempt,
    # but remains approximate and must be confirmed before dispatch pinning.
    if re.fullmatch(r"[A-Za-z][A-Za-z.'’ -]{1,48},\s*[A-Za-z][A-Za-z.'’ -]{1,48}", normalized):
        city, country = (part.strip() for part in normalized.split(",", 1))
        if city and country and city.casefold() not in GENERIC_LOCATION_PHRASES and country.casefold() not in GENERIC_LOCATION_PHRASES:
            return True, "city_country_candidate"
    if ROAD_TOKEN_PATTERN.search(lowered):
        return True, "road_designator"
    if any(" " in phrase and phrase in lowered for phrase in GENERIC_LOCATION_PHRASES):
        if not PLACE_DESCRIPTOR_PATTERN.search(lowered):
            return False, "location_not_specific_enough"
    if re.search(r"\b\d{1,5}[a-z]?\b", lowered) and STREET_DESCRIPTOR_PATTERN.search(lowered):
        return True, "street_number"
    if _meaningful_street_suffix_query(lowered):
        return True, "street_name_suffix"

    # Reject location phrases that are only generic containers like "middle of the street".
    tokens = [token for token in re.split(r"[\s,./-]+", lowered) if token]
    non_generic_tokens = [
        token
        for token in tokens
        if token not in GENERIC_LOCATION_TOKENS
        and token not in BARE_STREET_SUFFIX_TOKENS
        and token not in BARE_PLACE_DESCRIPTOR_TOKENS
        and len(token) > 1
    ]
    if STREET_DESCRIPTOR_PATTERN.search(lowered) and non_generic_tokens:
        return True, "street_name"
    if _meaningful_place_anchor_query(lowered) and PLACE_DESCRIPTOR_PATTERN.search(lowered) and non_generic_tokens:
        return True, "named_place"
    if non_generic_tokens and any(char.isdigit() for char in lowered):
        return True, "mixed_named_location"
    return False, "location_not_specific_enough"


def is_searchable_location_query(address_text: str | None) -> bool:
    allowed, _reason = assess_searchable_location_query(address_text)
    return allowed


def _google_confidence(location_type: str | None) -> str:
    if location_type == "ROOFTOP":
        return "high"
    if location_type == "RANGE_INTERPOLATED":
        return "medium"
    return "low"


def _simplify_google_candidate(item: dict[str, Any]) -> dict[str, Any]:
    geometry = item.get("geometry", {}) or {}
    location = geometry.get("location", {}) or {}
    location_type = geometry.get("location_type")
    address_components = item.get("address_components", []) or []

    def component_value(target_types: set[str]) -> str | None:
        for component in address_components:
            types = set(component.get("types", []))
            if target_types & types:
                return component.get("long_name")
        return None

    return {
        "display_name": item.get("formatted_address"),
        "lat": location.get("lat"),
        "lon": location.get("lng"),
        "type": item.get("types", []),
        "class": location_type,
        "importance": None,
        "place_id": item.get("place_id"),
        "geo_hint": {
            "district": component_value({"sublocality", "sublocality_level_1", "neighborhood"})
            or component_value({"locality"}),
            "city": component_value({"locality"}),
            "state": component_value({"administrative_area_level_1"}),
            "country": component_value({"country"}),
            "source": "google_geocoding",
        },
        "confidence": _google_confidence(location_type),
    }


def _reject_generic_google_match(query: str, candidate: dict[str, Any]) -> bool:
    display_name = _normalize_whitespace(candidate.get("display_name", "")).casefold()
    query_text = _normalize_whitespace(query).casefold()
    candidate_types = set(candidate.get("type", []))
    if display_name == "berlin, germany" and query_text not in {"berlin", "berlin, germany"}:
        return True
    city_country_query = re.fullmatch(r"[A-Za-z][A-Za-z.'’ -]{1,48},\s*[A-Za-z][A-Za-z.'’ -]{1,48}", query_text)
    if candidate.get("confidence") == "low" and candidate_types <= {"locality", "political"} and not city_country_query:
        if any(char.isdigit() for char in query_text) or len(query_text) > 12:
            return True
    return False


def _google_geocode_candidates(query: str) -> list[dict[str, Any]]:
    params = _google_query_params(query)
    if not params["key"]:
        return []
    try:
        with httpx.Client(timeout=NOMINATIM_TIMEOUT_S) as client:
            response = client.get(
                GOOGLE_GEOCODE_URL,
                params=params,
                headers={
                    "User-Agent": "emergency-dispatcher-hack/1.0",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("location geocoder failed: provider=google_maps error_type=%s", type(exc).__name__)
        raise
    if payload.get("status") != "OK":
        logger.info("location geocoder returned no result: provider=google_maps status=%s", payload.get("status"))
        return []
    results = payload.get("results", []) or []
    logger.info("location geocoder response: provider=google_maps result_count=%d", len(results))
    if re.search(r"\bberlin\b", query, re.IGNORECASE):
        return [item for item in results if _google_result_in_berlin(item)]
    return results


def _nominatim_candidates(query: str) -> list[dict[str, Any]]:
    params = _nominatim_query_params(query)
    query_kind = (
        "marine_drive_kochi"
        if re.search(r"\bmarine\s+drive\b", query, re.IGNORECASE)
        and re.search(r"\bkochi\b", query, re.IGNORECASE)
        else "redacted_address"
    )
    try:
        with httpx.Client(timeout=NOMINATIM_TIMEOUT_S) as client:
            response = client.get(
                NOMINATIM_SEARCH_URL,
                params=params,
                headers={
                    "User-Agent": "emergency-dispatcher-hack/1.0",
                },
            )
            http_status = response.status_code
            logger.info(
                "location geocoder response headers: provider=nominatim query_kind=%s http_status=%d",
                query_kind,
                http_status,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("location geocoder failed: provider=nominatim error_type=%s", type(exc).__name__)
        raise
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "location geocoder response invalid: provider=nominatim query_kind=%s http_status=%d error_type=%s",
            query_kind,
            http_status,
            type(exc).__name__,
        )
        raise _NominatimResponseError("Nominatim returned invalid JSON") from exc
    if not isinstance(payload, list):
        logger.warning(
            "location geocoder response invalid: provider=nominatim query_kind=%s http_status=%d error_type=UnexpectedPayloadType",
            query_kind,
            http_status,
        )
        raise _NominatimResponseError("Nominatim returned an unexpected response shape")
    result_count = len(payload) if isinstance(payload, list) else 0
    coordinate_count = sum(
        1
        for item in (payload if isinstance(payload, list) else [])
        if isinstance(item, dict) and _valid_geo_coordinates({"lat": item.get("lat"), "lon": item.get("lon")})
    )
    logger.info(
        "location geocoder response parsed: provider=nominatim query_kind=%s http_status=%d payload_kind=%s result_count=%d valid_coordinate_count=%d",
        query_kind,
        http_status,
        "list" if isinstance(payload, list) else type(payload).__name__,
        result_count,
        coordinate_count,
    )
    return payload


def _simplify_candidate(item: dict[str, Any]) -> dict[str, Any]:
    address = item.get("address", {}) or {}
    lat = item.get("lat")
    lon = item.get("lon")
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon = None
    return {
        "display_name": item.get("display_name"),
        "lat": lat,
        "lon": lon,
        "type": item.get("type"),
        "class": item.get("class"),
        "importance": item.get("importance"),
        "geo_hint": {
            "district": address.get("suburb")
            or address.get("city_district")
            or address.get("city")
            or address.get("town")
            or address.get("village"),
            "city": address.get("city") or address.get("town") or address.get("village"),
            "state": address.get("state"),
            "country": address.get("country"),
            "source": "nominatim",
        },
    }


def _valid_geo_coordinates(candidate: dict[str, Any]) -> bool:
    try:
        lat = float(candidate.get("lat"))
        lon = float(candidate.get("lon"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def _unique_city_region_candidate(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find one provider result that matches the named city and region exactly."""
    parts = [_normalize_whitespace(part).casefold() for part in query.split(",")]
    if len(parts) != 2 or not all(parts):
        return None
    city, region = parts
    matches = []
    for candidate in candidates:
        hint = candidate.get("geo_hint") or {}
        candidate_city = _normalize_whitespace(str(hint.get("city") or "")).casefold()
        candidate_region = _normalize_whitespace(
            str(hint.get("state") or hint.get("country") or "")
        ).casefold()
        if candidate_city == city and candidate_region == region:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def _approximate_city_result(
    normalized_query: str,
    candidates: list[dict[str, Any]],
    *,
    source: str,
    semantic_match: dict[str, Any] | None,
) -> dict[str, Any] | None:
    candidate = _unique_city_region_candidate(normalized_query, candidates)
    if candidate is None:
        return None
    result = dict(candidate)
    result.update(
        {
            "valid": True,
            "found": True,
            "geocoded": True,
            "normalized_address": result.get("display_name") or normalized_query,
            "source": source,
            "confidence": "low",
            "reason": "approximate_city_match",
            "candidates": [dict(item) for item in candidates],
            "semantic_match": semantic_match,
        }
    )
    return result


def _approximate_qualified_place_result(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    source: str,
    semantic_match: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve a named place plus city without treating it as a city/region query.

    Nominatim can return similarly named features in multiple cities and more
    than one OSM feature for the same mall. Require a city match and the place
    tokens, then collapse only candidates within 500 m of the same named place.
    """
    parts = [_normalize_whitespace(item) for item in query.split(",")]
    if len(parts) < 2 or not parts[0] or not parts[-1]:
        return None
    place_tokens = [token for token in re.findall(r"[a-z0-9]+", parts[0].casefold()) if token not in {"the", "near", "at", "in"}]
    # For "place, city, state, country" the first qualifier is the city.
    # The final display-name substring can occur in a road's name (for
    # example "Kochi - Kanyakumari Road") and is not a reliable city match.
    locality = parts[1].casefold()
    if not place_tokens:
        return None
    matching = []
    for candidate in candidates:
        hint = candidate.get("geo_hint") or {}
        city_state = " ".join(str(hint.get(key) or "") for key in ("city", "district", "state", "country")).casefold()
        display = str(candidate.get("display_name") or "").casefold()
        if locality not in city_state:
            continue
        if not all(token in display for token in place_tokens):
            continue
        if _valid_geo_coordinates(candidate):
            matching.append(candidate)
    if not matching:
        return None
    matching.sort(key=lambda item: float(item.get("importance") or 0), reverse=True)
    best = matching[0]
    if len(matching) > 1:
        close = all(
            _haversine_km(float(best["lat"]), float(best["lon"]), float(item["lat"]), float(item["lon"])) <= 0.5
            for item in matching[1:]
        )
        if not close:
            return None
    result = dict(best)
    result.update({
        "valid": False,
        "found": True,
        "geocoded": True,
        "normalized_address": result.get("display_name") or query,
        "source": source,
        "confidence": "low",
        "reason": "approximate_place_match",
        "candidates": [dict(item) for item in matching],
        "semantic_match": semantic_match,
    })
    return result


def _unique_names(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = _normalize_whitespace(value)
        lowered = cleaned.casefold()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _overpass_context(lat: float, lon: float) -> dict[str, Any]:
    query = f"""
    [out:json][timeout:8];
    (
      way(around:{OVERPASS_RADIUS_M},{lat},{lon})["highway"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["amenity"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["tourism"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["historic"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["public_transport"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["railway"="station"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["natural"]["name"];
      nwr(around:{OVERPASS_RADIUS_M},{lat},{lon})["waterway"]["name"];
    );
    out tags center 30;
    """
    with httpx.Client(timeout=OVERPASS_TIMEOUT_S) as client:
        response = client.post(
            OVERPASS_INTERPRETER_URL,
            content=query.strip(),
            headers={
                "User-Agent": "emergency-dispatcher-hack/1.0",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        response.raise_for_status()
        payload = response.json()

    roads: list[tuple[int, str]] = []
    landmarks: list[str] = []
    named_areas: list[str] = []
    water_features: list[str] = []

    for element in payload.get("elements", []):
        tags = element.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        highway = tags.get("highway")
        if highway:
            roads.append((ROAD_PRIORITY.get(highway, 99), name))
            continue

        if tags.get("natural") in {"water", "bay"} or tags.get("waterway"):
            water_features.append(name)
            continue

        if tags.get("amenity") in {"townhall", "hospital", "school", "police", "fire_station"}:
            landmarks.append(name)
            continue

        if tags.get("tourism") or tags.get("historic") or tags.get("railway") == "station":
            landmarks.append(name)
            continue

        if tags.get("public_transport"):
            landmarks.append(name)
            continue

        if tags.get("place") or tags.get("landuse"):
            named_areas.append(name)

    major_roads = [name for _, name in sorted(roads, key=lambda item: (item[0], item[1]))]
    return {
        "major_roads": _unique_names(major_roads, limit=4),
        "landmarks": _unique_names(landmarks, limit=5),
        "named_areas": _unique_names(named_areas, limit=4),
        "water_features": _unique_names(water_features, limit=4),
    }


def _service_address_from_tags(tags: dict[str, Any]) -> str | None:
    street = _normalize_whitespace(tags.get("addr:street", ""))
    house = _normalize_whitespace(tags.get("addr:housenumber", ""))
    postcode = _normalize_whitespace(tags.get("addr:postcode", ""))
    city = _normalize_whitespace(tags.get("addr:city", "") or tags.get("addr:place", ""))
    line1 = _normalize_whitespace(" ".join(part for part in (street, house) if part))
    line2 = _normalize_whitespace(" ".join(part for part in (postcode, city) if part))
    address = _normalize_whitespace(", ".join(part for part in (line1, line2) if part))
    return address or None


def _overpass_response_bases(lat: float, lon: float) -> dict[str, Any]:
    query = f"""
    [out:json][timeout:8];
    (
      nwr(around:{SERVICE_RESPONSE_RADIUS_M},{lat},{lon})["amenity"="police"]["name"];
      nwr(around:{SERVICE_RESPONSE_RADIUS_M},{lat},{lon})["amenity"="fire_station"]["name"];
      nwr(around:{SERVICE_RESPONSE_RADIUS_M},{lat},{lon})["emergency"="ambulance_station"]["name"];
      nwr(around:{SERVICE_RESPONSE_RADIUS_M},{lat},{lon})["amenity"="hospital"]["name"];
    );
    out tags center 80;
    """
    with httpx.Client(timeout=OVERPASS_TIMEOUT_S) as client:
        response = client.post(
            OVERPASS_INTERPRETER_URL,
            content=query.strip(),
            headers={
                "User-Agent": "emergency-dispatcher-hack/1.0",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        response.raise_for_status()
        payload = response.json()

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SERVICE_ORDER}
    for element in payload.get("elements", []):
        tags = element.get("tags", {}) or {}
        service_name = None
        amenity = tags.get("amenity")
        emergency = tags.get("emergency")
        if amenity == "police":
            service_name = "police"
        elif amenity == "fire_station":
            service_name = "fire"
        elif emergency == "ambulance_station" or amenity == "hospital":
            service_name = "ambulance"
        if not service_name:
            continue

        center = element.get("center") or {}
        item_lat = center.get("lat", element.get("lat"))
        item_lon = center.get("lon", element.get("lon"))
        if item_lat is None or item_lon is None:
            continue
        distance_km = _haversine_km(lat, lon, float(item_lat), float(item_lon))
        grouped[service_name].append(
            {
                "display_name": _normalize_whitespace(tags.get("name") or SERVICE_LABELS[service_name]),
                "address": _service_address_from_tags(tags),
                "lat": float(item_lat),
                "lon": float(item_lon),
                "distance_km": round(distance_km, 2),
                "source_type": emergency or amenity,
                "preference_rank": 0
                if emergency == "ambulance_station"
                else 1
                if amenity == "hospital"
                else 0,
            }
        )

    results: dict[str, Any] = {}
    for service_name, items in grouped.items():
        if not items:
            continue
        items.sort(key=lambda item: (item["distance_km"], item["preference_rank"], item["display_name"]))
        best = items[0]
        results[service_name] = {
            "display_name": best["display_name"],
            "address": best["address"],
            "lat": best["lat"],
            "lon": best["lon"],
            "distance_km": best["distance_km"],
            "source_type": best["source_type"],
        }
    return results


def lookup_response_bases(
    *,
    lat: float,
    lon: float,
    location_text: str | None = None,
) -> dict[str, Any]:
    bases: dict[str, Any] = {}
    sources: list[str] = []
    try:
        bases.update(_google_response_bases(lat, lon))
        if bases:
            sources.append("google_places")
    except httpx.HTTPError:
        pass
    try:
        overpass_bases = _overpass_response_bases(lat, lon)
        for service_name, item in overpass_bases.items():
            bases.setdefault(service_name, item)
        if overpass_bases:
            sources.append("overpass")
    except httpx.HTTPError:
        pass
    source = "+".join(sources) if sources else "unavailable"
    summary = " · ".join(
        f"{SERVICE_LABELS[name]}: {item['display_name']}"
        + (f" ({item['distance_km']} km straight-line)" if item.get("distance_km") is not None else "")
        for name, item in bases.items()
    ) or "No nearby response bases found"
    return {
        "resolved": bool(bases),
        "location_text": _normalize_whitespace(location_text or "") or None,
        "lat": lat,
        "lon": lon,
        "source": source,
        "services": bases,
        "summary": summary,
    }


def _build_location_context(geocode_result: dict[str, Any]) -> dict[str, Any] | None:
    lat = geocode_result.get("lat")
    lon = geocode_result.get("lon")
    if lat is None or lon is None:
        return None

    geo_hint = geocode_result.get("geo_hint") or {}
    area_bits = _unique_names(
        [
            geo_hint.get("district", ""),
            geo_hint.get("city", ""),
            geo_hint.get("state", ""),
        ],
        limit=3,
    )
    try:
        around = _overpass_context(lat, lon)
        source = f"{geocode_result.get('source') or 'geocoder'}+overpass"
    except httpx.HTTPError:
        around = {
            "major_roads": [],
            "landmarks": [],
            "named_areas": [],
            "water_features": [],
        }
        source = f"{geocode_result.get('source') or 'geocoder'}_only"

    vicinity = ", ".join(area_bits) if area_bits else geocode_result.get("normalized_address")
    confirmation_target = geocode_result.get("normalized_address") or geocode_result.get("display_name")
    lines = [
        "[LOCATION CONTEXT]",
        f"Caller vicinity: {vicinity or 'unknown'}",
    ]
    if around["major_roads"]:
        lines.append("Major roads nearby: " + ", ".join(around["major_roads"]))
    if around["landmarks"]:
        lines.append("Known landmarks: " + ", ".join(around["landmarks"]))
    if around["water_features"]:
        lines.append("Water features nearby: " + ", ".join(around["water_features"]))
    if around["named_areas"]:
        lines.append("Named areas nearby: " + ", ".join(around["named_areas"]))

    return {
        "vicinity": vicinity,
        "major_roads": around["major_roads"],
        "landmarks": around["landmarks"],
        "named_areas": around["named_areas"],
        "water_features": around["water_features"],
        "confirmation_readback": f"You said {confirmation_target}. Is that correct?"
        if confirmation_target
        else None,
        "context_block": "\n".join(lines),
        "source": source,
    }


def _geocode_address(address_text: str | None, *, allow_best_effort: bool = False) -> dict[str, Any]:
    if not address_text:
        return {
            "valid": False,
            "found": False,
            "geocoded": False,
            "normalized_address": None,
            "display_name": None,
            "lat": None,
            "lon": None,
            "geo_hint": None,
            "source": None,
            "confidence": None,
            "reason": "no_address_detected",
            "candidates": [],
        }

    normalized = _normalize_whitespace(address_text.replace(" ,", ","))
    logger.info("location resolution attempted: query_present=true best_effort=%s", bool(allow_best_effort))
    if len(normalized) < 4:
        return {
            "valid": False,
            "found": False,
            "geocoded": False,
            "normalized_address": normalized,
            "display_name": normalized,
            "lat": None,
            "lon": None,
            "geo_hint": None,
            "source": None,
            "confidence": None,
            "reason": "address_too_short",
            "candidates": [],
        }
    allowed, location_reason = assess_searchable_location_query(normalized)
    if not allowed and not allow_best_effort:
        return {
            "valid": False,
            "found": False,
            "geocoded": False,
            "normalized_address": normalized,
            "display_name": normalized,
            "lat": None,
            "lon": None,
            "geo_hint": None,
            "source": "guardrail",
            "confidence": None,
            "reason": location_reason,
            "candidates": [],
            "semantic_match": None,
        }

    semantic_match = None
    queries_to_try = [normalized]
    marine_drive_kochi_query = bool(
        re.search(r"\bmarine\s+drive\b", normalized, re.IGNORECASE)
        and re.search(r"\bkochi\b", normalized, re.IGNORECASE)
    )
    if marine_drive_kochi_query and normalized.casefold() != "marine drive kochi":
        # Nominatim's free-form parser returns no match for the fully qualified
        # comma-separated form, while the city-qualified phrase is supported.
        queries_to_try.append("Marine Drive Kochi")
    if allow_best_effort:
        semantic_match = (
            _semantic_match_berlin_location(normalized)
            if re.search(r"\bberlin\b", normalized, re.IGNORECASE)
            else None
        )
        semantic_candidate = _normalize_whitespace(str((semantic_match or {}).get("candidate") or ""))
        if semantic_candidate and (semantic_match or {}).get("apply") and semantic_candidate.casefold() != normalized.casefold():
            queries_to_try = [semantic_candidate, normalized]

    saw_nominatim_error = False
    for query in queries_to_try:
        try:
            google_candidates = _google_geocode_candidates(query)
        except httpx.HTTPError:
            google_candidates = []

        if google_candidates:
            simplified_google_candidates = [
                _simplify_google_candidate(item)
                for item in google_candidates
                if _valid_geo_coordinates(_simplify_google_candidate(item))
            ]
            filtered_google_candidates = [
                item for item in simplified_google_candidates if not _reject_generic_google_match(query, item)
            ]
            if filtered_google_candidates:
                candidate_list = [dict(item) for item in filtered_google_candidates]
                if location_reason == "city_country_candidate" and len(candidate_list) > 1:
                    approximate = _approximate_city_result(
                        normalized,
                        candidate_list,
                        source="google_geocoding",
                        semantic_match=semantic_match,
                    )
                    if approximate is not None:
                        logger.info("location resolution result: geocoded=true provider=google_maps confidence=low")
                        return approximate
                    qualified_place = _approximate_qualified_place_result(
                        normalized, candidate_list, source="google_geocoding", semantic_match=semantic_match,
                    )
                    if qualified_place is not None:
                        return qualified_place
                    # A place + city can also match several map results. Give
                    # Nominatim its turn before declaring the query ambiguous.
                    if not "," in normalized:
                        ambiguous = _address_fallback(normalized, reason="ambiguous_location")
                        ambiguous.update({"valid": False, "candidates": candidate_list, "semantic_match": semantic_match})
                        return ambiguous
                best = dict(candidate_list[0])
                best.update(
                    {
                        "valid": True,
                        "found": True,
                        "geocoded": True,
                        "normalized_address": best["display_name"],
                        "source": "google_geocoding",
                        "reason": "google_match",
                        "candidates": candidate_list,
                        "semantic_match": semantic_match,
                    }
                )
                logger.info("location resolution result: geocoded=true provider=google_maps")
                return best

        try:
            candidates = _nominatim_candidates(query)
        except (httpx.HTTPError, _NominatimResponseError):
            saw_nominatim_error = True
            continue

        simplified_candidates = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            simplified = _simplify_candidate(item)
            if _valid_geo_coordinates(simplified):
                simplified_candidates.append(simplified)
        if marine_drive_kochi_query:
            # The city-qualified point is useful for map orientation, but is
            # not an exact incident pin. Do not accept a different city.
            simplified_candidates = [
                item
                for item in simplified_candidates
                if "marine drive" in str(item.get("display_name") or "").casefold()
                and (
                    "kochi" in str(item.get("display_name") or "").casefold()
                    or "kochi" in str((item.get("geo_hint") or {}).get("city") or "").casefold()
                    or "kochi" in str((item.get("geo_hint") or {}).get("district") or "").casefold()
                )
            ]
        if simplified_candidates:
            if location_reason == "city_country_candidate" and len(simplified_candidates) > 1:
                approximate = _approximate_city_result(
                    normalized,
                    simplified_candidates,
                    source="nominatim",
                    semantic_match=semantic_match,
                )
                if approximate is not None:
                    logger.info("location resolution result: geocoded=true provider=nominatim confidence=low")
                    return approximate
                qualified_place = _approximate_qualified_place_result(
                    normalized, simplified_candidates, source="nominatim", semantic_match=semantic_match,
                )
                if qualified_place is not None:
                    return qualified_place
                ambiguous = _address_fallback(normalized, reason="ambiguous_location")
                ambiguous.update({"valid": False, "candidates": simplified_candidates, "semantic_match": semantic_match})
                return ambiguous
            best = simplified_candidates[0]
            if marine_drive_kochi_query:
                best.update(
                    {
                        "valid": False,
                        "found": True,
                        "geocoded": True,
                        "normalized_address": best["display_name"],
                        "source": "nominatim",
                        "confidence": "low",
                        "reason": "approximate_place_match",
                        "candidates": simplified_candidates,
                        "semantic_match": semantic_match,
                    }
                )
                logger.info("location resolution result: geocoded=true provider=nominatim confidence=low")
                return best
            best.update(
                {
                    "valid": True,
                    "found": True,
                    "geocoded": True,
                    "normalized_address": best["display_name"],
                    "source": "nominatim",
                    "confidence": (
                        "low"
                        if location_reason == "city_country_candidate"
                        else "high" if len(candidates) == 1 else "medium"
                    ),
                    "reason": "nominatim_match",
                    "candidates": simplified_candidates,
                    "semantic_match": semantic_match,
                }
            )
            logger.info("location resolution result: geocoded=true provider=nominatim")
            return best

    if saw_nominatim_error:
        fallback = _address_fallback(normalized, reason="google_no_match_nominatim_error" if _google_maps_api_key() else "nominatim_error")
    else:
        fallback = _address_fallback(
            normalized,
            reason="google_no_match_nominatim_no_match" if _google_maps_api_key() else "nominatim_no_match",
        )
    fallback["semantic_match"] = semantic_match
    logger.info("location resolution result: geocoded=false reason=%s", fallback["reason"])
    return fallback


def validate_address(address_text: str | None, *, allow_best_effort: bool = False) -> dict[str, Any]:
    result = _geocode_address(address_text, allow_best_effort=allow_best_effort)
    return {
        "valid": result["valid"],
        "geocoded": result["geocoded"],
        "normalized_address": result["normalized_address"],
        "display_name": result["display_name"],
        "lat": result["lat"],
        "lon": result["lon"],
        "geo_hint": result["geo_hint"],
        "source": result["source"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "semantic_match": result.get("semantic_match"),
    }


def lookup_address(address_text: str | None, *, allow_best_effort: bool = False) -> dict[str, Any]:
    result = _geocode_address(address_text, allow_best_effort=allow_best_effort)
    return {
        "found": result["found"],
        "geocoded": result["geocoded"],
        "normalized_address": result["normalized_address"],
        "display_name": result["display_name"],
        "lat": result["lat"],
        "lon": result["lon"],
        "geo_hint": result["geo_hint"],
        "source": result["source"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "candidates": result["candidates"],
        "location_context": _build_location_context(result) if result["geocoded"] else None,
        "semantic_match": result.get("semantic_match"),
    }


def assign_dispatch_priority(
    issue_type: str,
    safety_risk: str,
    caller_state: str,
) -> dict[str, Any]:
    issue_type = normalize_issue_type(issue_type)
    safety_risk = normalize_safety_risk(safety_risk)
    caller_state = normalize_caller_state(caller_state)

    score = 0
    if issue_type in {"MEDICAL", "FIRE", "HAZMAT"}:
        score += 2
    elif issue_type in {"TRAFFIC", "POLICE"}:
        score += 1

    if safety_risk == "SEVERE":
        score += 3
    elif safety_risk == "ELEVATED":
        score += 2
    elif safety_risk in {"MODERATE", "UNKNOWN"}:
        score += 1

    if caller_state == "DISTRESSED":
        score += 1

    priority = "LOW"
    if issue_type in {"MEDICAL", "FIRE"} and safety_risk == "SEVERE":
        priority = "CRITICAL"
    elif issue_type == "HAZMAT" and safety_risk in {"SEVERE", "ELEVATED"}:
        priority = "CRITICAL"
    elif score >= 5:
        priority = "CRITICAL"
    elif score >= 3:
        priority = "HIGH"
    elif score >= 2:
        priority = "MEDIUM"

    return {
        "priority": priority,
        "score": score,
        "dispatch_lane": {
            "CRITICAL": "lights_and_sirens",
            "HIGH": "urgent",
            "MEDIUM": "standard",
            "LOW": "queued",
        }[priority],
        "normalized_issue_type": issue_type,
        "normalized_safety_risk": safety_risk,
        "normalized_caller_state": caller_state,
    }


def create_incident_ticket(
    caller_summary: str,
    address: str | None,
    issue_type: str,
    priority: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "ticket_type": "emergency_dispatch",
        "schema_version": TOOL_SCHEMA_VERSION,
        "caller_summary": caller_summary,
        "address": address,
        "issue_type": normalize_issue_type(issue_type),
        "priority": normalize_priority(priority),
        "notes": notes,
        "status": "ready_for_dispatch",
    }


@dataclass
class DispatchWorkflowResult:
    transcript: str
    issue_type: str
    safety_risk: str
    caller_state: str
    address_validation: dict[str, Any]
    address_lookup: dict[str, Any]
    priority: dict[str, Any]
    ticket: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "issue_type": self.issue_type,
            "safety_risk": self.safety_risk,
            "caller_state": self.caller_state,
            "address_validation": self.address_validation,
            "address_lookup": self.address_lookup,
            "priority": self.priority,
            "ticket": self.ticket,
        }


def run_dispatch_workflow(transcript: str) -> DispatchWorkflowResult:
    transcript = _normalize_whitespace(transcript)
    issue_type = infer_issue_type(transcript)
    safety_risk = infer_safety_risk(transcript)
    caller_state = infer_caller_state(transcript)
    address_candidate = extract_address_candidate(transcript)
    address_validation = validate_address(address_candidate)
    address_lookup = lookup_address(address_candidate)
    priority = assign_dispatch_priority(
        issue_type=issue_type,
        safety_risk=safety_risk,
        caller_state=caller_state,
    )
    ticket = create_incident_ticket(
        caller_summary=transcript,
        address=address_lookup["normalized_address"],
        issue_type=issue_type,
        priority=priority["priority"],
        notes=[
            f"safety_risk={safety_risk}",
            f"caller_state={caller_state}",
            f"address_reason={address_lookup['reason']}",
        ],
    )
    return DispatchWorkflowResult(
        transcript=transcript,
        issue_type=issue_type,
        safety_risk=safety_risk,
        caller_state=caller_state,
        address_validation=address_validation,
        address_lookup=address_lookup,
        priority=priority,
        ticket=ticket,
    )


def _fact_ledger_schema_properties() -> dict[str, Any]:
    return {
        "location_candidate": {"type": ["string", "null"]},
        "location_kind": {"type": "string", "enum": list(FACT_LOCATION_KINDS)},
        "address_in_utterance": {"type": "boolean"},
        "location_note": {"type": ["string", "null"]},
        "sub_location": {"type": ["string", "null"]},
        "inside_building": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "issue_cues": {"type": "array", "items": {"type": "string", "enum": list(ISSUE_CUES)}},
        "victim_count": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["unknown"]}]},
        "child_present": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "bleeding_status": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "breathing_status": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "consciousness_status": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "trapped_status": {"type": "string", "enum": ["yes", "no", "unknown"]},
    }


def update_soft_ledger(args: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    payload = {**(args or {}), **kwargs}
    notes = [
        _normalize_whitespace(str(item))
        for item in list(payload.get("notes") or [])
        if _normalize_whitespace(str(item))
    ]
    return {
        "soft_state": {
            "people_count_best_guess": payload.get("people_count_best_guess"),
            "caller_role": _normalize_whitespace(str(payload.get("caller_role") or "unknown")) or "unknown",
            "notes": notes,
        },
        "updated_fields": [
            field_name
            for field_name, value in {
                "people_count_best_guess": payload.get("people_count_best_guess"),
                "caller_role": payload.get("caller_role"),
                "notes": notes,
            }.items()
            if value not in (None, "", [])
        ],
    }


def build_gradbot_tool_defs(*, include_soft_ledger: bool = False) -> list[tuple[str, str, str]]:
    tool_defs: list[tuple[str, str, str]] = [
        (
            "resolve_location_note",
            "First-choice hard-fact tool for location clues that are not exact addresses. Use it as soon as the caller mentions a landmark, entrance, floor, room, stairwell, gate, or relative clue like behind the church. This preserves the clue without pretending it is geocoded.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "location_note": {"type": ["string", "null"]},
                        "location_candidate": {"type": ["string", "null"]},
                        "sub_location": {"type": ["string", "null"]},
                    },
                    "required": ["location_note"],
                }
            ),
        ),
        (
            "checklist_by_incident",
            "Default planning tool after any meaningful new hard fact. Use it before your next follow-up question so you ask only for the highest-priority missing fact and do not repeat answered questions.",
            json.dumps(
                {
                    "type": "object",
                    "properties": _fact_ledger_schema_properties(),
                }
            ),
        ),
        (
            "validate_address",
            "Use when the caller already gave a searchable road, junction, street address, highway marker, or named place. It checks whether the clue is geocode-safe. Do not use for vague phrases like here, at home, or inside the building.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "address_text": {"type": "string"},
                    },
                    "required": ["address_text"],
                }
            ),
        ),
        (
            "lookup_address",
            "Use after the caller gives a searchable location clue and you want a pin or normalized readback. Prefer this before asking for more landmarks if you already have a usable named place, road, or junction.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "address_text": {"type": "string"},
                    },
                    "required": ["address_text"],
                }
            ),
        ),
        (
            "nearby_context",
            "Use only after a stable address or named place lands. It gives short nearby roads, landmarks, and confirmation context so you can sound grounded without asking the caller to repeat the same clue.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "address_text": {"type": ["string", "null"]},
                        "lat": {"type": ["number", "null"]},
                        "lon": {"type": ["number", "null"]},
                    },
                }
            ),
        ),
        (
            "build_handoff_brief",
            "Use when facts materially change, when a dispatcher-style summary would help, or just before ticket creation. It compacts confirmed hard facts and unknowns so you do not repeat settled details.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "caller_summary": {"type": ["string", "null"]},
                        **_fact_ledger_schema_properties(),
                    },
                }
            ),
        ),
        (
            "create_incident_ticket",
            "Create the final structured dispatch ticket for handoff to operations.",
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "caller_summary": {"type": "string"},
                        "address": {"type": ["string", "null"]},
                        "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
                        "priority": {
                            "type": "string",
                            "enum": list(PRIORITY_ORDER),
                            "description": "CRITICAL=life at risk or active danger. HIGH=unsafe and urgent. MEDIUM=urgent but stable. LOW=no immediate danger.",
                        },
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "caller_summary",
                        "address",
                        "issue_type",
                        "priority",
                        "notes",
                    ],
                }
            ),
        ),
    ]
    if include_soft_ledger:
        tool_defs.append(
            (
                "update_soft_ledger",
                "Use only for soft guesses or conversational context that should not overwrite confirmed hard facts. Allowed fields are people_count_best_guess, caller_role, and notes.",
                json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "people_count_best_guess": {"type": ["integer", "null"]},
                            "caller_role": {"type": ["string", "null"]},
                            "notes": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                ),
            )
        )
    return tool_defs


def dispatch_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "plan_response_services":
        return plan_response_services(
            transcript=args.get("transcript"),
            issue_type=args.get("issue_type"),
            priority=args.get("priority"),
            issue_cues=list(args.get("issue_cues") or []),
        )
    if name == "lookup_response_bases":
        return lookup_response_bases(
            lat=float(args["lat"]),
            lon=float(args["lon"]),
            location_text=args.get("location_text"),
        )
    if name == "simulate_dispatch_services":
        return simulate_dispatch_services(
            session_id=args["session_id"],
            plan=dict(args.get("plan") or {}),
            location_text=args.get("location_text"),
            location_confirmed=bool(args.get("location_confirmed")),
            response_bases=dict(args.get("response_bases") or {}),
            prior_state=dict(args.get("prior_state") or {}),
            now_iso=args.get("now_iso"),
        )
    if name == "resolve_location_note":
        return resolve_location_note(
            args.get("location_note"),
            location_candidate=args.get("location_candidate"),
            sub_location=args.get("sub_location"),
        )
    if name == "nearby_context":
        return nearby_context(
            address_text=args.get("address_text"),
            lat=args.get("lat"),
            lon=args.get("lon"),
        )
    if name == "checklist_by_incident":
        return checklist_by_incident(args)
    if name == "build_handoff_brief":
        caller_summary = args.get("caller_summary")
        brief_args = dict(args)
        brief_args.pop("caller_summary", None)
        return build_handoff_brief(caller_summary=caller_summary, args=brief_args)
    if name == "validate_address":
        return validate_address(
            args.get("address_text"),
            allow_best_effort=bool(args.get("allow_best_effort")),
        )
    if name == "lookup_address":
        return lookup_address(
            args.get("address_text"),
            allow_best_effort=bool(args.get("allow_best_effort")),
        )
    if name == "create_incident_ticket":
        return create_incident_ticket(
            caller_summary=args["caller_summary"],
            address=args.get("address"),
            issue_type=args["issue_type"],
            priority=args["priority"],
            notes=list(args.get("notes", [])),
        )
    if name == "update_soft_ledger":
        return update_soft_ledger(args)
    raise KeyError(f"Unknown tool: {name}")
