from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


BERLIN_LEXICON_SCHEMA_VERSION = "2026-04-26.v1"
DEFAULT_BERLIN_GEOJSON = Path.home() / "Downloads" / "adressen_berlin_adressen_berlin_WGS84.geojson"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BERLIN_LEXICON_CACHE = ROOT / "data" / "lexicon" / "berlin_location_lexicon_2026-04-26.v1.json"

_PLACE_NAMES = (
    "Delta Campus",
    "Berlin Hauptbahnhof",
    "Zoo station",
    "Ostkreuz station",
    "Alexanderplatz station",
    "Hermannplatz",
    "Tempelhofer Feld",
    "Charite Campus Mitte",
    "Neukoelln Arcaden",
    "Mall of Berlin",
    "Suedkreuz station",
    "Gesundbrunnen station",
)

_ROAD_LABELS = (
    "corner of",
    "junction of",
    "near",
    "by the exit for",
)

_LANDMARK_NOTES = (
    "in front of the glass office tower",
    "behind the red brick church",
    "by the loading dock",
    "near the bus loop",
    "next to the DHL lockers",
    "behind the supermarket",
    "by the west entrance",
    "near the big yellow crane",
    "next to the fenced playground",
    "behind the petrol station",
    "by the canal bridge stairs",
    "near the blue shipping containers",
)

_SUB_LOCATIONS = (
    "third floor",
    "second floor",
    "ground floor lobby",
    "rear stairwell",
    "west entrance",
    "back courtyard",
    "room 204",
    "unit 3B",
    "basement corridor",
    "front gate",
)

_VAGUE_LOCATIONS = (
    "at home",
    "inside the building",
    "inside my apartment",
    "somewhere in Berlin",
    "in the middle of the street",
    "outside",
    "near my house",
)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _address_row_from_properties(properties: dict[str, Any]) -> dict[str, str] | None:
    street = _clean_text(properties.get("str_name"))
    house_number = _clean_text(properties.get("hnr"))
    if not street or not house_number:
        return None
    suffix = _clean_text(properties.get("hnr_zusatz")) or ""
    postcode = _clean_text(properties.get("plz")) or ""
    ort_name = _clean_text(properties.get("ort_name")) or ""
    bez_name = _clean_text(properties.get("bez_name")) or ""
    rendered_number = f"{house_number}{suffix}"
    canonical = f"{street} {rendered_number}, {postcode} Berlin".strip().replace(" ,", ",")
    return {
        "street": street,
        "house_number": rendered_number,
        "postcode": postcode,
        "ort_name": ort_name,
        "bez_name": bez_name,
        "canonical": canonical,
        "canonical_with_ort": f"{street} {rendered_number}, {postcode} Berlin {ort_name}".strip(),
        "canonical_with_bezirk": f"{street} {rendered_number}, {postcode} Berlin {bez_name}".strip(),
    }


def build_berlin_location_lexicon(
    source_path: Path,
    cache_path: Path = DEFAULT_BERLIN_LEXICON_CACHE,
    *,
    max_exact_addresses: int = 12000,
    seed: int = 23,
) -> dict[str, Any]:
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    rng = random.Random(seed)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    exact_addresses: list[dict[str, str]] = []
    street_rows: dict[str, dict[str, str]] = {}
    ort_names: set[str] = set()
    bez_names: set[str] = set()
    postcodes: set[str] = set()

    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        if properties.get("typ") != "Adresse":
            continue
        if properties.get("qualitaet") not in {"Qualitaet A", "Qualitaet B", "RBS"}:
            continue
        row = _address_row_from_properties(properties)
        if row is None:
            continue
        if len(exact_addresses) < max_exact_addresses:
            exact_addresses.append(row)
        else:
            slot = rng.randint(0, len(exact_addresses))
            if slot < max_exact_addresses:
                exact_addresses[slot] = row

        if row["street"] not in street_rows:
            street_rows[row["street"]] = {
                "street": row["street"],
                "postcode": row["postcode"],
                "ort_name": row["ort_name"],
                "bez_name": row["bez_name"],
            }
        if row["ort_name"]:
            ort_names.add(row["ort_name"])
        if row["bez_name"]:
            bez_names.add(row["bez_name"])
        if row["postcode"]:
            postcodes.add(row["postcode"])

    lexicon = {
        "schema_version": BERLIN_LEXICON_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "seed": seed,
        "exact_addresses": exact_addresses,
        "streets": list(street_rows.values()),
        "ort_names": sorted(ort_names),
        "bez_names": sorted(bez_names),
        "postcodes": sorted(postcodes),
        "place_names": list(_PLACE_NAMES),
        "road_labels": list(_ROAD_LABELS),
        "landmark_notes": list(_LANDMARK_NOTES),
        "sub_locations": list(_SUB_LOCATIONS),
        "vague_locations": list(_VAGUE_LOCATIONS),
        "stats": {
            "exact_address_count": len(exact_addresses),
            "street_count": len(street_rows),
            "ort_count": len(ort_names),
            "bez_count": len(bez_names),
            "postcode_count": len(postcodes),
        },
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8")
    return lexicon


def load_or_build_berlin_location_lexicon(
    source_path: Path | None = None,
    cache_path: Path = DEFAULT_BERLIN_LEXICON_CACHE,
    *,
    max_exact_addresses: int = 12000,
    seed: int = 23,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    source_path = Path(source_path) if source_path is not None else DEFAULT_BERLIN_GEOJSON

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") == BERLIN_LEXICON_SCHEMA_VERSION
            and cached.get("source_path") == str(source_path)
            and source_path.exists()
            and cached.get("source_mtime_ns") == source_path.stat().st_mtime_ns
        ):
            return cached
        if not source_path.exists():
            return cached

    if not source_path.exists():
        raise FileNotFoundError(f"Berlin address source not found: {source_path}")
    return build_berlin_location_lexicon(
        source_path,
        cache_path=cache_path,
        max_exact_addresses=max_exact_addresses,
        seed=seed,
    )


def sample_exact_address_seed(lexicon: dict[str, Any], rng: random.Random) -> dict[str, str]:
    row = rng.choice(lexicon["exact_addresses"])
    variant = rng.choice(
        (
            row["canonical"],
            row["canonical_with_ort"],
            row["canonical_with_bezirk"],
            f"{row['street']} {row['house_number']} in {row['ort_name']}, Berlin".strip(", "),
        )
    )
    return {
        "spoken_location": variant,
        "location_candidate": row["canonical"],
        "location_kind": "exact_address",
        "district_hint": row["bez_name"],
        "ort_hint": row["ort_name"],
    }


def sample_place_name_seed(lexicon: dict[str, Any], rng: random.Random) -> dict[str, str]:
    place_name = rng.choice(lexicon["place_names"])
    district = rng.choice(lexicon["ort_names"]) if lexicon["ort_names"] else "Berlin"
    variant = rng.choice(
        (
            place_name,
            f"{place_name} in {district}",
            f"inside {place_name}",
        )
    )
    return {
        "spoken_location": variant,
        "location_candidate": place_name,
        "location_kind": "place_name",
        "district_hint": district,
    }


def sample_road_or_junction_seed(lexicon: dict[str, Any], rng: random.Random) -> dict[str, str]:
    first = rng.choice(lexicon["streets"])
    second = rng.choice(lexicon["streets"])
    label = rng.choice(lexicon["road_labels"])
    if label in {"corner of", "junction of"}:
        spoken = f"{label} {first['street']} and {second['street']}"
        location_candidate = f"{first['street']} / {second['street']}, Berlin"
    elif label == "near":
        spoken = f"near {first['street']} in {first['ort_name'] or 'Berlin'}"
        location_candidate = f"{first['street']}, Berlin"
    else:
        spoken = f"{label} {first['street']}"
        location_candidate = f"{first['street']}, Berlin"
    return {
        "spoken_location": spoken,
        "location_candidate": location_candidate,
        "location_kind": "road_or_junction",
        "district_hint": first["ort_name"] or first["bez_name"] or "Berlin",
    }


def sample_landmark_note(lexicon: dict[str, Any], rng: random.Random) -> str:
    return rng.choice(lexicon["landmark_notes"])


def sample_sub_location(lexicon: dict[str, Any], rng: random.Random) -> str:
    return rng.choice(lexicon["sub_locations"])


def sample_vague_location(lexicon: dict[str, Any], rng: random.Random) -> str:
    return rng.choice(lexicon["vague_locations"])
