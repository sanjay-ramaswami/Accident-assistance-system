import json

import httpx
import pytest

from emergency_dispatcher import dispatch_tools
from emergency_dispatcher.dispatch_tools import assign_dispatch_priority
from emergency_dispatcher.dispatch_tools import build_handoff_brief
from emergency_dispatcher.dispatch_tools import build_gradbot_tool_defs
from emergency_dispatcher.dispatch_tools import checklist_by_incident
from emergency_dispatcher.dispatch_tools import infer_issue_type
from emergency_dispatcher.dispatch_tools import lookup_address
from emergency_dispatcher.dispatch_tools import lookup_response_bases
from emergency_dispatcher.dispatch_tools import nearby_context
from emergency_dispatcher.dispatch_tools import plan_response_services
from emergency_dispatcher.dispatch_tools import resolve_location_note
from emergency_dispatcher.dispatch_tools import run_dispatch_workflow
from emergency_dispatcher.dispatch_tools import simulate_dispatch_services
from emergency_dispatcher.dispatch_tools import update_soft_ledger
from emergency_dispatcher.dispatch_tools import validate_address

_real_nominatim_candidates = dispatch_tools._nominatim_candidates


@pytest.fixture(autouse=True)
def stub_geocoders(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])

    def fake_candidates(query):
        if "Müllerstraße" in query:
            return [
                {
                    "display_name": "128 Müllerstraße, Wedding, Berlin, Germany",
                    "lat": "52.551",
                    "lon": "13.362",
                    "type": "road",
                    "class": "highway",
                    "importance": 0.72,
                    "address": {
                        "suburb": "Wedding",
                        "city": "Berlin",
                        "state": "Berlin",
                        "country": "Germany",
                    },
                }
            ]
        if "Main Street" in query:
            return [
                {
                    "display_name": "128 Main Street, Berlin, Germany",
                    "lat": "52.520",
                    "lon": "13.405",
                    "type": "road",
                    "class": "highway",
                    "importance": 0.62,
                    "address": {
                        "city": "Berlin",
                        "state": "Berlin",
                        "country": "Germany",
                    },
                }
            ]
        return []

    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_candidates)
    monkeypatch.setattr(
        dispatch_tools,
        "_overpass_context",
        lambda _lat, _lon: {
            "major_roads": [],
            "landmarks": [],
            "named_areas": [],
            "water_features": [],
        },
    )


def test_infer_issue_type_medical():
    assert infer_issue_type("The patient collapsed and is not breathing.") == "MEDICAL"


def test_dispatch_workflow_marks_fire_as_critical():
    result = run_dispatch_workflow(
        "Please hurry, I am at 128 Main Street, Berlin and there is smoke and an explosion risk."
    )
    assert result.issue_type == "FIRE"
    assert result.priority["priority"] in {"HIGH", "CRITICAL"}
    assert result.ticket["status"] == "ready_for_dispatch"


def test_dispatch_workflow_extracts_street_first_address():
    result = run_dispatch_workflow(
        "Please hurry, I am on Müllerstraße near number 128, Berlin and a car is on fire."
    )
    assert result.address_lookup["found"] is True
    assert result.ticket["address"] == "128 Müllerstraße, Wedding, Berlin, Germany"


def test_priority_tool_normalizes_free_form_labels():
    result = assign_dispatch_priority(
        issue_type="traffic accident",
        safety_risk="unknown, potentially high until confirmed",
        caller_state="caller reporting an accident, details unclear",
    )
    assert result["normalized_issue_type"] == "TRAFFIC"
    assert result["normalized_safety_risk"] == "ELEVATED"
    assert result["normalized_caller_state"] == "UNCERTAIN"
    assert result["priority"] in {"MEDIUM", "HIGH", "CRITICAL"}


def test_validate_address_uses_geocoding_when_available(monkeypatch):
    def fake_candidates(_query):
        return [
            {
                "display_name": "Müllerstraße 128, Wedding, Berlin, Germany",
                "lat": "52.551",
                "lon": "13.362",
                "type": "road",
                "class": "highway",
                "importance": 0.72,
                "address": {
                    "suburb": "Wedding",
                    "city": "Berlin",
                    "state": "Berlin",
                    "country": "Germany",
                },
            }
        ]

    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_candidates)

    result = validate_address("128 Müllerstraße, Berlin")

    assert result["valid"] is True
    assert result["geocoded"] is True
    assert result["source"] == "nominatim"
    assert result["lat"] == 52.551
    assert result["lon"] == 13.362
    assert result["normalized_address"] == "Müllerstraße 128, Wedding, Berlin, Germany"


def test_validate_address_prefers_google_when_available(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_google_geocode_candidates",
        lambda _query: [
            {
                "formatted_address": "Donaustraße 44, 12043 Berlin, Germany",
                "place_id": "test-place-id",
                "types": ["street_address"],
                "geometry": {
                    "location": {
                        "lat": 52.4799051,
                        "lng": 13.4394434,
                    },
                    "location_type": "ROOFTOP",
                },
                "address_components": [
                    {"long_name": "Donaustraße 44", "types": ["route"]},
                    {"long_name": "Berlin", "types": ["locality"]},
                    {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                    {"long_name": "Germany", "types": ["country"]},
                ],
            }
        ],
    )
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = validate_address("Delta Campus Berlin")

    assert result["valid"] is True
    assert result["geocoded"] is True
    assert result["source"] == "google_geocoding"
    assert result["confidence"] == "high"
    assert result["normalized_address"] == "Donaustraße 44, 12043 Berlin, Germany"
    assert result["lat"] == 52.4799051
    assert result["lon"] == 13.4394434


def test_lookup_address_google_result_is_json_serializable(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_google_geocode_candidates",
        lambda _query: [
            {
                "formatted_address": "Donaustraße 44, 12043 Berlin, Germany",
                "place_id": "test-place-id",
                "types": ["street_address"],
                "geometry": {
                    "location": {
                        "lat": 52.4799051,
                        "lng": 13.4394434,
                    },
                    "location_type": "ROOFTOP",
                },
                "address_components": [
                    {"long_name": "Donaustraße 44", "types": ["route"]},
                    {"long_name": "Berlin", "types": ["locality"]},
                    {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                    {"long_name": "Germany", "types": ["country"]},
                ],
            }
        ],
    )
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = lookup_address("Delta Campus Berlin")

    json.dumps(result)
    assert result["candidates"][0]["display_name"] == "Donaustraße 44, 12043 Berlin, Germany"
    assert result["candidates"][0] is not result


def test_lookup_address_allow_best_effort_can_resolve_place_like_candidate(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_load_berlin_location_semantic_index",
        lambda: {
            "entries": (
                {
                    "kind": "place",
                    "candidate": "Tempelhofer Feld",
                    "normalized": "tempelhofer feld",
                    "compact": "tempelhoferfeld",
                    "stem": "tempelhofer",
                    "suffix": "feld",
                },
            ),
            "by_suffix_prefix": {
                ("feld", "te"): [
                    {
                        "kind": "place",
                        "candidate": "Tempelhofer Feld",
                        "normalized": "tempelhofer feld",
                        "compact": "tempelhoferfeld",
                        "stem": "tempelhofer",
                        "suffix": "feld",
                    }
                ],
                ("feld", "t"): [
                    {
                        "kind": "place",
                        "candidate": "Tempelhofer Feld",
                        "normalized": "tempelhofer feld",
                        "compact": "tempelhoferfeld",
                        "stem": "tempelhofer",
                        "suffix": "feld",
                    }
                ],
            },
            "by_suffix": {
                "feld": [
                    {
                        "kind": "place",
                        "candidate": "Tempelhofer Feld",
                        "normalized": "tempelhofer feld",
                        "compact": "tempelhoferfeld",
                        "stem": "tempelhofer",
                        "suffix": "feld",
                    }
                ]
            },
            "by_prefix": {
                "te": [
                    {
                        "kind": "place",
                        "candidate": "Tempelhofer Feld",
                        "normalized": "tempelhofer feld",
                        "compact": "tempelhoferfeld",
                        "stem": "tempelhofer",
                        "suffix": "feld",
                    }
                ],
                "t": [
                    {
                        "kind": "place",
                        "candidate": "Tempelhofer Feld",
                        "normalized": "tempelhofer feld",
                        "compact": "tempelhoferfeld",
                        "stem": "tempelhofer",
                        "suffix": "feld",
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(
        dispatch_tools,
        "_google_geocode_candidates",
        lambda _query: [
            {
                "formatted_address": "Tempelhofer Feld, Tempelhofer Damm, 12101 Berlin, Germany",
                "place_id": "tempelhofer-feld",
                "types": ["park", "point_of_interest"],
                "geometry": {
                    "location": {
                        "lat": 52.4748569,
                        "lng": 13.4005926,
                    },
                    "location_type": "GEOMETRIC_CENTER",
                },
                "address_components": [
                    {"long_name": "Tempelhofer Feld", "types": ["establishment"]},
                    {"long_name": "Berlin", "types": ["locality"]},
                    {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                    {"long_name": "Germany", "types": ["country"]},
                ],
            }
        ],
    )
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = lookup_address("Templehofer Feld, Berlin", allow_best_effort=True)

    assert result["found"] is True
    assert result["source"] == "google_geocoding"
    assert result["normalized_address"] == "Tempelhofer Feld, Tempelhofer Damm, 12101 Berlin, Germany"
    assert result["semantic_match"]["candidate"] == "Tempelhofer Feld"
    assert result["semantic_match"]["apply"] is True


def test_semantic_match_berlin_location_rejects_ambiguous_guess(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_load_berlin_location_semantic_index",
        lambda: {
            "entries": (
                {
                    "kind": "street",
                    "candidate": "Merziger Straße",
                    "normalized": "merziger strasse",
                    "compact": "merzigerstrasse",
                    "stem": "merziger",
                    "suffix": "strasse",
                },
                {
                    "kind": "street",
                    "candidate": "Maarer Straße",
                    "normalized": "maarer strasse",
                    "compact": "maarerstrasse",
                    "stem": "maarer",
                    "suffix": "strasse",
                },
            ),
            "by_suffix_prefix": {
                ("strasse", "ma"): [
                    {
                        "kind": "street",
                        "candidate": "Merziger Straße",
                        "normalized": "merziger strasse",
                        "compact": "merzigerstrasse",
                        "stem": "merziger",
                        "suffix": "strasse",
                    },
                    {
                        "kind": "street",
                        "candidate": "Maarer Straße",
                        "normalized": "maarer strasse",
                        "compact": "maarerstrasse",
                        "stem": "maarer",
                        "suffix": "strasse",
                    },
                ],
                ("strasse", "m"): [
                    {
                        "kind": "street",
                        "candidate": "Merziger Straße",
                        "normalized": "merziger strasse",
                        "compact": "merzigerstrasse",
                        "stem": "merziger",
                        "suffix": "strasse",
                    },
                    {
                        "kind": "street",
                        "candidate": "Maarer Straße",
                        "normalized": "maarer strasse",
                        "compact": "maarerstrasse",
                        "stem": "maarer",
                        "suffix": "strasse",
                    },
                ],
            },
            "by_suffix": {
                "strasse": [
                    {
                        "kind": "street",
                        "candidate": "Merziger Straße",
                        "normalized": "merziger strasse",
                        "compact": "merzigerstrasse",
                        "stem": "merziger",
                        "suffix": "strasse",
                    },
                    {
                        "kind": "street",
                        "candidate": "Maarer Straße",
                        "normalized": "maarer strasse",
                        "compact": "maarerstrasse",
                        "stem": "maarer",
                        "suffix": "strasse",
                    },
                ]
            },
            "by_prefix": {
                "ma": [
                    {
                        "kind": "street",
                        "candidate": "Merziger Straße",
                        "normalized": "merziger strasse",
                        "compact": "merzigerstrasse",
                        "stem": "merziger",
                        "suffix": "strasse",
                    },
                    {
                        "kind": "street",
                        "candidate": "Maarer Straße",
                        "normalized": "maarer strasse",
                        "compact": "maarerstrasse",
                        "stem": "maarer",
                        "suffix": "strasse",
                    },
                ],
                "m": [
                    {
                        "kind": "street",
                        "candidate": "Merziger Straße",
                        "normalized": "merziger strasse",
                        "compact": "merzigerstrasse",
                        "stem": "merziger",
                        "suffix": "strasse",
                    },
                    {
                        "kind": "street",
                        "candidate": "Maarer Straße",
                        "normalized": "maarer strasse",
                        "compact": "maarerstrasse",
                        "stem": "maarer",
                        "suffix": "strasse",
                    },
                ],
            },
        },
    )

    result = dispatch_tools._semantic_match_berlin_location("Madriger Strasse")

    assert result is not None
    assert result["apply"] is False


def test_lookup_address_allow_best_effort_uses_semantic_candidate_before_geocoding(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_load_berlin_location_semantic_index",
        lambda: {
            "entries": (
                {
                    "kind": "street",
                    "candidate": "Oxforder Straße",
                    "normalized": "oxforder strasse",
                    "compact": "oxforderstrasse",
                    "stem": "oxforder",
                    "suffix": "strasse",
                },
            ),
            "by_suffix_prefix": {
                ("strasse", "ox"): [
                    {
                        "kind": "street",
                        "candidate": "Oxforder Straße",
                        "normalized": "oxforder strasse",
                        "compact": "oxforderstrasse",
                        "stem": "oxforder",
                        "suffix": "strasse",
                    }
                ],
                ("strasse", "o"): [
                    {
                        "kind": "street",
                        "candidate": "Oxforder Straße",
                        "normalized": "oxforder strasse",
                        "compact": "oxforderstrasse",
                        "stem": "oxforder",
                        "suffix": "strasse",
                    }
                ],
            },
            "by_suffix": {
                "strasse": [
                    {
                        "kind": "street",
                        "candidate": "Oxforder Straße",
                        "normalized": "oxforder strasse",
                        "compact": "oxforderstrasse",
                        "stem": "oxforder",
                        "suffix": "strasse",
                    }
                ]
            },
            "by_prefix": {
                "ox": [
                    {
                        "kind": "street",
                        "candidate": "Oxforder Straße",
                        "normalized": "oxforder strasse",
                        "compact": "oxforderstrasse",
                        "stem": "oxforder",
                        "suffix": "strasse",
                    }
                ],
                "o": [
                    {
                        "kind": "street",
                        "candidate": "Oxforder Straße",
                        "normalized": "oxforder strasse",
                        "compact": "oxforderstrasse",
                        "stem": "oxforder",
                        "suffix": "strasse",
                    }
                ],
            },
        },
    )

    def fake_google(query):
        if query == "Oxforder Straße":
            return [
                {
                    "formatted_address": "Oxforder Str., 13349 Berlin, Germany",
                    "place_id": "oxforder",
                    "types": ["route"],
                    "geometry": {
                        "location": {"lat": 52.5594, "lng": 13.3491},
                        "location_type": "GEOMETRIC_CENTER",
                    },
                    "address_components": [
                        {"long_name": "Oxforder Straße", "types": ["route"]},
                        {"long_name": "Berlin", "types": ["locality"]},
                        {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                        {"long_name": "Germany", "types": ["country"]},
                    ],
                }
            ]
        if query == "Oxford Strasse":
            return [
                {
                    "formatted_address": "Fasanenstraße 6, 10623 Berlin, Germany",
                    "place_id": "bad-jump",
                    "types": ["street_address"],
                    "geometry": {
                        "location": {"lat": 52.5069424, "lng": 13.3277458},
                        "location_type": "ROOFTOP",
                    },
                    "address_components": [
                        {"long_name": "Fasanenstraße 6", "types": ["route"]},
                        {"long_name": "Berlin", "types": ["locality"]},
                        {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                        {"long_name": "Germany", "types": ["country"]},
                    ],
                }
            ]
        return []

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", fake_google)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = lookup_address("Oxford Strasse, Berlin", allow_best_effort=True)

    assert result["normalized_address"] == "Oxforder Str., 13349 Berlin, Germany"
    assert result["semantic_match"]["candidate"] == "Oxforder Straße"
    assert result["semantic_match"]["apply"] is True


def test_semantic_match_ignores_trailing_berlin_and_recovers_street(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "load_or_build_berlin_location_lexicon",
        lambda: {"streets": [{"street": "Oxforder Straße"}], "place_names": []},
    )
    dispatch_tools._load_berlin_location_semantic_entries.cache_clear()
    dispatch_tools._load_berlin_location_semantic_index.cache_clear()
    try:
        result = dispatch_tools._semantic_match_berlin_location("Oxford Strassen, Berlin")
    finally:
        dispatch_tools._load_berlin_location_semantic_entries.cache_clear()
        dispatch_tools._load_berlin_location_semantic_index.cache_clear()

    assert result is not None
    assert result["candidate"] == "Oxforder Straße"
    assert result["apply"] is True


def test_resolve_location_note_strips_now_at_prefix():
    result = resolve_location_note(
        "I am right now at Dublin Ostrasse.",
        location_candidate="Dublin Ostrasse",
    )

    assert result["anchor_location"] == "Dublin Ostrasse"


def test_validate_address_rejects_generic_google_city_match(monkeypatch):
    monkeypatch.setattr(
        dispatch_tools,
        "_google_geocode_candidates",
        lambda _query: [
            {
                "formatted_address": "Berlin, Germany",
                "place_id": "generic-berlin",
                "types": ["locality", "political"],
                "geometry": {
                    "location": {
                        "lat": 52.5200066,
                        "lng": 13.404954,
                    },
                    "location_type": "APPROXIMATE",
                },
                "address_components": [
                    {"long_name": "Berlin", "types": ["locality"]},
                    {"long_name": "Berlin", "types": ["administrative_area_level_1"]},
                    {"long_name": "Germany", "types": ["country"]},
                ],
            }
        ],
    )
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = validate_address("Reeperbahn 1")

    assert result["geocoded"] is False
    assert result["source"] == "text_fallback"


def test_lookup_address_falls_back_when_geocoder_returns_no_match(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    result = lookup_address("12 Quartz Road, Berlin")

    assert result["found"] is False
    assert result["geocoded"] is False
    assert result["source"] == "text_fallback"
    assert result["normalized_address"] == "12 Quartz Road, Berlin"
    assert result["reason"] in {"nominatim_no_match", "google_no_match_nominatim_no_match"}


def test_location_provider_errors_return_unresolved_result_without_crashing(monkeypatch):
    def provider_error(_query):
        raise httpx.ConnectError("private provider detail")

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", provider_error)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", provider_error)
    monkeypatch.setattr(dispatch_tools, "_google_maps_api_key", lambda: None)

    result = lookup_address("12 Quartz Road, Berlin")

    assert result["found"] is False
    assert result["geocoded"] is False
    assert result["lat"] is None and result["lon"] is None
    assert result["reason"] == "nominatim_error"


def test_city_country_candidate_is_geocoded_but_remains_approximate(monkeypatch):
    queries = []

    def fake_google(query):
        queries.append(query)
        return [{
            "formatted_address": "Dubai, United Arab Emirates",
            "geometry": {"location": {"lat": 25.2048, "lng": 55.2708}, "location_type": "APPROXIMATE"},
                "address_components": [
                    {"long_name": "Dubai", "types": ["locality"]},
                    {"long_name": "United Arab Emirates", "types": ["country"]},
                ],
        }]

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", fake_google)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: [])

    allowed, reason = dispatch_tools.assess_searchable_location_query("Dubai, United Arab Emirates")
    result = lookup_address("Dubai, United Arab Emirates", allow_best_effort=True)

    assert (allowed, reason) == (True, "city_country_candidate")
    assert queries == ["Dubai, United Arab Emirates"]
    assert result["geocoded"] is True
    assert result["lat"] == 25.2048
    assert result["lon"] == 55.2708
    assert result["confidence"] in {"low", "medium"}


def test_city_country_with_multiple_geocoder_matches_stays_unresolved(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(
        dispatch_tools,
        "_nominatim_candidates",
        lambda _query: [
            {"display_name": "Springfield, Illinois, United States", "lat": "39.78", "lon": "-89.64", "type": "city", "address": {"city": "Springfield", "country": "United States"}},
            {"display_name": "Springfield, Massachusetts, United States", "lat": "42.10", "lon": "-72.59", "type": "city", "address": {"city": "Springfield", "country": "United States"}},
        ],
    )

    result = lookup_address("Springfield, United States", allow_best_effort=True)

    assert result["found"] is False
    assert result["geocoded"] is False
    assert result["reason"] == "ambiguous_location"
    assert result["lat"] is None
    assert len(result["candidates"]) == 2


def test_city_state_uses_one_exact_named_city_candidate_as_approximate(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(
        dispatch_tools,
        "_nominatim_candidates",
        lambda _query: [
            {
                "display_name": "Kochi, Ernakulam, Kerala, 682001, India",
                "lat": "9.9679032",
                "lon": "76.2444378",
                "type": "city",
                "address": {"city": "Kochi", "state": "Kerala", "country": "India"},
            },
            {
                "display_name": "Kochi, Ernakulam, Kanayannur, Kerala, India",
                "lat": "9.9711453",
                "lon": "76.284638",
                "type": "local_authority",
                "address": {"city": "Ernakulam", "state": "Kerala", "country": "India"},
            },
            {
                "display_name": "Kochi, Ernakulam, Kerala, India",
                "lat": "9.9800645",
                "lon": "76.2338365",
                "type": "administrative",
                "address": {"state": "Kerala", "country": "India"},
            },
        ],
    )

    result = lookup_address("Kochi, Kerala", allow_best_effort=True)

    assert result["geocoded"] is True
    assert result["lat"] == pytest.approx(9.9679032)
    assert result["lon"] == pytest.approx(76.2444378)
    assert result["confidence"] == "low"
    assert result["reason"] == "approximate_city_match"
    assert len(result["candidates"]) == 3


def test_marine_drive_without_city_requires_clarification():
    result = resolve_location_note(None, location_candidate="Marine Drive")

    assert result["requires_city_clarification"] is True
    assert result["reason"] == "ambiguous_location_without_city"
    assert "Which city" in result["clarification_question"]


def test_city_qualified_marine_drive_uses_nominatim_fallback_and_is_approximate(monkeypatch):
    nominatim_queries = []

    def fake_nominatim(query):
        nominatim_queries.append(query)
        if query == "Marine Drive Kochi":
            return [
                {
                    "display_name": "Marine Drive, New Delhi, India",
                    "lat": "28.6139",
                    "lon": "77.2090",
                    "type": "road",
                    "address": {"city": "New Delhi", "state": "Delhi", "country": "India"},
                },
                {
                    "display_name": "Marine Drive, Kochi, Marine Drive Walkway, Shenoys, Ernakulam, Kerala, India",
                    "lat": "9.9784782",
                    "lon": "76.2757727",
                    "type": "attraction",
                    "address": {"city": "Kochi", "state": "Kerala", "country": "India"},
                },
                {
                    "display_name": "Kochi, Ernakulam, Kerala, India",
                    "lat": "9.9679032",
                    "lon": "76.2444378",
                    "type": "city",
                    "address": {"city": "Kochi", "state": "Kerala", "country": "India"},
                },
            ]
        return []

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(dispatch_tools, "_google_maps_api_key", lambda: None)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_nominatim)

    result = lookup_address("Marine Drive, Kochi, Kerala, India", allow_best_effort=True)

    assert nominatim_queries == ["Marine Drive, Kochi, Kerala, India", "Marine Drive Kochi"]
    assert result["found"] is True
    assert result["geocoded"] is True
    assert result["lat"] == pytest.approx(9.9784782)
    assert result["lon"] == pytest.approx(76.2757727)
    assert result["confidence"] == "low"
    assert result["reason"] == "approximate_place_match"
    assert len(result["candidates"]) == 1


def test_nominatim_place_city_query_filters_wrong_city_and_collapses_same_venue(monkeypatch):
    candidates = [
        {"display_name": "Lulu mall, Kochuveli, Thiruvananthapuram, Kerala, India", "lat": "8.5138", "lon": "76.9002", "importance": 0.0001, "type": "bus_stop", "address": {"city": "Thiruvananthapuram", "suburb": "Kochuveli", "state": "Kerala", "country": "India"}},
        {"display_name": "LuLu International Shopping Mall, Kochi, Edapally, Kochi, Kerala, India", "lat": "10.0272185", "lon": "76.3078873", "importance": 0.0907, "type": "mall", "address": {"city": "Kochi", "suburb": "Edapally", "state": "Kerala", "country": "India"}},
        {"display_name": "LuLu Mall, Kochi, Edappally, Kochi, Kerala, India", "lat": "10.0274822", "lon": "76.3079131", "importance": 0.0907, "type": "retail", "address": {"city": "Kochi", "suburb": "Edappally", "state": "Kerala", "country": "India"}},
    ]
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(dispatch_tools, "_google_maps_api_key", lambda: None)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", lambda _query: candidates)

    result = lookup_address("Lulu Mall, Kochi", allow_best_effort=True)

    assert result["geocoded"] is True
    assert result["confidence"] == "low"
    assert result["reason"] == "approximate_place_match"
    assert result["lat"] == pytest.approx(10.0272185)
    assert result["lon"] == pytest.approx(76.3078873)
    assert all("Thiruvananthapuram" not in item["display_name"] for item in result["candidates"])


def test_city_qualified_marine_drive_rejects_out_of_range_provider_coordinates(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(dispatch_tools, "_google_maps_api_key", lambda: None)
    monkeypatch.setattr(
        dispatch_tools,
        "_nominatim_candidates",
        lambda _query: [{
            "display_name": "Marine Drive, Kochi, Kerala, India",
            "lat": "95",
            "lon": "76",
            "type": "attraction",
            "address": {"city": "Kochi", "state": "Kerala", "country": "India"},
        }],
    )

    result = lookup_address("Marine Drive, Kochi, Kerala, India", allow_best_effort=True)

    assert result["found"] is False
    assert result["lat"] is None and result["lon"] is None
    assert result["reason"] == "nominatim_no_match"


def test_nominatim_http_failure_logs_status_and_redacts_location_query(monkeypatch, caplog):
    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            request = httpx.Request("GET", "https://nominatim.openstreetmap.org/search")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("PRIVATE_QUERY_AND_TOKEN", request=request, response=response)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            return FakeResponse()

    caplog.set_level("INFO", logger=dispatch_tools.__name__)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", _real_nominatim_candidates)
    monkeypatch.setattr(dispatch_tools.httpx, "Client", FakeClient)
    with pytest.raises(httpx.HTTPStatusError):
        dispatch_tools._nominatim_candidates("Private Caller Address")

    assert "http_status=429" in caplog.text
    assert "Private Caller Address" not in caplog.text
    assert "PRIVATE_QUERY_AND_TOKEN" not in caplog.text


def test_nominatim_invalid_json_is_a_safe_geocoder_failure(monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("PRIVATE_RESPONSE_BODY")

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", _real_nominatim_candidates)
    monkeypatch.setattr(dispatch_tools.httpx, "Client", FakeClient)
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(dispatch_tools, "_google_maps_api_key", lambda: None)

    result = lookup_address("Marine Drive, Kochi, Kerala, India", allow_best_effort=True)

    assert result["reason"] == "nominatim_error"
    assert result["lat"] is None and result["lon"] is None
    assert result["source"] == "text_fallback"


def test_validate_address_rejects_vague_location_phrase_without_geocoding(monkeypatch):
    google_called = False
    nominatim_called = False

    def fake_google(_query):
        nonlocal google_called
        google_called = True
        return []

    def fake_nominatim(_query):
        nonlocal nominatim_called
        nominatim_called = True
        return []

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", fake_google)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_nominatim)

    result = validate_address("middle of the street")

    assert result["valid"] is False
    assert result["geocoded"] is False
    assert result["source"] == "guardrail"
    assert result["reason"] == "location_not_specific_enough"
    assert google_called is False
    assert nominatim_called is False


def test_lookup_address_rejects_vague_location_phrase_without_geocoding(monkeypatch):
    google_called = False
    nominatim_called = False

    def fake_google(_query):
        nonlocal google_called
        google_called = True
        return []

    def fake_nominatim(_query):
        nonlocal nominatim_called
        nominatim_called = True
        return []

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", fake_google)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_nominatim)

    result = lookup_address("inside the building")

    assert result["found"] is False
    assert result["geocoded"] is False
    assert result["source"] == "guardrail"
    assert result["reason"] == "location_not_specific_enough"
    assert result["location_context"] is None
    assert google_called is False
    assert nominatim_called is False


def test_validate_address_allows_searchable_street_suffix_phrase(monkeypatch):
    google_called = False
    nominatim_called = False

    def fake_google(_query):
        nonlocal google_called
        google_called = True
        return []

    def fake_nominatim(_query):
        nonlocal nominatim_called
        nominatim_called = True
        return []

    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", fake_google)
    monkeypatch.setattr(dispatch_tools, "_nominatim_candidates", fake_nominatim)

    result = validate_address("Mullerstrasse near number 128 in Berlin")

    assert result["source"] == "text_fallback"
    assert result["reason"] in {"nominatim_no_match", "google_no_match_nominatim_no_match"}
    assert google_called is True or nominatim_called is True


def test_nominatim_queries_remain_global_without_explicit_jurisdiction():
    params = dispatch_tools._nominatim_query_params("Janpath Road, New Delhi, India")

    assert params["q"] == "Janpath Road, New Delhi, India"
    assert "countrycodes" not in params
    assert "viewbox" not in params
    assert "bounded" not in params


def test_nominatim_keeps_berlin_bias_only_for_explicit_berlin_query():
    params = dispatch_tools._nominatim_query_params("Müllerstraße 128, Berlin")

    assert params["q"] == "Müllerstraße 128, Berlin"
    assert params["countrycodes"] == "de"
    assert params["bounded"] == 1
    assert params["viewbox"] == dispatch_tools.BERLIN_VIEWBOX


def test_google_query_does_not_force_berlin_or_germany(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    params = dispatch_tools._google_query_params("Janpath Road, New Delhi, India")

    assert params["address"] == "Janpath Road, New Delhi, India"
    assert "region" not in params
    assert "components" not in params
    assert "bounds" not in params


def test_geocoder_rejects_invalid_coordinates(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_geocode_candidates", lambda _query: [])
    monkeypatch.setattr(
        dispatch_tools,
        "_nominatim_candidates",
        lambda _query: [{"display_name": "Invalid", "lat": "91", "lon": "13", "address": {}}],
    )

    result = validate_address("Janpath Road, New Delhi, India")

    assert result["geocoded"] is False
    assert result["lat"] is None
    assert result["lon"] is None
    assert result["source"] == "text_fallback"


def test_resolve_location_note_preserves_landmark_without_fake_pin():
    result = resolve_location_note("behind the red brick church", sub_location="west entrance")

    assert result["note_type"] == "landmark_relative"
    assert result["location_status"] == "best_effort_note"
    assert result["usable_for_dispatch"] is True
    assert result["search_allowed"] is False
    assert result["sub_location"] == "west entrance"


def test_resolve_location_note_extracts_named_place_with_sub_location():
    result = resolve_location_note("somewhere close to Delta Campus, fourth floor", sub_location="fourth floor")

    assert result["anchor_location"] == "Delta Campus"
    assert result["location_status"] == "usable_place"
    assert result["sub_location"] == "fourth floor"


def test_resolve_location_note_strips_command_prefix_from_named_place():
    result = resolve_location_note("find delta campus.")

    assert result["anchor_location"] == "delta campus"
    assert result["search_allowed"] is True
    assert result["location_status"] == "usable_place"


def test_resolve_location_note_does_not_promote_incident_words_as_place_name():
    result = resolve_location_note(
        "There was an accident at Delta Campus.",
        location_candidate="Delta Campus",
    )

    assert result["anchor_location"] == "Delta Campus"


def test_resolve_location_note_promotes_new_street_anchor_from_note():
    result = resolve_location_note(
        "Near Müllerstraße, outside the subway station, toward the sea exit.",
        location_candidate="Alexander Platz",
        sub_location="outside the subway station, toward the sea exit",
    )

    assert result["anchor_location"] == "Müllerstraße"
    assert result["search_allowed"] is True
    assert result["location_status"] == "usable_place"


def test_resolve_location_note_does_not_promote_bare_street_suffix():
    result = resolve_location_note("strasse")

    assert result["anchor_location"] is None
    assert result["search_allowed"] is False
    assert result["location_status"] == "best_effort_note"


def test_nearby_context_returns_summary_for_geocoded_address():
    result = nearby_context(address_text="128 Müllerstraße, Berlin")

    assert result["resolved"] is True
    assert result["normalized_address"] == "128 Müllerstraße, Wedding, Berlin, Germany"
    assert result["summary"]
    assert result["reason"] in {"nominatim+overpass", "nominatim_only"}


def test_checklist_by_incident_prioritizes_location_anchor_after_issue():
    result = checklist_by_incident(
        {
            "issue_cues": ["fire"],
            "location_note": "behind the supermarket",
            "sub_location": "rear stairwell",
            "victim_count": "unknown",
        }
    )

    assert result["location_status"] == "best_effort_note"
    assert result["missing_fields"][0] == "location_anchor"
    assert result["next_question_field"] == "location_candidate"
    assert result["location_search_allowed"] is False


def test_build_handoff_brief_summarizes_best_effort_dispatch_state():
    result = build_handoff_brief(
        caller_summary="There is smoke behind the supermarket by the rear stairwell.",
        args={
            "issue_cues": ["smoke", "fire"],
            "location_note": "behind the supermarket",
            "sub_location": "rear stairwell",
            "victim_count": 2,
        },
    )

    assert result["status"] == "dispatchable_with_note"
    assert "smoke" in result["one_line"]
    assert "Best effort location note" in result["dispatcher_note"]
    assert "location anchor" in " ".join(result["unknowns"])


def test_live_gradbot_tool_set_keeps_fact_tools_and_retires_weak_ones():
    tool_names = [name for name, _description, _schema in build_gradbot_tool_defs()]

    assert tool_names == [
        "resolve_location_note",
        "checklist_by_incident",
        "validate_address",
        "lookup_address",
        "nearby_context",
        "build_handoff_brief",
        "create_incident_ticket",
    ]


def test_live_gradbot_tool_set_can_opt_into_soft_ledger_updates():
    tool_names = [name for name, _description, _schema in build_gradbot_tool_defs(include_soft_ledger=True)]

    assert tool_names[-1] == "update_soft_ledger"


def test_update_soft_ledger_only_returns_soft_fields():
    result = update_soft_ledger(
        {
            "people_count_best_guess": 2,
            "caller_role": "witness",
            "notes": ["caller sounds unsure", "child may be nearby"],
        }
    )

    assert result["soft_state"]["people_count_best_guess"] == 2
    assert result["soft_state"]["caller_role"] == "witness"
    assert result["soft_state"]["notes"] == ["caller sounds unsure", "child may be nearby"]
    assert "people_count_best_guess" in result["updated_fields"]


def test_plan_response_services_flags_hostage_call_for_police_and_ambulance():
    result = plan_response_services(
        transcript="I am being held hostage, they have a gun, and my daughter is bleeding.",
        issue_type="POLICE",
    )

    assert result["human_monitoring"] is False
    assert result["human_review_recommended"] is True
    assert result["monitor_name"] is None
    assert result["priority"] == "CRITICAL"
    assert result["services"]["police"]["needed"] is True
    assert result["services"]["ambulance"]["needed"] is True


def test_simulate_dispatch_services_transitions_to_dispatched():
    plan = plan_response_services(
        transcript="There is smoke coming from the apartment and someone is trapped inside.",
        issue_type="FIRE",
    )
    first = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text="Müllerstraße 128, Berlin",
        location_confirmed=True,
        now_iso="2026-04-26T01:00:00+00:00",
    )
    later = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text="Müllerstraße 128, Berlin",
        location_confirmed=True,
        prior_state=first,
        now_iso="2026-04-26T01:00:05+00:00",
    )

    assert later["dispatchable"] is True
    assert later["services"]["fire"]["status"] == "dispatched"
    assert later["services"]["ambulance"]["status"] == "dispatched"
    assert later["services"]["fire"]["eta_min"] is None
    assert later["services"]["fire"]["eta_text"] is None
    assert "SIMULATION ONLY" in later["summary"]


def test_simulated_services_do_not_advance_for_unconfirmed_location_or_invent_eta():
    plan = plan_response_services(
        transcript="There is smoke coming from a vehicle.",
        issue_type="FIRE",
    )
    result = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text="Müllerstraße 128, Berlin",
        location_confirmed=False,
        response_bases={"fire": {"eta_min": 1, "eta_text": "1 min"}},
        now_iso="2026-04-26T01:00:05+00:00",
    )

    assert result["dispatchable"] is False
    assert result["services"]["fire"]["status"] == "locating"
    assert result["services"]["fire"]["eta_min"] is None
    assert result["services"]["fire"]["eta_text"] is None
    assert "checking the nearest available unit" in result["announcement_line"]
    assert "Alex" not in result["announcement_line"]


def test_lookup_response_bases_returns_grounded_service_locations(monkeypatch):
    monkeypatch.setattr(dispatch_tools, "_google_response_bases", lambda _lat, _lon: {})
    monkeypatch.setattr(
        dispatch_tools,
        "_overpass_response_bases",
        lambda _lat, _lon: {
            "police": {
                "display_name": "Abschnitt 52",
                "address": "Musterstraße 10, 10115 Berlin",
                "lat": 52.53,
                "lon": 13.40,
                "distance_km": 0.9,
                "source_type": "police",
            }
        },
    )

    result = lookup_response_bases(lat=52.52, lon=13.405, location_text="Berlin Mitte")

    assert result["resolved"] is True
    assert result["services"]["police"]["display_name"] == "Abschnitt 52"
    assert result["services"]["police"].get("eta_text") is None


def test_simulate_dispatch_services_uses_response_base_details():
    plan = plan_response_services(
        transcript="There is smoke coming from the apartment and someone is trapped inside.",
        issue_type="FIRE",
    )
    result = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text="Müllerstraße 128, Berlin",
        location_confirmed=True,
        response_bases={
            "fire": {
                "display_name": "Feuerwache Wedding",
                "address": "Müllerstraße 12, Berlin",
                "distance_km": 1.4,
                "source_type": "fire_station",
            }
        },
        now_iso="2026-04-26T01:00:05+00:00",
    )

    assert result["services"]["fire"]["base_name"] == "Feuerwache Wedding"
    assert result["services"]["fire"]["eta_text"] is None


def test_simulate_dispatch_services_carries_announcement_briefly_until_next_prompt():
    plan = plan_response_services(
        transcript="I am trapped in a crash and there is smoke.",
        issue_type="FIRE",
    )
    first = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text=None,
        location_confirmed=False,
        now_iso="2026-04-26T01:00:00+00:00",
    )
    assert first["announcement_line"]

    carried = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text=None,
        location_confirmed=False,
        prior_state=first,
        now_iso="2026-04-26T01:00:02+00:00",
    )
    assert carried["announcement_line"] == first["announcement_line"]

    expired = simulate_dispatch_services(
        session_id="demo-session",
        plan=plan,
        location_text=None,
        location_confirmed=False,
        prior_state=first,
        now_iso="2026-04-26T01:00:12+00:00",
    )
    assert expired["announcement_line"] is None


def test_dispatch_announcement_claims_are_bound_to_persisted_simulation_status():
    plan = plan_response_services(transcript="A person is bleeding after a crash.", issue_type="MEDICAL")
    locating = simulate_dispatch_services(
        session_id="demo-session", plan=plan, location_text="Kochi", location_confirmed=False,
        now_iso="2026-04-26T01:00:00+00:00",
    )
    assert "checking" in locating["announcement_line"]
    assert "assigned" not in locating["announcement_line"].lower()

    assigned = simulate_dispatch_services(
        session_id="demo-session", plan=plan, location_text="Kochi", location_confirmed=True,
        prior_state=locating, now_iso="2026-04-26T01:00:02+00:00",
    )
    assert assigned["services"]["ambulance"]["status"] == "assigned"
    assert "simulated ambulance has been assigned" in assigned["announcement_line"]
    assert "en route" not in assigned["announcement_line"].lower()
    assert "arrived" not in assigned["announcement_line"].lower()


def test_failed_or_unavailable_assignment_never_generates_success_claim():
    line = dispatch_tools._service_announcement_line("ambulance", "no_unit_available", None)
    assert "No simulated ambulance is available" in line
    assert "assigned" not in line.lower()
    assert "arrived" not in line.lower()

    arrived = dispatch_tools._service_announcement_line("ambulance", "arrived", None)
    assert "simulated ambulance has arrived" in arrived
    en_route = dispatch_tools._service_announcement_line("ambulance", "en_route", None)
    assert "simulated ambulance is en route" in en_route
