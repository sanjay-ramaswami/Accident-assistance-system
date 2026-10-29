import json

from emergency_dispatcher.berlin_location_lexicon import build_berlin_location_lexicon


def test_build_berlin_location_lexicon_from_small_geojson(tmp_path):
    source_path = tmp_path / "berlin.geojson"
    cache_path = tmp_path / "berlin_cache.json"
    source_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "hnr": 20,
                            "hnr_zusatz": None,
                            "str_name": "Werderstraße",
                            "plz": "13587",
                            "bez_name": "Spandau",
                            "ort_name": "Hakenfelde",
                            "qualitaet": "Qualitaet A",
                            "typ": "Adresse",
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "hnr": 7,
                            "hnr_zusatz": "A",
                            "str_name": "Donaustraße",
                            "plz": "12043",
                            "bez_name": "Neukölln",
                            "ort_name": "Neukölln",
                            "qualitaet": "Qualitaet A",
                            "typ": "Adresse",
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lexicon = build_berlin_location_lexicon(source_path, cache_path=cache_path, max_exact_addresses=10, seed=5)

    assert lexicon["stats"]["exact_address_count"] == 2
    assert any(row["canonical"] == "Werderstraße 20, 13587 Berlin" for row in lexicon["exact_addresses"])
    assert any(row["canonical"] == "Donaustraße 7A, 12043 Berlin" for row in lexicon["exact_addresses"])
    assert cache_path.exists()
