from fastapi.testclient import TestClient

from emergency_dispatcher.server import app


client = TestClient(app)


def test_demo_route_serves_console():
    response = client.get("/demo")

    assert response.status_code == 200
    assert "Live Dispatch Console" in response.text


def test_old_enhanced_demo_route_is_removed():
    response = client.get("/demo-enhanced")

    assert response.status_code == 404


def test_review_route_still_exists():
    response = client.get("/review")

    assert response.status_code == 200


def test_audio_config_uses_pcm_for_live_ai_coustics():
    response = client.get("/api/audio-config")

    assert response.status_code == 200
    assert response.json()["pcm"] is True
    assert response.json()["pcm_input"] is True
    assert response.json()["sample_rate"] == 24000
    assert response.json()["chunk_samples"] == 1920
