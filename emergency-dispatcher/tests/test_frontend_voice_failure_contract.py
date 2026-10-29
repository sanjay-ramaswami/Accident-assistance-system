"""Frontend contract: live voice failures must be explicit and honest.

Pins that server-sent provider errors (quota exhaustion, LLM watchdog, provider
initialization failures) are surfaced verbatim in the UI instead of being
dropped by the audio player, and that the dashboard never claims a live
connection or assistant audio that the backend did not confirm.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from emergency_dispatcher import server


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
AUDIO_UTILS_JS = (FRONTEND / "audio-utils.js").read_text(encoding="utf-8")
INDEX_HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")


def test_server_sent_ws_errors_are_surfaced_with_code_and_message():
    assert 'if (message.type === "error") {' in APP_JS
    assert "live provider error (${errorCode}): ${errorText}" in APP_JS
    assert 'setStatus("Error", "error");' in APP_JS
    # The explicit error must be consumed here and not fall through to the
    # generic audio player handler that discards the server message text.
    assert "return;" in APP_JS.split('if (message.type === "error") {', 1)[1].split("} catch (_)", 1)[0]


def test_connection_status_only_reports_connected_from_backend_confirmation():
    # "Connected" is only set for an explicit backend connection_status frame.
    assert 'if (message.status === "connected")' in APP_JS
    assert 'setStatus("Connected", "live");' in APP_JS
    # Opening the socket alone never claims a live provider session.
    open_block = APP_JS.split("callSocket.onopen = () => {", 1)[1].split("};", 1)[0]
    assert 'setStatus("Connected"' not in open_block


def test_abnormal_close_codes_map_to_unavailable_not_connected():
    assert "code === 1011" in AUDIO_UTILS_JS
    assert 'status: "Unavailable"' in AUDIO_UTILS_JS
    assert "No dispatcher audio was confirmed." in AUDIO_UTILS_JS


def test_text_only_responses_are_reported_as_not_played():
    assert '"text_without_audio"' in AUDIO_UTILS_JS
    assert '"not_played"' in AUDIO_UTILS_JS
    assert '"played"' in AUDIO_UTILS_JS


def test_voice_failure_keeps_local_first_aid_and_text_triage_wired_independently():
    # Local panels must keep their own fetch paths so they work during a
    # provider outage, without any websocket dependency.
    assert 'fetch("/api/first-aid"' in APP_JS
    assert 'fetch("/api/triage/complaint-code"' in APP_JS
    assert 'fetch("/api/triage"' in APP_JS


def test_demo_refreshes_versioned_app_assets_and_helpers():
    client = TestClient(server.app)
    page = client.get("/demo")

    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert 'styles.css?v=20261030-location-dispatch-1' in page.text
    assert 'audio-utils.js?v=20261029-complaint-ml-1' in page.text
    assert 'complaint-code-utils.js?v=20261029-complaint-ml-1' in page.text
    assert 'app.js?v=20261030-location-dispatch-4' in page.text
    assert 'location-utils.js?v=20261029-map-state-2' in page.text
    assert 'session-report-utils.js?v=20261029-session-report-3' in page.text
    assert 'live-ml-utils.js?v=20261030-explicit-code-2' in page.text
    assert 'window.DispatcherAudioUtils.relayAudioMetrics' in APP_JS

    audio_utils = client.get("/demo-static/audio-utils.js?v=20261029-complaint-ml-1")
    assert audio_utils.status_code == 200
    assert "function relayAudioMetrics(audioProcessor, onMetrics)" in audio_utils.text

    classifier_utils = client.get("/demo-static/complaint-code-utils.js?v=20261029-complaint-ml-1")
    assert classifier_utils.status_code == 200
    assert "presentComplaintCodeResult" in classifier_utils.text
    live_classifier_utils = client.get("/demo-static/live-ml-utils.js?v=20261030-explicit-code-2")
    assert live_classifier_utils.status_code == 200
    assert "createLiveComplaintClassifier" in live_classifier_utils.text
    session_report_utils = client.get("/demo-static/session-report-utils.js?v=20261029-session-report-3")
    assert session_report_utils.status_code == 200
    assert "postSessionReport" in session_report_utils.text


def test_live_session_report_avoids_keepalive_limit_and_surfaces_bounded_failure_states():
    assert "DispatcherSessionReportUtils.postSessionReport" in APP_JS
    assert "keepalive: true" not in APP_JS
    assert "automatic sync paused after 3 failed requests" in APP_JS
    assert "Retry session sync" in INDEX_HTML
    assert "malformed_response" in (FRONTEND / "session-report-utils.js").read_text(encoding="utf-8")
    assert "invalid_schema" in (FRONTEND / "session-report-utils.js").read_text(encoding="utf-8")


def test_provider_timeout_never_inserts_a_simulated_voice_reply():
    assert "no client-generated reply" in APP_JS
    assert "Simulated demo response · not live AI" not in INDEX_HTML
    assert 'id="provider-timeout-status"' in INDEX_HTML
