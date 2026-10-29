import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from emergency_dispatcher import session_store


def test_concurrent_session_patches_remain_valid_and_are_not_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    session_id = "concurrent-session-store-regression"
    recorders = [session_store.SessionRecorder(session_id) for _ in range(8)]
    barrier = Barrier(len(recorders))

    def update(index: int) -> None:
        barrier.wait()
        for revision in range(20):
            recorders[index].patch_session(
                {"concurrent_updates": {str(index): revision}}
            )

    with ThreadPoolExecutor(max_workers=len(recorders)) as pool:
        list(pool.map(update, range(len(recorders))))

    path = tmp_path / f"{session_id}.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["concurrent_updates"] == {str(index): 19 for index in range(8)}
    assert session_store.get_session(session_id)["concurrent_updates"] == persisted[
        "concurrent_updates"
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_circular_tool_result_does_not_break_live_session_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    recorder = session_store.SessionRecorder("circular-tool-result-regression")
    result = {"status": "ok"}
    result["nested"] = result

    recorder.append_tool_call(name="test_tool", args={}, result=result)

    saved = session_store.get_session("circular-tool-result-regression")
    assert saved["tool_calls"][0]["result"]["status"] == "ok"
    assert saved["tool_calls"][0]["result"]["nested"] == {
        "_serialization_note": "circular_reference_omitted"
    }


def test_session_index_skips_corrupt_record_without_logging_sensitive_content(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path)
    session_store.SessionRecorder("valid-session").patch_server({"status": "simulated"})
    private_marker = "PRIVATE_CALLER_TRANSCRIPT_MUST_NOT_APPEAR"
    (tmp_path / "damaged-session.json").write_text('{"text":"' + private_marker + '"}}', encoding="utf-8")

    with caplog.at_level("WARNING", logger=session_store.__name__):
        sessions = session_store.list_sessions()

    assert [item["session_id"] for item in sessions] == ["valid-session"]
    assert "session_index_record_skipped error_type=JSONDecodeError" in caplog.text
    assert private_marker not in caplog.text
    assert (tmp_path / "damaged-session.json").exists()
