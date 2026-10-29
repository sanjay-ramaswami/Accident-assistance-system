#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_slm_fact_ledger_training_data import DEFAULT_BASE_URL  # noqa: E402


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _status_snapshot(batch: object) -> dict[str, object]:
    counts = getattr(batch, "request_counts", None)
    completed = int(getattr(counts, "completed", 0) or 0)
    failed = int(getattr(counts, "failed", 0) or 0)
    total = int(getattr(counts, "total", 0) or 0)
    pending = max(total - completed - failed, 0)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "request_counts": {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total": total,
        },
        "percent_complete": round((completed / total) * 100, 2) if total else 0.0,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "errors": _jsonable(batch.errors),
    }


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(description="Check progress for an SLM fact-ledger OpenAI Batch job.")
    parser.add_argument("--batch-id", help="OpenAI batch id.")
    parser.add_argument("--launch", type=Path, help="Launch JSON created by prepare_slm_fact_ledger_batch.py")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--watch-seconds", type=int, default=0, help="Poll repeatedly every N seconds until completion.")
    args = parser.parse_args()

    batch_id = args.batch_id
    launch_path = args.launch
    if batch_id is None and launch_path is None:
        raise SystemExit("Provide either --batch-id or --launch")
    if launch_path is not None:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        batch_id = batch_id or launch.get("batch_id")
        if not batch_id:
            raise SystemExit(f"No batch_id found in launch file: {launch_path}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    status_path = None
    if launch_path is not None:
        status_path = launch_path.with_name(launch_path.stem.replace("_launch_", "_status_") + ".json")

    while True:
        batch = client.batches.retrieve(batch_id)
        snapshot = _status_snapshot(batch)
        if status_path is not None:
            status_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))

        if args.watch_seconds <= 0 or snapshot["status"] in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(args.watch_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
