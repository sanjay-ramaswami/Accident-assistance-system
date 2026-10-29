#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from emergency_dispatcher.slm_guidance import CONSTRAINT_MODES
from emergency_dispatcher.slm_guidance import DECODER_SYSTEM_PROMPT
from emergency_dispatcher.slm_guidance import GUIDANCE_ISSUE_TYPES
from emergency_dispatcher.slm_guidance import GUIDANCE_PRIORITIES
from emergency_dispatcher.slm_guidance import SLM_GUIDANCE_SCHEMA_VERSION
from emergency_dispatcher.slm_guidance import YES_NO_UNKNOWN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "generated" / "batch_jobs" / "slm_guidance_batch_validated_merged_20260425_211031.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "generated" / "batch_jobs"


PIONEER_SYSTEM_PROMPT = DECODER_SYSTEM_PROMPT


def build_decoder_record(example: dict[str, Any]) -> dict[str, Any]:
    assistant_payload = {
        "extracted_fields": example["extracted_fields"],
        "severity_score": example["severity_score"],
        "severity_delta": example["severity_delta"],
        "severity_drivers": example["severity_drivers"],
        "human_recommended": example["human_recommended"],
        "human_required": example["human_required"],
        "agent_should_continue": example["agent_should_continue"],
        "constraint_mode": example["constraint_mode"],
        "address_in_utterance": example["address_in_utterance"],
    }
    user_payload = {
        "previous_severity": example["previous_severity"],
        "raw_utterance": example["raw_utterance"],
    }
    return {
        "messages": [
            {"role": "system", "content": PIONEER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the validated SLM guidance dataset into Pioneer decoder JSONL format.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Validated merged SLM guidance dataset.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    decoder_rows = [build_decoder_record(row) for row in rows]
    out_path = args.out_dir / f"pioneer_decoder_slm_guidance_{SLM_GUIDANCE_SCHEMA_VERSION}.jsonl"
    write_jsonl(out_path, decoder_rows)

    print(
        json.dumps(
            {
                "schema_version": SLM_GUIDANCE_SCHEMA_VERSION,
                "count_exported": len(decoder_rows),
                "input_path": str(args.input),
                "pioneer_decoder_dataset_path": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
