#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_slm_fact_ledger_training_data import DEFAULT_BASE_URL  # noqa: E402
from generate_slm_fact_ledger_training_data import DEFAULT_MODEL  # noqa: E402
from generate_slm_fact_ledger_training_data import ScenarioTemplate  # noqa: E402
from generate_slm_fact_ledger_training_data import repair_prompt  # noqa: E402
from generate_slm_fact_ledger_training_data import validate_example  # noqa: E402
from generate_slm_fact_ledger_training_data import write_jsonl  # noqa: E402

from emergency_dispatcher.slm_fact_ledger import FACT_LEDGER_SCHEMA_VERSION  # noqa: E402
from emergency_dispatcher.slm_fact_ledger import FactLedgerTrainingExample  # noqa: E402
from emergency_dispatcher.slm_fact_ledger import build_pioneer_fact_ledger_record  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "data" / "generated" / "batch_jobs"


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        manifest[row["custom_id"]] = row
    return manifest


def _load_output_text(output_path: Path | None, output_file_id: str | None, base_url: str) -> str:
    if output_path is not None:
        return output_path.read_text(encoding="utf-8")
    if output_file_id is None:
        raise SystemExit("Provide either --output-path or --output-file-id")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client.files.retrieve_content(output_file_id)


def _extract_candidate(output_row: dict[str, Any]) -> dict[str, Any]:
    if output_row.get("error"):
        raise ValueError(output_row["error"].get("message", "batch request error"))
    response_body = ((output_row.get("response") or {}).get("body") or {})
    choices = response_body.get("choices") or []
    if not choices:
        raise ValueError("missing choices in batch response")
    content = ((choices[0].get("message") or {}).get("content")) or "{}"
    return json.loads(content)


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Validate and finalize an OpenAI Batch output into the fact-ledger SLM dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL created by prepare_slm_fact_ledger_batch.py")
    parser.add_argument("--output-path", type=Path, help="Local batch output JSONL file")
    parser.add_argument("--output-file-id", help="OpenAI output file id to download")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model label for repair requests.")
    parser.add_argument("--target-count", type=int, default=5000, help="Desired final count to export for Pioneer.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for finalized outputs.")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_id = _load_manifest(args.manifest)
    output_text = _load_output_text(args.output_path, args.output_file_id, args.base_url)

    raw_candidates_path = args.out_dir / f"slm_fact_ledger_batch_raw_{stamp}.jsonl"
    validated_all_path = args.out_dir / f"slm_fact_ledger_batch_validated_{stamp}.jsonl"
    final_dataset_path = args.out_dir / f"slm_fact_ledger_batch_final_{args.target_count}_{stamp}.jsonl"
    pioneer_path = args.out_dir / f"pioneer_decoder_slm_fact_ledger_{FACT_LEDGER_SCHEMA_VERSION}_{args.target_count}_{stamp}.jsonl"
    failures_path = args.out_dir / f"slm_fact_ledger_batch_failures_{stamp}.json"
    repair_path = args.out_dir / f"slm_fact_ledger_batch_repairs_{stamp}.jsonl"

    raw_rows: list[dict[str, Any]] = []
    validated_rows: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []

    for line in output_text.splitlines():
        if not line.strip():
            continue
        output_row = json.loads(line)
        custom_id = output_row.get("custom_id")
        manifest_row = manifest_by_id.get(custom_id)
        if manifest_row is None:
            failures.append({"custom_id": custom_id, "error": "missing manifest row"})
            continue
        index = int(manifest_row["index"])
        plan = ScenarioTemplate(**manifest_row["plan"])
        candidate: dict[str, Any] = {}
        try:
            candidate = _extract_candidate(output_row)
            raw_rows.append({"custom_id": custom_id, "index": index, "candidate": candidate, "plan": asdict(plan)})
            record = FactLedgerTrainingExample.model_validate(candidate).model_dump()
            errors = validate_example(plan, record)
            if errors:
                raise ValueError("; ".join(errors))
            validated_rows.append((index, record))
        except Exception as exc:
            failures.append({"custom_id": custom_id, "index": index, "plan": asdict(plan), "error": str(exc)})
            repair_rows.append(
                {
                    "custom_id": f"repair-{custom_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You create strict fact-ledger emergency dispatch training data. Return JSON only.",
                            },
                            {"role": "user", "content": repair_prompt(plan, candidate, [str(exc)])},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                }
            )

    raw_rows.sort(key=lambda row: row["index"])
    validated_rows.sort(key=lambda item: item[0])
    validated_only = [row for _, row in validated_rows]
    final_rows = validated_only[: args.target_count]
    pioneer_rows = [build_pioneer_fact_ledger_record(row) for row in final_rows]

    write_jsonl(raw_candidates_path, raw_rows)
    write_jsonl(validated_all_path, validated_only)
    write_jsonl(final_dataset_path, final_rows)
    write_jsonl(pioneer_path, pioneer_rows)
    write_jsonl(repair_path, repair_rows)
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "schema_version": FACT_LEDGER_SCHEMA_VERSION,
                "count_validated_total": len(validated_only),
                "count_exported_final": len(final_rows),
                "target_count": args.target_count,
                "enough_for_target": len(final_rows) >= args.target_count,
                "count_failed": len(failures),
                "validated_all_path": str(validated_all_path),
                "final_dataset_path": str(final_dataset_path),
                "pioneer_decoder_dataset_path": str(pioneer_path),
                "raw_candidates_path": str(raw_candidates_path),
                "repair_batch_input_path": str(repair_path),
                "failures_path": str(failures_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
