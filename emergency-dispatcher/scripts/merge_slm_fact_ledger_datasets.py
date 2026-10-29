#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emergency_dispatcher.slm_fact_ledger import FACT_LEDGER_SCHEMA_VERSION
from emergency_dispatcher.slm_fact_ledger import FactLedgerTrainingExample
from emergency_dispatcher.slm_fact_ledger import build_pioneer_fact_ledger_record


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "generated" / "batch_jobs"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple validated fact-ledger datasets into one final dataset and Pioneer decoder file."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Validated fact-ledger JSONL files.")
    parser.add_argument("--target-count", type=int, default=5000, help="Desired merged final count.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Do not deduplicate identical rows across shards. Useful when you need an immediate fixed-size training file.",
    )
    args = parser.parse_args()

    seen_keys: set[str] = set()
    merged_rows: list[dict] = []
    duplicate_rows_skipped = 0

    for input_path in args.inputs:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                normalized = FactLedgerTrainingExample.model_validate(row).model_dump()
                dedupe_key = json.dumps(
                    {
                        "caller_turn": normalized["caller_turn"],
                        "prior_facts": normalized["prior_facts"],
                        "merged_facts": normalized["merged_facts"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if not args.allow_duplicates:
                    if dedupe_key in seen_keys:
                        duplicate_rows_skipped += 1
                        continue
                    seen_keys.add(dedupe_key)
                merged_rows.append(normalized)
                if len(merged_rows) >= args.target_count:
                    break
        if len(merged_rows) >= args.target_count:
            break

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "merged_nodedupe" if args.allow_duplicates else "merged"
    final_dataset_path = args.out_dir / f"slm_fact_ledger_{suffix}_final_{args.target_count}_{stamp}.jsonl"
    pioneer_path = args.out_dir / f"pioneer_decoder_slm_fact_ledger_{FACT_LEDGER_SCHEMA_VERSION}_{args.target_count}_{suffix}_{stamp}.jsonl"
    pioneer_rows = [build_pioneer_fact_ledger_record(row) for row in merged_rows]

    write_jsonl(final_dataset_path, merged_rows)
    write_jsonl(pioneer_path, pioneer_rows)

    print(
        json.dumps(
            {
                "schema_version": FACT_LEDGER_SCHEMA_VERSION,
                "count_merged": len(merged_rows),
                "target_count": args.target_count,
                "enough_for_target": len(merged_rows) >= args.target_count,
                "allow_duplicates": args.allow_duplicates,
                "duplicate_rows_skipped": duplicate_rows_skipped,
                "final_dataset_path": str(final_dataset_path),
                "pioneer_decoder_dataset_path": str(pioneer_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
