#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_slm_fact_ledger_training_data import DEFAULT_BASE_URL  # noqa: E402
from generate_slm_fact_ledger_training_data import DEFAULT_MODEL  # noqa: E402
from generate_slm_fact_ledger_training_data import build_plan  # noqa: E402
from generate_slm_fact_ledger_training_data import example_prompt  # noqa: E402
from generate_slm_fact_ledger_training_data import write_jsonl  # noqa: E402

from emergency_dispatcher.berlin_location_lexicon import DEFAULT_BERLIN_GEOJSON  # noqa: E402
from emergency_dispatcher.berlin_location_lexicon import DEFAULT_BERLIN_LEXICON_CACHE  # noqa: E402
from emergency_dispatcher.berlin_location_lexicon import load_or_build_berlin_location_lexicon  # noqa: E402
from emergency_dispatcher.slm_fact_ledger import FACT_LEDGER_SCHEMA_VERSION  # noqa: E402


ROOT = SCRIPT_DIR.parents[0]
DEFAULT_OUT_DIR = ROOT / "data" / "generated" / "batch_jobs"


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Prepare and optionally start an OpenAI Batch job for Berlin-seeded fact-ledger SLM data."
    )
    parser.add_argument("--target-count", type=int, default=5000, help="Desired final validated example count.")
    parser.add_argument("--request-count", type=int, help="Exact number of requests to submit. Defaults to target-count * oversample.")
    parser.add_argument("--oversample", type=float, default=1.2, help="Oversample multiplier used when request-count is omitted.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--seed", type=int, default=11, help="Deterministic seed for plan and Berlin seed selection.")
    parser.add_argument("--berlin-geojson", type=Path, default=DEFAULT_BERLIN_GEOJSON, help="Berlin address GeoJSON source.")
    parser.add_argument("--berlin-cache", type=Path, default=DEFAULT_BERLIN_LEXICON_CACHE, help="Cached compact Berlin lexicon JSON.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for manifest and batch input.")
    parser.add_argument("--start", action="store_true", help="Upload the input file and create the batch immediately.")
    args = parser.parse_args()

    request_count = args.request_count or max(args.target_count, math.ceil(args.target_count * args.oversample))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lexicon = load_or_build_berlin_location_lexicon(
        args.berlin_geojson,
        cache_path=args.berlin_cache,
        seed=args.seed,
    )

    batch_input_path = args.out_dir / f"slm_fact_ledger_batch_input_{stamp}.jsonl"
    manifest_path = args.out_dir / f"slm_fact_ledger_batch_manifest_{stamp}.jsonl"
    launch_path = args.out_dir / f"slm_fact_ledger_batch_launch_{stamp}.json"

    request_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for index in range(request_count):
        plan = build_plan(index, lexicon, seed=args.seed)
        custom_id = f"slm-fact-ledger-{index:05d}"
        request_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You create strict fact-ledger emergency dispatch training data. Return JSON only.",
                        },
                        {"role": "user", "content": example_prompt(plan, index)},
                    ],
                    "response_format": {"type": "json_object"},
                },
            }
        )
        manifest_rows.append(
            {
                "custom_id": custom_id,
                "index": index,
                "target_count": args.target_count,
                "schema_version": FACT_LEDGER_SCHEMA_VERSION,
                "model": args.model,
                "plan": asdict(plan),
            }
        )

    write_jsonl(batch_input_path, request_rows)
    write_jsonl(manifest_path, manifest_rows)

    launch_payload: dict[str, object] = {
        "schema_version": FACT_LEDGER_SCHEMA_VERSION,
        "target_count": args.target_count,
        "request_count": request_count,
        "oversample": args.oversample,
        "model": args.model,
        "berlin_lexicon_cache_path": str(args.berlin_cache),
        "berlin_lexicon_stats": lexicon.get("stats"),
        "batch_input_path": str(batch_input_path),
        "manifest_path": str(manifest_path),
        "started": False,
    }

    if args.start:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Missing OPENAI_API_KEY in environment or .env")
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        upload = client.files.create(file=batch_input_path.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "job": "slm_fact_ledger_generation",
                "schema_version": FACT_LEDGER_SCHEMA_VERSION,
                "target_count": str(args.target_count),
                "request_count": str(request_count),
                "model": args.model,
            },
        )
        launch_payload.update(
            {
                "started": True,
                "input_file_id": upload.id,
                "batch_id": batch.id,
                "batch_status": batch.status,
            }
        )

    launch_path.write_text(json.dumps(launch_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**launch_payload, "launch_path": str(launch_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
