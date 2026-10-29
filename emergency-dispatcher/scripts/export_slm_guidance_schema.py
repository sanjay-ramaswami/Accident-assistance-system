#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emergency_dispatcher.slm_guidance import SLM_GUIDANCE_SCHEMA_VERSION
from emergency_dispatcher.slm_guidance import schema_snapshots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "schema"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export versioned JSON schema snapshots for the SLM guidance dataset.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for schema files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = schema_snapshots()

    seed_path = args.out_dir / f"slm_guidance_seed_{SLM_GUIDANCE_SCHEMA_VERSION}.json"
    record_path = args.out_dir / f"slm_guidance_record_{SLM_GUIDANCE_SCHEMA_VERSION}.json"
    manifest_path = args.out_dir / f"slm_guidance_schema_manifest_{SLM_GUIDANCE_SCHEMA_VERSION}.json"

    seed_path.write_text(json.dumps(snapshots["seed"], indent=2, ensure_ascii=False), encoding="utf-8")
    record_path.write_text(json.dumps(snapshots["record"], indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SLM_GUIDANCE_SCHEMA_VERSION,
                "seed_schema_path": str(seed_path),
                "record_schema_path": str(record_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": SLM_GUIDANCE_SCHEMA_VERSION,
                "seed_schema_path": str(seed_path),
                "record_schema_path": str(record_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
