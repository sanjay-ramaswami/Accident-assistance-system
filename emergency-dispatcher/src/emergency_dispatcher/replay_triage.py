from __future__ import annotations

import argparse
import json
from pathlib import Path

from .settings import get_settings
from .triage import replay_triage_from_session_id
from .triage import replay_triage_to_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay triage analysis over a saved session.")
    parser.add_argument("session_id", help="Saved session id without path")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the replay output to runs/replay/<session_id>_triage.json instead of stdout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for --write",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.write:
        path = replay_triage_to_file(
            session_id=args.session_id,
            settings=settings,
            output_path=args.output,
        )
        print(path)
    else:
        payload = replay_triage_from_session_id(args.session_id, settings)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
