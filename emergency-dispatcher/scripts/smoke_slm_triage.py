#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from emergency_dispatcher.gradbot_adapter import build_session_config
from emergency_dispatcher.settings import get_settings
from emergency_dispatcher.triage import analyze_session_triage


def _build_session(utterances: list[str]) -> dict[str, Any]:
    return {
        "session_id": "slm-smoke",
        "client": {
            "transcripts": {
                "user": [{"text": text} for text in utterances],
                "agent": [],
            }
        },
        "triage_turns": [],
        "merged_triage_state": {},
        "triage_meta": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-pass triage smoke test through the live triage adapter.")
    parser.add_argument(
        "--text",
        action="append",
        dest="utterances",
        help="Caller utterance. Pass multiple times to simulate multiple turns.",
    )
    parser.add_argument("--show-prompt-tail", type=int, default=24, help="How many prompt lines to print from the tail.")
    args = parser.parse_args()

    utterances = args.utterances or [
        "My apartment kitchen is filling with smoke and my son is still inside on the third floor. I can't get back in.",
    ]
    session = _build_session(utterances)
    settings = get_settings()
    update = analyze_session_triage(session, settings)
    if update is None:
        raise SystemExit("No triage update produced.")

    print("=== TRIAGE UPDATE ===")
    print(json.dumps(update.model_dump(), indent=2, ensure_ascii=False))
    print()
    print("=== PROMPT TAIL ===")
    config = build_session_config(triage_context=update.prompt_context)
    for line in config.instructions.splitlines()[-args.show_prompt_tail :]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
