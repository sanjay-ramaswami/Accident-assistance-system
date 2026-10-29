from __future__ import annotations

import argparse
import json

from .dispatch_tools import build_gradbot_tool_defs
from .dispatch_tools import run_dispatch_workflow
from .provider_checks import run_provider_checks
from .scenarios import get_scenario
from .settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lean smoke runner for the emergency dispatcher demo chain."
    )
    parser.add_argument(
        "--scenario",
        default="road-fire-berlin",
        help="Scenario id from data/emergency_scenarios.json",
    )
    parser.add_argument(
        "--transcript",
        help="Optional raw transcript override",
    )
    parser.add_argument(
        "--skip-provider-checks",
        action="store_true",
        help="Skip Gradium and ai-coustics auth/init checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    scenario = get_scenario(args.scenario)
    transcript = args.transcript or scenario.transcript

    payload = {
        "scenario": scenario.model_dump(),
        "tool_defs": [
            {
                "name": name,
                "description": description,
                "parameters_json": parameters_json,
            }
            for name, description, parameters_json in build_gradbot_tool_defs()
        ],
        "workflow_result": run_dispatch_workflow(transcript).to_dict(),
    }
    if not args.skip_provider_checks:
        payload["provider_checks"] = run_provider_checks(settings)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

