from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .settings import REPO_ROOT


class Scenario(BaseModel):
    id: str
    title: str
    noise_profile: str
    transcript: str
    expected_issue_type: str
    expected_priority: str
    expected_address: str


SCENARIO_FILE = REPO_ROOT / "data" / "emergency_scenarios.json"


def load_scenarios() -> list[Scenario]:
    payload = json.loads(SCENARIO_FILE.read_text())
    return [Scenario.model_validate(item) for item in payload]


def get_scenario(scenario_id: str) -> Scenario:
    for scenario in load_scenarios():
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario id: {scenario_id}")

