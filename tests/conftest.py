"""Shared fixtures.

The benchmark is loaded once per session. Five hundred scenario files parsed
per test would make the suite slow enough that it stops being run, and a test
suite nobody runs is worse than none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.scenario import ScenarioSet, load_dir


SCENARIOS = Path("bench/scenarios")


@pytest.fixture(scope="session")
def dataset() -> ScenarioSet:
    if not SCENARIOS.exists():
        pytest.skip("no benchmark on disk - run `make gen` first")
    return load_dir(SCENARIOS)


@pytest.fixture(scope="session")
def keyring() -> dict[str, str]:
    import json

    path = SCENARIOS / "_keyring.json"
    if not path.exists():
        pytest.skip("no keyring - run `make gen` first")
    return json.loads(path.read_text(encoding="utf-8"))
