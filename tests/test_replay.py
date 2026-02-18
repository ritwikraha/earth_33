"""Test: replay serialization round-trip."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.replay import ReplayLogger
from config_io.utils import load_json


def test_replay_round_trip():
    """Save a replay, load it, and verify contents match."""
    logger = ReplayLogger(seed=42, config_dict={"test": True})

    logger.log_step(
        step=0,
        time_info={"day": 80, "hour": 6},
        agent_state={"pos": {"x": 10, "y": 20}, "hydration": 80.0, "energy": 75.0},
        local_state={"terrain": "PLAINS", "air_temp_c": 22.0},
        action={"action": "MOVE_E", "reason": "exploring"},
        events={"encounter": False, "rain": False},
    )
    logger.log_step(
        step=1,
        time_info={"day": 80, "hour": 7},
        agent_state={"pos": {"x": 11, "y": 20}, "hydration": 78.5, "energy": 73.2},
        local_state={"terrain": "FOREST", "air_temp_c": 21.5},
        action={"action": "FORAGE", "reason": "low energy"},
        events={"encounter": False, "rain": True},
    )

    logger.set_summary({
        "survived_steps": 2,
        "survived_hours": 2.0,
        "cause_of_death": "ALIVE",
    })

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    logger.save(tmp_path)
    loaded = ReplayLogger.load(tmp_path)

    # Verify
    assert loaded["meta"]["seed"] == 42
    assert loaded["config"]["test"] is True
    assert len(loaded["steps"]) == 2
    assert loaded["steps"][0]["action"]["action"] == "MOVE_E"
    assert loaded["steps"][1]["events"]["rain"] is True
    assert loaded["summary"]["survived_steps"] == 2
    assert loaded["summary"]["cause_of_death"] == "ALIVE"

    # Verify JSON round-trip integrity
    raw = json.dumps(logger.data, sort_keys=True)
    raw2 = json.dumps(loaded, sort_keys=True)
    assert raw == raw2, "JSON round-trip mismatch"

    # Cleanup
    Path(tmp_path).unlink()


if __name__ == "__main__":
    test_replay_round_trip()
    print("PASS: test_replay_round_trip")
