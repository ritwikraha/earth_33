"""Test: deterministic world generation and episode results given same seed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config_io.config import load_config
from sim.world import WorldState
from sim.env import Environment
from config_io.schema import ActionType
from agents.heuristic_agent import HeuristicAgent


def test_world_determinism():
    """Two worlds with same seed must produce identical arrays."""
    config = load_config()
    seed = 12345

    w1 = WorldState(config, seed)
    w2 = WorldState(config, seed)

    np.testing.assert_array_equal(w1.elevation, w2.elevation)
    np.testing.assert_array_equal(w1.terrain_id, w2.terrain_id)
    np.testing.assert_array_equal(w1.water_mask, w2.water_mask)
    np.testing.assert_array_equal(w1.vegetation_biomass, w2.vegetation_biomass)
    np.testing.assert_array_equal(w1.wildlife_risk, w2.wildlife_risk)
    np.testing.assert_array_equal(w1.shelter_quality, w2.shelter_quality)
    np.testing.assert_array_equal(w1.movement_cost, w2.movement_cost)
    np.testing.assert_array_equal(w1.dist_to_water, w2.dist_to_water)


def test_episode_determinism():
    """Two episodes with same seed and heuristic agent must have identical outcomes."""
    config = load_config()
    seed = 999

    def run(s: int) -> dict:
        env = Environment(config, s)
        agent = HeuristicAgent()
        for _ in range(min(100, config.sim.max_steps)):
            if not env.organism.alive:
                break
            obs = env.get_observation()
            action = agent.act(obs)
            env.step(action.action)
        env.finalize_replay()
        return env.replay.data["summary"]

    s1 = run(seed)
    s2 = run(seed)

    assert s1["survived_steps"] == s2["survived_steps"], (
        f"Steps differ: {s1['survived_steps']} vs {s2['survived_steps']}"
    )
    assert s1["cause_of_death"] == s2["cause_of_death"], (
        f"Cause differs: {s1['cause_of_death']} vs {s2['cause_of_death']}"
    )


if __name__ == "__main__":
    test_world_determinism()
    print("PASS: test_world_determinism")
    test_episode_determinism()
    print("PASS: test_episode_determinism")
