"""Episode metrics computation."""

from __future__ import annotations

from typing import Any


def compute_metrics(replay_data: dict) -> dict[str, Any]:
    """Compute all metrics from a replay dict."""
    steps = replay_data.get("steps", [])
    summary = replay_data.get("summary", {})
    config = replay_data.get("config", {})
    dt = config.get("sim", {}).get("dt_hours", 1.0)

    n_steps = len(steps)
    survived_hours = n_steps * dt
    survived_days = survived_hours / 24.0

    # Track visited cells
    visited: set[tuple[int, int]] = set()
    near_death_count = 0
    total_wildlife_risk = 0.0
    invalid_action_count = 0
    encounter_count = 0

    for s in steps:
        agent = s.get("agent", {})
        pos = agent.get("pos", {})
        x, y = pos.get("x", 0), pos.get("y", 0)
        visited.add((x, y))

        # Near death: any vital below 10
        if agent.get("hydration", 100) < 10:
            near_death_count += 1
        if agent.get("energy", 100) < 10:
            near_death_count += 1

        # Wildlife risk at visited cells
        local = s.get("local", {})
        total_wildlife_risk += local.get("wildlife_risk", 0.0) if "wildlife_risk" in local else 0.0

        # Encounters
        events = s.get("events", {})
        if events.get("encounter", False):
            encounter_count += 1

    exploration_rate = len(visited) / max(n_steps, 1)
    avg_risk = total_wildlife_risk / max(n_steps, 1)

    # Trophy and hunter metrics
    trophy_found = summary.get("trophy_found", False)
    hunter_death = summary.get("cause_of_death", "") == "HUNTED"

    return {
        "survived_steps": n_steps,
        "survived_hours": round(survived_hours, 1),
        "survived_days": round(survived_days, 2),
        "cause_of_death": summary.get("cause_of_death", "UNKNOWN"),
        "unique_cells_visited": len(visited),
        "exploration_rate": round(exploration_rate, 4),
        "avg_risk_exposure": round(avg_risk, 4),
        "near_death_count": near_death_count,
        "encounter_count": encounter_count,
        "invalid_action_count": invalid_action_count,
        "trophy_found": trophy_found,
        "hunter_death": hunter_death,
    }
