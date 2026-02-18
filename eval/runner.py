"""Episode runner: run single or batched episodes headless."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from config_io.config import Config
from config_io.schema import ActionType
from config_io.utils import save_json, ensure_dir
from eval.metrics import compute_metrics
from sim.env import Environment

logger = logging.getLogger(__name__)


def make_agent(agent_type: str, seed: int = 0, config: Config | None = None) -> BaseAgent:
    """Factory for agent creation."""
    if agent_type == "random":
        return RandomAgent(seed=seed)
    elif agent_type == "heuristic":
        return HeuristicAgent()
    elif agent_type == "llm":
        from agents.llm_agent import LLMAgent
        cfg = config.agent if config else None
        return LLMAgent(
            provider=cfg.llm_provider if cfg else "openai",
            model=cfg.llm_model if cfg else "gpt-4o-mini",
            max_retries=cfg.llm_max_retries if cfg else 2,
            temperature=cfg.llm_temperature if cfg else 0.7,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode(
    config: Config,
    seed: int,
    agent: BaseAgent,
    headless: bool = True,
    replay_path: str | None = None,
) -> dict[str, Any]:
    """Run a single episode. Returns replay data dict."""
    env = Environment(config, seed)
    renderer = None

    if not headless:
        from render.pygame_renderer import PygameRenderer
        renderer = PygameRenderer(config, env.world)

    step = 0
    last_action = ""
    last_events: list[str] = []

    try:
        while step < config.sim.max_steps and env.organism.alive and not env.trophy_won:
            # Render if visual
            if renderer is not None:
                from sim.dynamics import get_time_info
                time_info = get_time_info(step, config)
                if not renderer.handle_events():
                    break

                # Gather hunter and trophy data for rendering
                hunters_for_render = env.hunter_manager.get_all_hunter_positions()
                trophy_pos = None
                if env.trophy_manager.tcfg.enabled and not env.trophy_manager.found:
                    trophy_pos = (env.trophy_manager.trophy_x, env.trophy_manager.trophy_y)

                renderer.render(
                    env.organism, step, time_info,
                    last_action, last_events,
                    hunters=hunters_for_render if hunters_for_render else None,
                    trophy_pos=trophy_pos,
                )

            # Get observation and act
            obs = env.get_observation()
            action_result = agent.act(obs)
            action = action_result.action

            # Step environment
            result = env.step(action)
            last_action = action.value
            last_events = result["events"]
            step += 1

            if result["dead"] or result.get("trophy_won", False):
                break

    finally:
        if renderer is not None:
            # Show final frame briefly
            from sim.dynamics import get_time_info
            time_info = get_time_info(step, config)

            hunters_for_render = env.hunter_manager.get_all_hunter_positions()
            trophy_pos = None
            if env.trophy_manager.tcfg.enabled:
                trophy_pos = (env.trophy_manager.trophy_x, env.trophy_manager.trophy_y)

            renderer.render(
                env.organism, step, time_info,
                last_action, last_events,
                hunters=hunters_for_render if hunters_for_render else None,
                trophy_pos=trophy_pos,
            )
            import time
            time.sleep(1.0)
            renderer.close()

    # Finalize
    env.finalize_replay()
    replay_data = env.replay.data

    if replay_path:
        env.replay.save(replay_path)
        logger.info(f"Replay saved to {replay_path}")

    return replay_data


def run_evaluation(
    config: Config,
    agent_type: str,
    seeds: list[int],
    output_dir: str = "runs",
) -> dict[str, Any]:
    """Run multiple episodes and produce metrics CSV + summary."""
    out = ensure_dir(output_dir)
    all_metrics: list[dict] = []

    for seed in seeds:
        logger.info(f"Running seed {seed}...")
        agent = make_agent(agent_type, seed=seed, config=config)
        replay_path = str(out / f"eval_seed{seed}.json")
        replay_data = run_episode(config, seed, agent, headless=True, replay_path=replay_path)
        metrics = compute_metrics(replay_data)
        metrics["seed"] = seed
        all_metrics.append(metrics)
        logger.info(f"  Seed {seed}: survived {metrics['survived_steps']} steps, "
                     f"cause: {metrics['cause_of_death']}")

    # Write CSV
    csv_path = str(out / "evaluation_metrics.csv")
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        logger.info(f"CSV written to {csv_path}")

    # Aggregate summary
    if all_metrics:
        avg_steps = sum(m["survived_steps"] for m in all_metrics) / len(all_metrics)
        avg_hours = sum(m["survived_hours"] for m in all_metrics) / len(all_metrics)
        deaths = {}
        for m in all_metrics:
            cod = m["cause_of_death"]
            deaths[cod] = deaths.get(cod, 0) + 1
        trophy_wins = sum(1 for m in all_metrics if m.get("trophy_found", False))
    else:
        avg_steps = 0
        avg_hours = 0
        deaths = {}
        trophy_wins = 0

    summary = {
        "agent_type": agent_type,
        "num_seeds": len(seeds),
        "avg_survived_steps": round(avg_steps, 1),
        "avg_survived_hours": round(avg_hours, 1),
        "death_causes": deaths,
        "trophy_wins": trophy_wins,
        "all_metrics": all_metrics,
    }
    summary_path = str(out / "evaluation_summary.json")
    save_json(summary, summary_path)
    logger.info(f"Summary written to {summary_path}")

    return summary
