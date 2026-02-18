"""CLI command: run a single episode."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_io.config import load_config
from eval.runner import run_episode, make_agent
from eval.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Earth2D survival episode")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent", type=str, default="heuristic",
                        choices=["random", "heuristic", "llm"])
    parser.add_argument("--headless", action="store_true", help="Run without pygame window")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    overrides = {}
    if args.max_steps is not None:
        overrides["sim"] = {"max_steps": args.max_steps}

    config = load_config(args.config, overrides)
    agent = make_agent(args.agent, seed=args.seed, config=config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = f"runs/{timestamp}_seed{args.seed}.json"

    Path("runs").mkdir(exist_ok=True)

    logging.info(f"Running episode: agent={args.agent}, seed={args.seed}, "
                 f"map={config.map.width}x{config.map.height}, max_steps={config.sim.max_steps}")

    replay_data = run_episode(
        config, args.seed, agent,
        headless=args.headless,
        replay_path=replay_path,
    )

    metrics = compute_metrics(replay_data)
    summary = replay_data.get("summary", {})

    print("\n=== Episode Summary ===")
    print(f"  Survived: {metrics['survived_steps']} steps "
          f"({metrics['survived_hours']} hours / {metrics['survived_days']} days)")
    print(f"  Cause of death: {summary.get('cause_of_death', 'N/A')}")
    print(f"  Cells explored: {metrics['unique_cells_visited']}")
    print(f"  Exploration rate: {metrics['exploration_rate']:.3f}")
    print(f"  Near-death events: {metrics['near_death_count']}")
    print(f"  Wildlife encounters: {metrics['encounter_count']}")
    print(f"  Replay saved: {replay_path}")


if __name__ == "__main__":
    main()
