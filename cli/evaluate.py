"""CLI command: batch evaluation across seeds."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_io.config import load_config
from eval.runner import run_evaluation


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed spec like '0-9' or '1,3,5' or '0-99'."""
    seeds: list[int] = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate Earth2D survival")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--agent", type=str, default="heuristic",
                        choices=["random", "heuristic", "llm"])
    parser.add_argument("--seeds", type=str, default="0-9", help="Seed range, e.g. '0-9' or '1,3,5'")
    parser.add_argument("--output", type=str, default="runs", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    seeds = parse_seeds(args.seeds)

    logging.info(f"Evaluating: agent={args.agent}, seeds={seeds[0]}-{seeds[-1]} ({len(seeds)} total)")

    summary = run_evaluation(config, args.agent, seeds, args.output)

    print("\n=== Evaluation Summary ===")
    print(f"  Agent: {summary['agent_type']}")
    print(f"  Seeds: {summary['num_seeds']}")
    print(f"  Avg survived steps: {summary['avg_survived_steps']}")
    print(f"  Avg survived hours: {summary['avg_survived_hours']}")
    print(f"  Death causes: {summary['death_causes']}")


if __name__ == "__main__":
    main()
