"""Ant Colony Optimization (ACO) agent.

Ants lay pheromone trails on the grid.  Each step a colony of ants walks
from the organism's position, choosing neighbours probabilistically based
on pheromone intensity and a heuristic desirability (fitness).  The best
ant's path is reinforced with pheromone; all pheromone evaporates slowly.
"""

from __future__ import annotations

import numpy as np

from agents.swarm_base import SwarmAgentBase
from config_io.config import Config


class ACOAgent(SwarmAgentBase):
    """Ant Colony Optimization agent."""

    DEFAULT_POP = 12

    def __init__(self, seed: int = 42, config: Config | None = None):
        pop = config.swarm.population_size if config else self.DEFAULT_POP
        radius = config.swarm.search_radius if config else 15
        super().__init__(pop, radius, seed, config)

        # ACO hyper-parameters
        self._alpha = 1.0          # pheromone influence
        self._beta = 2.0           # heuristic influence
        self._decay = 0.1          # evaporation rate
        self._deposit = 1.0        # pheromone deposit amount
        self._min_pher = 0.1
        self._max_pher = 10.0
        self._walk_steps: int = max(radius, 30)  # steps each ant walks

        # Pheromone grid (lazy init)
        self._pheromone: np.ndarray | None = None

    # ── Helpers ───────────────────────────────────────────────────────

    _DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N S E W

    def _ant_walk(
        self, start_x: int, start_y: int, org_x: int, org_y: int,
        observation: dict,
    ) -> tuple[list[tuple[int, int]], float]:
        """Single ant random walk. Returns (path, final_fitness)."""
        x, y = start_x, start_y
        path: list[tuple[int, int]] = [(x, y)]

        for _ in range(self._walk_steps):
            # Collect valid neighbours and their attractiveness
            neighbours: list[tuple[int, int]] = []
            probs: list[float] = []

            for dx, dy in self._DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self._world_w and 0 <= ny < self._world_h:
                    pher = self._pheromone[ny, nx] ** self._alpha
                    # Heuristic: use fitness as desirability (shift to positive)
                    fit = self._evaluate_fitness(
                        nx, ny, org_x, org_y, observation,
                    )
                    heur = max(0.01, fit + 100.0) ** self._beta
                    neighbours.append((nx, ny))
                    probs.append(pher * heur)

            if not neighbours:
                break

            # Normalise probabilities
            total = sum(probs)
            if total <= 0:
                # Uniform random
                idx = int(self.rng.integers(0, len(neighbours)))
            else:
                probs_arr = np.array(probs) / total
                idx = int(self.rng.choice(len(neighbours), p=probs_arr))

            x, y = neighbours[idx]
            path.append((x, y))

        final_fit = self._evaluate_fitness(x, y, org_x, org_y, observation)
        return path, final_fit

    # ──────────────────────────────────────────────────────────────────

    def _run_swarm_iteration(
        self, org_x: int, org_y: int, observation: dict,
    ) -> tuple[int, int]:
        N = self.pop_size

        # Lazy-init pheromone grid
        if self._pheromone is None and self._world_h > 0:
            self._pheromone = np.ones(
                (self._world_h, self._world_w), dtype=float,
            )

        if self._pheromone is None:
            # World snapshot not yet available — fall back
            self._clone_positions = []
            return org_x, org_y

        # Run N ants — some from organism, others from random map positions
        best_path: list[tuple[int, int]] = []
        best_fit = -1e9
        ant_finals: list[tuple[int, int]] = []

        for i in range(N):
            if i < N // 3:
                # First third start from organism (local search)
                sx, sy = org_x, org_y
            else:
                # Rest start from random map positions (global exploration)
                sx = int(self.rng.integers(0, self._world_w))
                sy = int(self.rng.integers(0, self._world_h))
            path, fit = self._ant_walk(sx, sy, org_x, org_y, observation)
            ant_finals.append(path[-1])
            if fit > best_fit:
                best_fit = fit
                best_path = path

        # Deposit pheromone on best path
        deposit_val = self._deposit * max(0.01, best_fit + 100.0) * 0.01
        for px, py in best_path:
            self._pheromone[py, px] += deposit_val

        # Evaporate
        self._pheromone *= (1.0 - self._decay)

        # Clamp pheromone
        np.clip(self._pheromone, self._min_pher, self._max_pher,
                out=self._pheromone)

        # Clone positions = final positions of all ants
        self._clone_positions = ant_finals

        # Return the best ant's final position
        if best_path:
            return best_path[-1]
        return org_x, org_y

    def reset(self) -> None:
        super().reset()
        self._pheromone = None
