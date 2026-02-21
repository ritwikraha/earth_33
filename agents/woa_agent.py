"""Whale Optimization Algorithm (WOA) agent.

Whales hunt using two mechanisms:
  1. Shrinking encircling — tighten circle around the best prey.
  2. Spiral bubble-net — logarithmic spiral path toward prey.
Random exploration occurs when the search vector is large.
"""

from __future__ import annotations

import math

import numpy as np

from agents.swarm_base import SwarmAgentBase
from config_io.config import Config


class WOAAgent(SwarmAgentBase):
    """Whale Optimization Algorithm agent."""

    DEFAULT_POP = 6

    def __init__(self, seed: int = 42, config: Config | None = None):
        pop = config.swarm.population_size if config else self.DEFAULT_POP
        radius = config.swarm.search_radius if config else 15
        self._inner_iters = config.swarm.inner_iterations if config else 3
        super().__init__(pop, radius, seed, config)

        self._whales: np.ndarray | None = None  # (N, 2)
        self._best_whale: np.ndarray | None = None
        self._best_fit: float = -1e9
        self._spiral_b: float = 1.0  # spiral constant
        self._global_iter: int = 0
        self._max_global_iter: int = 200

    # ──────────────────────────────────────────────────────────────────

    def _run_swarm_iteration(
        self, org_x: int, org_y: int, observation: dict,
    ) -> tuple[int, int]:
        N = self.pop_size

        # Lazy init
        if self._whales is None:
            self._whales = self._init_population_around(org_x, org_y, N)
            self._best_whale = np.array([org_x, org_y], dtype=float)

        for _ in range(self._inner_iters):
            # Evaluate fitness and find best
            for i in range(N):
                px, py = self._clamp(self._whales[i, 0], self._whales[i, 1])
                fit = self._evaluate_fitness(px, py, org_x, org_y, observation)
                if fit > self._best_fit:
                    self._best_fit = fit
                    self._best_whale = self._whales[i].copy()

            # Decreasing 'a' from 2 to 0
            a = 2.0 * (1.0 - self._global_iter / self._max_global_iter)
            a = max(a, 0.0)

            for i in range(N):
                r = self.rng.random()
                A = 2.0 * a * r - a
                C = 2.0 * self.rng.random()
                p = self.rng.random()

                if p < 0.5:
                    if abs(A) < 1.0:
                        # Shrinking encircle: move toward best
                        D = np.abs(C * self._best_whale - self._whales[i])
                        self._whales[i] = self._best_whale - A * D
                    else:
                        # Exploration: move toward random whale
                        rand_idx = self.rng.integers(0, N)
                        rand_whale = self._whales[rand_idx]
                        D = np.abs(C * rand_whale - self._whales[i])
                        self._whales[i] = rand_whale - A * D
                else:
                    # Spiral bubble-net
                    D_prime = np.abs(self._best_whale - self._whales[i])
                    l = self.rng.uniform(-1.0, 1.0)
                    exp_term = math.exp(self._spiral_b * l)
                    cos_term = math.cos(2.0 * math.pi * l)
                    self._whales[i] = (
                        D_prime * exp_term * cos_term + self._best_whale
                    )

            # Clamp
            if self._world_w > 0:
                self._whales[:, 0] = np.clip(
                    self._whales[:, 0], 0, self._world_w - 1,
                )
                self._whales[:, 1] = np.clip(
                    self._whales[:, 1], 0, self._world_h - 1,
                )

            self._global_iter += 1

        # Clone positions
        self._clone_positions = [
            self._clamp(self._whales[i, 0], self._whales[i, 1])
            for i in range(N)
        ]

        return self._clamp(self._best_whale[0], self._best_whale[1])

    def reset(self) -> None:
        super().reset()
        self._whales = None
        self._best_whale = None
        self._best_fit = -1e9
        self._global_iter = 0
