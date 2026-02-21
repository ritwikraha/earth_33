"""Particle Swarm Optimization (PSO) agent.

Each particle has position, velocity, personal best, and contributes to a
global best.  The organism moves one cell per step toward the global best
position found by the swarm.
"""

from __future__ import annotations

import numpy as np

from agents.swarm_base import SwarmAgentBase
from config_io.config import Config


class PSOAgent(SwarmAgentBase):
    """PSO-based agent: particles explore and the organism follows the swarm."""

    DEFAULT_POP = 10

    def __init__(self, seed: int = 42, config: Config | None = None):
        pop = config.swarm.population_size if config else self.DEFAULT_POP
        radius = config.swarm.search_radius if config else 15
        self._inner_iters = config.swarm.inner_iterations if config else 3
        super().__init__(pop, radius, seed, config)

        # PSO hyper-parameters
        self._w = 0.9       # inertia weight (high to maintain spread)
        self._c1 = 1.2      # cognitive coefficient
        self._c2 = 1.0      # social coefficient (lower to reduce convergence)
        self._max_vel = 8.0  # velocity clamp (high for wide map traversal)

        # Per-particle state (initialised lazily)
        self._positions: np.ndarray | None = None  # (N, 2) float
        self._velocities: np.ndarray | None = None  # (N, 2) float
        self._pbest: np.ndarray | None = None       # personal best pos
        self._pbest_fit: np.ndarray | None = None   # personal best fitness
        self._gbest: np.ndarray | None = None       # global best pos
        self._gbest_fit: float = -1e9

    # ──────────────────────────────────────────────────────────────────

    def _run_swarm_iteration(
        self, org_x: int, org_y: int, observation: dict,
    ) -> tuple[int, int]:
        N = self.pop_size

        # Lazy init
        if self._positions is None:
            self._positions = self._init_population_around(org_x, org_y, N)
            self._velocities = (
                self.rng.uniform(-1, 1, size=(N, 2)) * 2.0
            )
            self._pbest = self._positions.copy()
            self._pbest_fit = np.full(N, -1e9)
            self._gbest = np.array([org_x, org_y], dtype=float)
            self._gbest_fit = -1e9

        # Reset global best each step to avoid stale convergence
        self._gbest_fit = -1e9

        for _ in range(self._inner_iters):
            # Evaluate fitness
            for i in range(N):
                px, py = self._clamp(self._positions[i, 0], self._positions[i, 1])
                fit = self._evaluate_fitness(px, py, org_x, org_y, observation)

                # Update personal best
                if fit > self._pbest_fit[i]:
                    self._pbest_fit[i] = fit
                    self._pbest[i] = self._positions[i].copy()

                # Update global best
                if fit > self._gbest_fit:
                    self._gbest_fit = fit
                    self._gbest = self._positions[i].copy()

            # Update velocities and positions
            r1 = self.rng.random((N, 2))
            r2 = self.rng.random((N, 2))

            cognitive = self._c1 * r1 * (self._pbest - self._positions)
            social = self._c2 * r2 * (self._gbest[None, :] - self._positions)
            # Random exploration component to maintain diversity
            exploration = self.rng.uniform(-2.0, 2.0, size=(N, 2))

            self._velocities = (
                self._w * self._velocities + cognitive + social + exploration
            )
            # Clamp velocity
            self._velocities = np.clip(
                self._velocities, -self._max_vel, self._max_vel,
            )

            self._positions += self._velocities

            # Clamp positions to world bounds
            if self._world_w > 0:
                self._positions[:, 0] = np.clip(
                    self._positions[:, 0], 0, self._world_w - 1,
                )
                self._positions[:, 1] = np.clip(
                    self._positions[:, 1], 0, self._world_h - 1,
                )

        # Respawn particles stuck at boundaries back to random positions
        if self._world_w > 0:
            for i in range(N):
                px, py = self._positions[i]
                at_edge = (px <= 1 or px >= self._world_w - 2 or
                           py <= 1 or py >= self._world_h - 2)
                if at_edge and self.rng.random() < 0.3:
                    self._positions[i] = [
                        self.rng.integers(0, self._world_w),
                        self.rng.integers(0, self._world_h),
                    ]
                    self._velocities[i] = self.rng.uniform(-2, 2, size=2)

        # Store clone positions for renderer
        self._clone_positions = [
            self._clamp(self._positions[i, 0], self._positions[i, 1])
            for i in range(N)
        ]

        bx, by = self._clamp(self._gbest[0], self._gbest[1])
        return bx, by

    def reset(self) -> None:
        super().reset()
        self._positions = None
        self._velocities = None
        self._pbest = None
        self._pbest_fit = None
        self._gbest = None
        self._gbest_fit = -1e9
