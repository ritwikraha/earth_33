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
        self._w = 0.7       # inertia weight
        self._c1 = 1.5      # cognitive coefficient (personal best pull)
        self._c2 = 2.0      # social coefficient (global best pull)
        self._max_vel = 3.0  # velocity clamp

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

            self._velocities = (
                self._w * self._velocities + cognitive + social
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

        # Gentle re-centering: mild pull toward organism to keep swarm relevant
        org_pos = np.array([org_x, org_y], dtype=float)
        self._positions += 0.05 * (org_pos[None, :] - self._positions)

        # Clamp positions after re-centering
        if self._world_w > 0:
            self._positions[:, 0] = np.clip(
                self._positions[:, 0], 0, self._world_w - 1,
            )
            self._positions[:, 1] = np.clip(
                self._positions[:, 1], 0, self._world_h - 1,
            )

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
