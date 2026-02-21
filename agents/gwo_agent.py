"""Grey Wolf Optimization (GWO) agent.

Wolves are ranked in a hierarchy: alpha (best), beta (2nd), delta (3rd),
and omega (rest).  Each step the pack converges on prey guided by the
top three wolves.  The organism moves toward the alpha position.
"""

from __future__ import annotations

import numpy as np

from agents.swarm_base import SwarmAgentBase
from config_io.config import Config


class GWOAgent(SwarmAgentBase):
    """Grey Wolf Optimization agent."""

    DEFAULT_POP = 8

    def __init__(self, seed: int = 42, config: Config | None = None):
        pop = config.swarm.population_size if config else self.DEFAULT_POP
        radius = config.swarm.search_radius if config else 15
        self._inner_iters = config.swarm.inner_iterations if config else 3
        super().__init__(pop, radius, seed, config)

        self._wolves: np.ndarray | None = None  # (N, 2) float
        self._global_iter: int = 0
        self._max_global_iter: int = 200  # for 'a' decay across episode

    # ──────────────────────────────────────────────────────────────────

    def _run_swarm_iteration(
        self, org_x: int, org_y: int, observation: dict,
    ) -> tuple[int, int]:
        N = self.pop_size

        # Lazy init / re-center
        if self._wolves is None:
            self._wolves = self._init_population_around(org_x, org_y, N)

        for _ in range(self._inner_iters):
            # Evaluate fitness for each wolf
            fitness = np.array([
                self._evaluate_fitness(
                    *self._clamp(self._wolves[i, 0], self._wolves[i, 1]),
                    org_x, org_y, observation,
                )
                for i in range(N)
            ])

            # Rank: alpha, beta, delta
            ranked = np.argsort(fitness)[::-1]  # descending
            alpha_pos = self._wolves[ranked[0]].copy()
            beta_pos = self._wolves[ranked[min(1, N - 1)]].copy()
            delta_pos = self._wolves[ranked[min(2, N - 1)]].copy()

            # Linearly decreasing parameter 'a' from 2 to 0
            a = 2.0 * (1.0 - self._global_iter / self._max_global_iter)
            a = max(a, 0.0)

            # Update each wolf position
            for i in range(N):
                # Coefficients for alpha
                r1 = self.rng.random(2)
                r2 = self.rng.random(2)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - self._wolves[i])
                X1 = alpha_pos - A1 * D_alpha

                # Coefficients for beta
                r1 = self.rng.random(2)
                r2 = self.rng.random(2)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - self._wolves[i])
                X2 = beta_pos - A2 * D_beta

                # Coefficients for delta
                r1 = self.rng.random(2)
                r2 = self.rng.random(2)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos - self._wolves[i])
                X3 = delta_pos - A3 * D_delta

                # New position: average of three leader influences
                self._wolves[i] = (X1 + X2 + X3) / 3.0

            # Clamp to world bounds
            if self._world_w > 0:
                self._wolves[:, 0] = np.clip(
                    self._wolves[:, 0], 0, self._world_w - 1,
                )
                self._wolves[:, 1] = np.clip(
                    self._wolves[:, 1], 0, self._world_h - 1,
                )

            self._global_iter += 1

        # Store clone positions
        self._clone_positions = [
            self._clamp(self._wolves[i, 0], self._wolves[i, 1])
            for i in range(N)
        ]

        # Return alpha position (best fitness)
        return self._clamp(alpha_pos[0], alpha_pos[1])

    def reset(self) -> None:
        super().reset()
        self._wolves = None
        self._global_iter = 0
