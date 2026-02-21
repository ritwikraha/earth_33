"""Shared base class for swarm/bio-inspired optimization agents."""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np
from numpy.random import Generator

from agents.base import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from config_io.config import Config
from config_io.schema import AgentAction, ActionType


class SwarmAgentBase(BaseAgent):
    """Base class for all swarm-based optimisation agents.

    Handles:
    - World data caching from observation snapshots
    - Hunter position memory (from visible_hunters)
    - Trophy direction estimation (from trophy hints)
    - Shared fitness function for evaluating candidate positions
    - Survival override (delegate to heuristic when vitals are critical)
    - Target-position-to-action translation
    - Clone position storage for renderer
    """

    def __init__(
        self,
        population_size: int,
        search_radius: int,
        seed: int,
        config: Config | None = None,
    ):
        self.rng: Generator = np.random.default_rng(seed)
        self.pop_size = population_size
        self.search_radius = search_radius
        self.config = config

        # ── World cache ──────────────────────────────────────────────
        self._world_w: int = 0
        self._world_h: int = 0
        self._movement_cost: np.ndarray | None = None
        self._water_avail: np.ndarray | None = None
        self._vegetation: np.ndarray | None = None
        self._wildlife_risk: np.ndarray | None = None
        self._water_mask: np.ndarray | None = None
        self._elevation: np.ndarray | None = None

        # ── Internal state ───────────────────────────────────────────
        self._known_hunters: dict[int, dict] = {}
        self._trophy_dir: np.ndarray = np.array([0.0, 0.0])  # (dx, dy) estimate
        self._trophy_magnitude: float = 1.0  # confidence in direction
        self._last_trophy_dist: str | None = None  # for warm/cold tracking
        self._clone_positions: list[tuple[int, int]] = []
        self._heuristic = HeuristicAgent()
        self._step_count: int = 0

    # ── Cache management ─────────────────────────────────────────────

    def _update_world_cache(self, observation: dict) -> None:
        """Cache world grid arrays from the observation's world_snapshot."""
        snapshot = observation.get("world_snapshot")
        if snapshot is None:
            return
        if self._movement_cost is None:
            self._world_w = snapshot["width"]
            self._world_h = snapshot["height"]
            self._movement_cost = snapshot["movement_cost"]
            self._water_avail = snapshot["water_availability"]
            self._vegetation = snapshot["vegetation_biomass"]
            self._wildlife_risk = snapshot["wildlife_risk"]
            self._water_mask = snapshot["water_mask"]
            self._elevation = snapshot["elevation"]
        else:
            # Update dynamic arrays (vegetation/water can change over time)
            self._water_avail = snapshot["water_availability"]
            self._vegetation = snapshot["vegetation_biomass"]

    def _update_hunter_memory(self, observation: dict) -> None:
        """Track visible hunters; remember last known positions."""
        visible = observation.get("visible_hunters", [])
        for h in visible:
            hid = h["id"]
            self._known_hunters[hid] = {
                "x": h["pos"]["x"],
                "y": h["pos"]["y"],
                "distance": h["distance"],
                "is_chasing": h.get("is_chasing", False),
                "last_seen": self._step_count,
            }
        # Decay: forget hunters not seen for 30 steps
        stale_ids = [
            hid for hid, info in self._known_hunters.items()
            if self._step_count - info["last_seen"] > 30
        ]
        for hid in stale_ids:
            del self._known_hunters[hid]

    def _update_trophy_estimate(self, observation: dict) -> None:
        """Refine trophy direction estimate from hints."""
        trophy = observation.get("trophy")
        if trophy is None:
            return

        # Directional hint
        direction = trophy.get("trophy_direction")
        if direction:
            dir_map = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
            if direction in dir_map:
                dx, dy = dir_map[direction]
                # Blend: 70% new direction, 30% old estimate
                self._trophy_dir = 0.3 * self._trophy_dir + 0.7 * np.array(
                    [dx, dy], dtype=float
                )
                norm = np.linalg.norm(self._trophy_dir)
                if norm > 0:
                    self._trophy_dir /= norm

        # Warm/cold feedback adjusts confidence
        warmth = trophy.get("warm_cold")
        if warmth == "warmer":
            self._trophy_magnitude = min(3.0, self._trophy_magnitude * 1.2)
        elif warmth == "colder":
            # Reverse direction slightly
            self._trophy_magnitude = max(0.3, self._trophy_magnitude * 0.7)

        # Distance hint adjusts magnitude
        dist_hint = trophy.get("distance")
        if dist_hint:
            dist_scale = {
                "very_close": 3.0,
                "close": 2.0,
                "moderate": 1.0,
                "far": 0.5,
            }
            self._trophy_magnitude = dist_scale.get(dist_hint, 1.0)

    # ── Fitness function ─────────────────────────────────────────────

    def _evaluate_fitness(
        self, x: int, y: int, org_x: int, org_y: int, obs: dict,
    ) -> float:
        """Evaluate fitness of candidate position (x, y).

        Higher = better. Components:
        1. Trophy attraction (follow estimated direction)
        2. Safety (avoid known hunters)
        3. Resources (water + food, weighted by vitals urgency)
        4. Terrain cost (penalize expensive cells)
        5. Water tile avoidance
        """
        # Out of bounds → extremely bad
        if x < 0 or x >= self._world_w or y < 0 or y >= self._world_h:
            return -1000.0

        # Water tile → impassable for organism
        if self._water_mask is not None and self._water_mask[y, x]:
            return -500.0

        score = 0.0
        agent = obs.get("agent", {})
        hydration = agent.get("hydration", 50.0)
        energy = agent.get("energy", 50.0)

        dx = x - org_x
        dy = y - org_y

        # ── 1. Trophy attraction ─────────────────────────────────────
        if np.linalg.norm(self._trophy_dir) > 0.01:
            trophy_score = (
                dx * self._trophy_dir[0] + dy * self._trophy_dir[1]
            ) * self._trophy_magnitude
            score += trophy_score * 2.0

        # ── 3. Safety — avoid hunters ────────────────────────────────
        for hinfo in self._known_hunters.values():
            hx, hy = hinfo["x"], hinfo["y"]
            dist = math.sqrt((x - hx) ** 2 + (y - hy) ** 2)
            est_radius = 7.0  # conservative estimate of detection radius
            if dist < est_radius:
                # Exponential penalty inside detection zone
                penalty = (est_radius - dist) ** 2
                if hinfo.get("is_chasing"):
                    penalty *= 2.0
                score -= penalty * 1.5

        # ── 4. Resources — weighted by vitals urgency ────────────────
        if self._water_avail is not None:
            water_urgency = max(0.0, (60.0 - hydration) / 60.0)
            score += self._water_avail[y, x] * water_urgency * 5.0

        if self._vegetation is not None:
            food_urgency = max(0.0, (60.0 - energy) / 60.0)
            score += self._vegetation[y, x] * food_urgency * 4.0

        # ── 5. Terrain cost penalty ──────────────────────────────────
        if self._movement_cost is not None:
            score -= (self._movement_cost[y, x] - 1.0) * 2.0

        # ── 6. Wildlife risk penalty ─────────────────────────────────
        if self._wildlife_risk is not None:
            score -= self._wildlife_risk[y, x] * 3.0

        return score

    # ── Survival override ────────────────────────────────────────────

    def _survival_override(self, observation: dict) -> AgentAction | None:
        """Delegate to heuristic agent when vitals are critically low."""
        agent = observation.get("agent", {})
        hydration = agent.get("hydration", 50.0)
        energy = agent.get("energy", 50.0)
        fatigue = agent.get("fatigue", 0.0)
        injury = agent.get("injury", 0.0)

        # Critical thresholds — hand control to survival heuristic
        if hydration < 20 or energy < 20 or fatigue > 85 or injury > 70:
            return self._heuristic.act(observation)

        # Nearby hunter danger — flee
        visible = observation.get("visible_hunters", [])
        if visible:
            nearest = min(visible, key=lambda h: h["distance"])
            if nearest["distance"] <= 3:
                return self._heuristic.act(observation)

        return None

    # ── Action translation ───────────────────────────────────────────

    def _best_position_to_action(
        self,
        best_x: int,
        best_y: int,
        org_x: int,
        org_y: int,
        mask: list[str],
    ) -> AgentAction:
        """Convert a target position into a single-step MOVE_* action."""
        dx = best_x - org_x
        dy = best_y - org_y

        mask_set = set(mask)

        # Determine primary and secondary movement directions
        candidates: list[tuple[ActionType, str]] = []
        if abs(dx) >= abs(dy):
            # Prefer horizontal
            if dx > 0:
                candidates.append((ActionType.MOVE_E, "Moving E toward swarm target"))
            elif dx < 0:
                candidates.append((ActionType.MOVE_W, "Moving W toward swarm target"))
            if dy > 0:
                candidates.append((ActionType.MOVE_S, "Moving S toward swarm target"))
            elif dy < 0:
                candidates.append((ActionType.MOVE_N, "Moving N toward swarm target"))
        else:
            # Prefer vertical
            if dy > 0:
                candidates.append((ActionType.MOVE_S, "Moving S toward swarm target"))
            elif dy < 0:
                candidates.append((ActionType.MOVE_N, "Moving N toward swarm target"))
            if dx > 0:
                candidates.append((ActionType.MOVE_E, "Moving E toward swarm target"))
            elif dx < 0:
                candidates.append((ActionType.MOVE_W, "Moving W toward swarm target"))

        for action, reason in candidates:
            if action.value in mask_set:
                return AgentAction(
                    action=action, reason=reason, confidence=0.7,
                )

        # If no movement valid (or target == current), try any move
        for act_name in ["MOVE_E", "MOVE_S", "MOVE_N", "MOVE_W"]:
            if act_name in mask_set:
                return AgentAction(
                    action=ActionType(act_name),
                    reason="Exploring (no direct path to swarm target)",
                    confidence=0.4,
                )

        # Fallback: REST
        return AgentAction(
            action=ActionType.REST,
            reason="Resting (blocked)",
            confidence=0.3,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        """Clamp position to world bounds and return integer coords."""
        cx = int(np.clip(x, 0, self._world_w - 1))
        cy = int(np.clip(y, 0, self._world_h - 1))
        return cx, cy

    def _init_population_around(
        self, org_x: int, org_y: int, pop_size: int,
    ) -> np.ndarray:
        """Initialise population within search_radius of the organism.

        All clones start within the search radius so the swarm algorithm
        can converge from a coherent starting formation.
        """
        positions = np.zeros((pop_size, 2), dtype=float)
        r = self.search_radius
        for i in range(pop_size):
            dx = self.rng.integers(-r, r + 1)
            dy = self.rng.integers(-r, r + 1)
            positions[i] = self._clamp(org_x + dx, org_y + dy)
        return positions

    # ── Abstract method ──────────────────────────────────────────────

    @abstractmethod
    def _run_swarm_iteration(
        self, org_x: int, org_y: int, observation: dict,
    ) -> tuple[int, int]:
        """Run one step of the swarm algorithm.

        Must update self._clone_positions with current particle/agent
        positions and return the best (x, y) target position.
        """
        ...

    # ── Main act() ───────────────────────────────────────────────────

    def act(self, observation: dict) -> AgentAction:
        """Swarm-based decision: update caches, check survival, run swarm."""
        self._step_count += 1
        self._update_world_cache(observation)
        self._update_hunter_memory(observation)
        self._update_trophy_estimate(observation)

        # Survival takes priority
        override = self._survival_override(observation)
        if override is not None:
            return override

        org_x = observation["agent"]["pos"]["x"]
        org_y = observation["agent"]["pos"]["y"]

        best_x, best_y = self._run_swarm_iteration(org_x, org_y, observation)

        mask = observation.get("action_mask", [a.value for a in ActionType])
        return self._best_position_to_action(best_x, best_y, org_x, org_y, mask)

    def get_clone_positions(self) -> list[tuple[int, int]] | None:
        """Return virtual clone positions for rendering."""
        return self._clone_positions if self._clone_positions else None

    def reset(self) -> None:
        """Reset agent state between episodes."""
        self._clone_positions = []
        self._known_hunters = {}
        self._trophy_dir = np.array([0.0, 0.0])
        self._trophy_magnitude = 1.0
        self._step_count = 0
