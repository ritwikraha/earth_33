"""Trophy objective: a goal item the organism must find to win."""

from __future__ import annotations

import math
from typing import Optional

from numpy.random import Generator

from config_io.config import Config
from config_io.schema import Terrain
from sim.world import WorldState


class TrophyManager:
    """Manages the trophy placement and proximity hints."""

    def __init__(
        self,
        config: Config,
        world: WorldState,
        rng: Generator,
        player_spawn: tuple[int, int],
    ):
        self.config = config
        self.tcfg = config.trophy
        self.world = world
        self.found = False
        self.trophy_x: int = -1
        self.trophy_y: int = -1
        self._prev_distance: float = 9999.0

        if not self.tcfg.enabled:
            return

        self._place_trophy(rng, player_spawn)

    def _place_trophy(
        self,
        rng: Generator,
        player_spawn: tuple[int, int],
    ) -> None:
        """Place trophy at least min_distance_from_spawn away, not on water."""
        px, py = player_spawn
        for _ in range(2000):
            x = int(rng.integers(3, self.world.w - 3))
            y = int(rng.integers(3, self.world.h - 3))
            dist = abs(x - px) + abs(y - py)
            terrain = self.world.terrain_at(x, y)
            if dist >= self.tcfg.min_distance_from_spawn and terrain != Terrain.WATER:
                self.trophy_x, self.trophy_y = x, y
                return
        # Fallback: place at opposite corner area from spawn
        self.trophy_x = max(3, min(self.world.w - 4, self.world.w - px - 1))
        self.trophy_y = max(3, min(self.world.h - 4, self.world.h - py - 1))

    def check_found(self, player_x: int, player_y: int) -> bool:
        """Check if player has reached the trophy (adjacent or same cell)."""
        if not self.tcfg.enabled or self.found:
            return False
        if abs(player_x - self.trophy_x) <= 1 and abs(player_y - self.trophy_y) <= 1:
            self.found = True
            return True
        return False

    def get_hint(self, player_x: int, player_y: int, step: int) -> dict:
        """Generate proximity hint for the agent."""
        if not self.tcfg.enabled:
            return {}

        dist = math.sqrt(
            (player_x - self.trophy_x) ** 2 + (player_y - self.trophy_y) ** 2
        )
        hint: dict = {"trophy_distance_approx": "unknown"}

        # Graduated distance hints
        if dist < 5:
            hint["trophy_distance_approx"] = "very_close"
        elif dist < 15:
            hint["trophy_distance_approx"] = "close"
        elif dist < 30:
            hint["trophy_distance_approx"] = "moderate"
        else:
            hint["trophy_distance_approx"] = "far"

        # Warmer/colder feedback
        if self.tcfg.warm_cold_enabled:
            if dist < self._prev_distance - 0.5:
                hint["trophy_temperature"] = "warmer"
            elif dist > self._prev_distance + 0.5:
                hint["trophy_temperature"] = "colder"
            else:
                hint["trophy_temperature"] = "same"

        # Directional hint every N steps
        if step % self.tcfg.hint_interval == 0:
            dx = self.trophy_x - player_x
            dy = self.trophy_y - player_y
            if abs(dx) > abs(dy):
                hint["trophy_direction"] = "E" if dx > 0 else "W"
            else:
                hint["trophy_direction"] = "S" if dy > 0 else "N"

        self._prev_distance = dist
        return hint
