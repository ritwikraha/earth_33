"""Hunter NPC entities: patrol, detect, chase, kill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

from config_io.config import Config
from config_io.schema import Terrain
from sim.world import WorldState


@dataclass
class Hunter:
    """A single hunter NPC."""
    id: int
    x: int
    y: int
    detection_radius: int
    chase_speed: int
    patrol_speed: int
    is_chasing: bool = False
    patrol_direction: tuple[int, int] = (1, 0)
    steps_since_direction_change: int = 0

    def distance_to(self, tx: int, ty: int) -> float:
        """Euclidean distance to target."""
        return math.sqrt((self.x - tx) ** 2 + (self.y - ty) ** 2)

    def manhattan_distance_to(self, tx: int, ty: int) -> int:
        """Manhattan distance to target."""
        return abs(self.x - tx) + abs(self.y - ty)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pos": {"x": self.x, "y": self.y},
            "detection_radius": self.detection_radius,
            "is_chasing": self.is_chasing,
        }


class HunterManager:
    """Manages all hunter NPCs for the simulation."""

    def __init__(
        self,
        config: Config,
        world: WorldState,
        rng: Generator,
        player_spawn: tuple[int, int],
    ):
        self.config = config
        self.hcfg = config.hunters
        self.world = world
        self.rng = rng
        self.hunters: list[Hunter] = []

        if not self.hcfg.enabled:
            return

        self._spawn_hunters(player_spawn)

    def _spawn_hunters(self, player_spawn: tuple[int, int]) -> None:
        """Place hunters at least spawn_margin distance from the player."""
        px, py = player_spawn
        placed = 0
        attempts = 0
        while placed < self.hcfg.count and attempts < 1000:
            x = int(self.rng.integers(2, self.world.w - 2))
            y = int(self.rng.integers(2, self.world.h - 2))
            dist = abs(x - px) + abs(y - py)
            terrain = self.world.terrain_at(x, y)
            # Don't spawn on water or too close to player
            if dist >= self.hcfg.spawn_margin and terrain != Terrain.WATER:
                radius = int(self.rng.integers(
                    self.hcfg.min_detection_radius,
                    self.hcfg.max_detection_radius + 1,
                ))
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                direction = dirs[int(self.rng.integers(0, 4))]
                hunter = Hunter(
                    id=placed,
                    x=x,
                    y=y,
                    detection_radius=radius,
                    chase_speed=self.hcfg.chase_speed,
                    patrol_speed=self.hcfg.patrol_speed,
                    patrol_direction=direction,
                )
                self.hunters.append(hunter)
                placed += 1
            attempts += 1

    def update(self, player_x: int, player_y: int) -> Optional[int]:
        """Move all hunters, check for kill.

        Returns the id of the hunter that killed the player, or None.
        """
        killer_id = None
        for h in self.hunters:
            dist = h.distance_to(player_x, player_y)

            if dist <= h.detection_radius:
                # Chase mode
                h.is_chasing = True
                self._move_toward(h, player_x, player_y, h.chase_speed)
            else:
                h.is_chasing = False
                self._patrol(h)

            # Kill check: hunter occupies same cell or adjacent
            if h.manhattan_distance_to(player_x, player_y) <= 1:
                killer_id = h.id

        return killer_id

    def _move_toward(self, h: Hunter, tx: int, ty: int, speed: int) -> None:
        """Move hunter toward target by up to `speed` cells."""
        for _ in range(speed):
            dx = 0 if tx == h.x else (1 if tx > h.x else -1)
            dy = 0 if ty == h.y else (1 if ty > h.y else -1)
            # Prefer the axis with greater distance
            if abs(tx - h.x) >= abs(ty - h.y):
                nx, ny = h.x + dx, h.y
            else:
                nx, ny = h.x, h.y + dy
            if self.world.in_bounds(nx, ny):
                terrain = self.world.terrain_at(nx, ny)
                if terrain != Terrain.WATER:
                    h.x, h.y = nx, ny
                else:
                    # Try perpendicular to avoid water
                    if abs(tx - h.x) >= abs(ty - h.y):
                        alt_nx, alt_ny = h.x, h.y + dy
                    else:
                        alt_nx, alt_ny = h.x + dx, h.y
                    if (self.world.in_bounds(alt_nx, alt_ny) and
                            self.world.terrain_at(alt_nx, alt_ny) != Terrain.WATER):
                        h.x, h.y = alt_nx, alt_ny

    def _patrol(self, h: Hunter) -> None:
        """Random walk patrol movement."""
        h.steps_since_direction_change += 1
        # Change direction randomly every 5-15 steps
        if h.steps_since_direction_change > 5 and self.rng.random() < 0.2:
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            h.patrol_direction = dirs[int(self.rng.integers(0, 4))]
            h.steps_since_direction_change = 0

        dx, dy = h.patrol_direction
        nx, ny = h.x + dx, h.y + dy
        if self.world.in_bounds(nx, ny):
            terrain = self.world.terrain_at(nx, ny)
            if terrain != Terrain.WATER:
                h.x, h.y = nx, ny
            else:
                # Bounce: reverse direction
                h.patrol_direction = (-dx, -dy)
        else:
            # Bounce: reverse direction
            h.patrol_direction = (-dx, -dy)

    def get_visible_hunters(
        self,
        player_x: int,
        player_y: int,
        visibility_radius: int,
    ) -> list[dict]:
        """Return hunter data for those within the player's visibility.

        Note: detection_radius is deliberately NOT revealed to the agent.
        The LLM must estimate it from sighting history.
        """
        visible = []
        for h in self.hunters:
            dist = h.manhattan_distance_to(player_x, player_y)
            if dist <= visibility_radius:
                visible.append({
                    "id": h.id,
                    "pos": {"x": h.x, "y": h.y},
                    "distance": dist,
                    "is_chasing": h.is_chasing,
                })
        return visible

    def get_all_hunter_positions(self) -> list[dict]:
        """For rendering (shows all hunters regardless of visibility)."""
        return [h.to_dict() for h in self.hunters]
