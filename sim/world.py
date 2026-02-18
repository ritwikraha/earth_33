"""World generation: procedural terrain, elevation, water, vegetation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from config_io.config import Config
from config_io.schema import Terrain, Soil, VegetationType


# ── Terrain generation helpers ─────────────────────────────────────────────

# Mapping terrain -> typical soil
_TERRAIN_SOIL: dict[Terrain, list[Soil]] = {
    Terrain.PLAINS: [Soil.LOAMY, Soil.CLAY],
    Terrain.FOREST: [Soil.LOAMY, Soil.PEAT],
    Terrain.DESERT: [Soil.SANDY, Soil.ROCKY],
    Terrain.TUNDRA: [Soil.ROCKY, Soil.PEAT],
    Terrain.MOUNTAIN: [Soil.ROCKY],
    Terrain.WATER: [Soil.CLAY],
    Terrain.SWAMP: [Soil.PEAT, Soil.CLAY],
}

_TERRAIN_VEG: dict[Terrain, list[VegetationType]] = {
    Terrain.PLAINS: [VegetationType.GRASS, VegetationType.SHRUB],
    Terrain.FOREST: [VegetationType.TREES, VegetationType.SHRUB],
    Terrain.DESERT: [VegetationType.NONE, VegetationType.SHRUB],
    Terrain.TUNDRA: [VegetationType.GRASS, VegetationType.NONE],
    Terrain.MOUNTAIN: [VegetationType.NONE, VegetationType.GRASS],
    Terrain.WATER: [VegetationType.NONE],
    Terrain.SWAMP: [VegetationType.GRASS, VegetationType.SHRUB],
}

_TERRAIN_WILDLIFE: dict[Terrain, float] = {
    Terrain.PLAINS: 0.10,
    Terrain.FOREST: 0.20,
    Terrain.DESERT: 0.05,
    Terrain.TUNDRA: 0.08,
    Terrain.MOUNTAIN: 0.12,
    Terrain.WATER: 0.03,
    Terrain.SWAMP: 0.15,
}

_TERRAIN_SHELTER: dict[Terrain, float] = {
    Terrain.PLAINS: 0.05,
    Terrain.FOREST: 0.40,
    Terrain.DESERT: 0.02,
    Terrain.TUNDRA: 0.05,
    Terrain.MOUNTAIN: 0.25,
    Terrain.WATER: 0.0,
    Terrain.SWAMP: 0.10,
}

_TERRAIN_MOVE_COST: dict[Terrain, float] = {
    Terrain.PLAINS: 1.0,
    Terrain.FOREST: 1.3,
    Terrain.DESERT: 1.4,
    Terrain.TUNDRA: 1.2,
    Terrain.MOUNTAIN: 2.0,
    Terrain.WATER: 3.0,
    Terrain.SWAMP: 1.8,
}


def _perlin_like_noise(w: int, h: int, rng: Generator, octaves: int = 4, scale: float = 0.05) -> np.ndarray:
    """Simple fractal noise via layered smoothed random fields."""
    result = np.zeros((h, w), dtype=np.float64)
    for octave in range(octaves):
        freq = 2 ** octave
        amp = 1.0 / freq
        # Generate low-res noise and upscale via interpolation
        lw = max(2, int(w * scale * freq))
        lh = max(2, int(h * scale * freq))
        noise = rng.standard_normal((lh, lw))
        # Bilinear upscale using numpy
        rows = np.linspace(0, lh - 1, h)
        cols = np.linspace(0, lw - 1, w)
        r0 = np.floor(rows).astype(int)
        c0 = np.floor(cols).astype(int)
        r1 = np.clip(r0 + 1, 0, lh - 1)
        c1 = np.clip(c0 + 1, 0, lw - 1)
        dr = rows - r0
        dc = cols - c0
        upscaled = (
            noise[np.ix_(r0, c0)] * (1 - dr[:, None]) * (1 - dc[None, :])
            + noise[np.ix_(r1, c0)] * dr[:, None] * (1 - dc[None, :])
            + noise[np.ix_(r0, c1)] * (1 - dr[:, None]) * dc[None, :]
            + noise[np.ix_(r1, c1)] * dr[:, None] * dc[None, :]
        )
        result += amp * upscaled
    # Normalize to [0, 1]
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-8:
        result = (result - lo) / (hi - lo)
    return result


class WorldState:
    """Holds all grid-level state arrays for the world."""

    def __init__(self, config: Config, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.w = config.map.width
        self.h = config.map.height

        # Generate arrays
        self.elevation = self._gen_elevation()
        self.terrain_id = self._gen_terrain()
        self.soil_id = np.zeros((self.h, self.w), dtype=np.int8)
        self.water_mask = np.zeros((self.h, self.w), dtype=bool)
        self.water_availability = np.zeros((self.h, self.w), dtype=np.float64)
        self.vegetation_biomass = np.zeros((self.h, self.w), dtype=np.float64)
        self.vegetation_type_id = np.zeros((self.h, self.w), dtype=np.int8)
        self.wildlife_risk = np.zeros((self.h, self.w), dtype=np.float64)
        self.shelter_quality = np.zeros((self.h, self.w), dtype=np.float64)
        self.movement_cost = np.ones((self.h, self.w), dtype=np.float64)

        # Environmental fields (updated each step)
        self.air_temperature_c = np.zeros((self.h, self.w), dtype=np.float64)
        self.humidity = np.zeros((self.h, self.w), dtype=np.float64)
        self.precip_prob = np.zeros((self.h, self.w), dtype=np.float64)

        # Distance to water (precomputed)
        self.dist_to_water: np.ndarray | None = None

        self._place_water_bodies()
        self._init_cell_properties()
        self._compute_dist_to_water()

    # ── Elevation ──────────────────────────────────────────────────────

    def _gen_elevation(self) -> np.ndarray:
        elev = _perlin_like_noise(
            self.w, self.h, self.rng,
            octaves=self.config.map.elevation_octaves,
            scale=self.config.map.elevation_scale,
        )
        # Scale to 0-3000 m
        return elev * 3000.0

    # ── Terrain ────────────────────────────────────────────────────────

    def _gen_terrain(self) -> np.ndarray:
        """Assign terrain based on elevation bands + noise + weights."""
        terrain = np.zeros((self.h, self.w), dtype=np.int8)
        terrains = list(Terrain)
        weights_cfg = self.config.map.terrain_weights
        # Normalise weights
        names = [t.value for t in terrains]
        raw_w = np.array([weights_cfg.get(n, 0.0) for n in names], dtype=np.float64)
        raw_w /= raw_w.sum()

        noise = _perlin_like_noise(self.w, self.h, self.rng, octaves=3, scale=0.08)

        # Assign via cumulative thresholds on noise
        cum = np.cumsum(raw_w)
        for y in range(self.h):
            for x in range(self.w):
                val = noise[y, x]
                idx = np.searchsorted(cum, val)
                idx = min(idx, len(terrains) - 1)
                terrain[y, x] = idx
        return terrain

    # ── Water bodies ───────────────────────────────────────────────────

    def _place_water_bodies(self) -> None:
        """Place lakes and rivers by setting terrain to WATER."""
        water_idx = list(Terrain).index(Terrain.WATER)

        # Lakes: circular blobs at low-elevation points
        for _ in range(self.config.map.num_lakes):
            # Find a low-elevation point
            cy = self.rng.integers(5, self.h - 5)
            cx = self.rng.integers(5, self.w - 5)
            radius = self.rng.integers(2, 6)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        if dx * dx + dy * dy <= radius * radius:
                            self.terrain_id[ny, nx] = water_idx
                            self.water_mask[ny, nx] = True

        # Rivers: random walks from high to low elevation
        for _ in range(self.config.map.num_rivers):
            y = self.rng.integers(0, self.h)
            x = self.rng.integers(0, self.w)
            for _ in range(max(self.w, self.h)):
                if not (0 <= y < self.h and 0 <= x < self.w):
                    break
                self.terrain_id[y, x] = water_idx
                self.water_mask[y, x] = True
                # Move toward lower elevation (with randomness)
                best_e = self.elevation[y, x]
                best_d = (0, 1)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        if self.elevation[ny, nx] < best_e or self.rng.random() < 0.3:
                            best_e = self.elevation[ny, nx]
                            best_d = (dy, dx)
                y += best_d[0]
                x += best_d[1]

    # ── Initialize cell properties from terrain ────────────────────────

    def _init_cell_properties(self) -> None:
        terrains = list(Terrain)
        for y in range(self.h):
            for x in range(self.w):
                t = terrains[self.terrain_id[y, x]]

                # Soil
                soils = _TERRAIN_SOIL[t]
                self.soil_id[y, x] = list(Soil).index(
                    soils[self.rng.integers(0, len(soils))]
                )

                # Vegetation
                veg_options = _TERRAIN_VEG[t]
                veg = veg_options[self.rng.integers(0, len(veg_options))]
                self.vegetation_type_id[y, x] = list(VegetationType).index(veg)
                if veg == VegetationType.NONE:
                    self.vegetation_biomass[y, x] = 0.0
                elif veg == VegetationType.GRASS:
                    self.vegetation_biomass[y, x] = self.rng.uniform(0.1, 0.4)
                elif veg == VegetationType.SHRUB:
                    self.vegetation_biomass[y, x] = self.rng.uniform(0.2, 0.6)
                else:  # TREES
                    self.vegetation_biomass[y, x] = self.rng.uniform(0.5, 0.9)

                # Wildlife
                self.wildlife_risk[y, x] = _TERRAIN_WILDLIFE[t] * self.rng.uniform(0.5, 1.5)
                self.wildlife_risk[y, x] = np.clip(self.wildlife_risk[y, x], 0.0, 1.0)

                # Shelter
                self.shelter_quality[y, x] = _TERRAIN_SHELTER[t] * self.rng.uniform(0.5, 1.5)
                self.shelter_quality[y, x] = np.clip(self.shelter_quality[y, x], 0.0, 1.0)

                # Movement cost
                self.movement_cost[y, x] = _TERRAIN_MOVE_COST[t]

                # Water availability for water cells
                if self.water_mask[y, x]:
                    self.water_availability[y, x] = 1.0

    # ── Distance to water ──────────────────────────────────────────────

    def _compute_dist_to_water(self) -> None:
        """BFS distance from every cell to nearest water cell."""
        from collections import deque
        dist = np.full((self.h, self.w), fill_value=9999.0, dtype=np.float64)
        queue: deque[tuple[int, int]] = deque()
        for y in range(self.h):
            for x in range(self.w):
                if self.water_mask[y, x]:
                    dist[y, x] = 0.0
                    queue.append((y, x))
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    nd = dist[cy, cx] + 1.0
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        queue.append((ny, nx))
        self.dist_to_water = dist

    # ── Helpers ────────────────────────────────────────────────────────

    def terrain_at(self, x: int, y: int) -> Terrain:
        return list(Terrain)[self.terrain_id[y, x]]

    def vegetation_type_at(self, x: int, y: int) -> VegetationType:
        return list(VegetationType)[self.vegetation_type_id[y, x]]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def find_spawn_point(self) -> tuple[int, int]:
        """Find a reasonable spawn point: non-water, moderate conditions."""
        terrains = list(Terrain)
        for _ in range(1000):
            x = int(self.rng.integers(5, self.w - 5))
            y = int(self.rng.integers(5, self.h - 5))
            t = terrains[self.terrain_id[y, x]]
            if t not in (Terrain.WATER, Terrain.MOUNTAIN):
                return (x, y)
        # Fallback: center
        return (self.w // 2, self.h // 2)
