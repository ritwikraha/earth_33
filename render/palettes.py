"""Color palettes for terrain and overlay rendering."""

from __future__ import annotations

from config_io.schema import Terrain

# RGB tuples for terrain types
TERRAIN_COLORS: dict[Terrain, tuple[int, int, int]] = {
    Terrain.PLAINS:   (144, 190, 109),
    Terrain.FOREST:   (56, 118, 29),
    Terrain.DESERT:   (222, 199, 150),
    Terrain.TUNDRA:   (200, 220, 230),
    Terrain.MOUNTAIN: (139, 137, 137),
    Terrain.WATER:    (65, 105, 225),
    Terrain.SWAMP:    (107, 142, 35),
}

# For terrain by index (matches enum order)
TERRAIN_COLORS_BY_IDX: list[tuple[int, int, int]] = [
    TERRAIN_COLORS[t] for t in Terrain
]


def temperature_color(temp_c: float) -> tuple[int, int, int]:
    """Map temperature to blue-red gradient. -20..50 range."""
    t = max(-20.0, min(50.0, temp_c))
    frac = (t + 20.0) / 70.0  # 0..1
    r = int(frac * 255)
    b = int((1.0 - frac) * 255)
    g = int((1.0 - abs(frac - 0.5) * 2) * 100)
    return (r, g, b)


def water_color(water_avail: float) -> tuple[int, int, int]:
    """Map water availability [0,1] to brown-blue."""
    f = max(0.0, min(1.0, water_avail))
    r = int((1.0 - f) * 180 + f * 30)
    g = int((1.0 - f) * 140 + f * 80)
    b = int((1.0 - f) * 80 + f * 220)
    return (r, g, b)


def vegetation_color(biomass: float) -> tuple[int, int, int]:
    """Map vegetation [0,1] to brown-green."""
    f = max(0.0, min(1.0, biomass))
    r = int((1.0 - f) * 160 + f * 20)
    g = int((1.0 - f) * 120 + f * 180)
    b = int((1.0 - f) * 60 + f * 20)
    return (r, g, b)


def wildlife_color(risk: float) -> tuple[int, int, int]:
    """Map wildlife risk [0,1] to green-red."""
    f = max(0.0, min(1.0, risk))
    r = int(f * 255)
    g = int((1.0 - f) * 200)
    return (r, g, 0)


def shelter_color(quality: float) -> tuple[int, int, int]:
    """Map shelter quality [0,1] to gray-cyan."""
    f = max(0.0, min(1.0, quality))
    r = int((1.0 - f) * 100)
    g = int((1.0 - f) * 100 + f * 200)
    b = int((1.0 - f) * 100 + f * 220)
    return (r, g, b)


def elevation_color(elev_m: float, max_elev: float = 3000.0) -> tuple[int, int, int]:
    """Map elevation to grayscale."""
    f = max(0.0, min(1.0, elev_m / max_elev))
    v = int(40 + f * 215)
    return (v, v, v)


# HUD colors
HUD_BG = (30, 30, 40)
HUD_TEXT = (220, 220, 220)
HUD_BAR_BG = (60, 60, 70)
BAR_HYDRATION = (50, 130, 255)
BAR_ENERGY = (255, 200, 50)
BAR_FATIGUE = (200, 100, 50)
BAR_INJURY = (255, 60, 60)
BAR_INFECTION = (180, 50, 180)
AGENT_COLOR = (255, 255, 0)
TRAIL_COLOR = (255, 200, 0, 128)

# Hunter colors
HUNTER_COLOR = (200, 60, 60)
HUNTER_CHASING_COLOR = (255, 0, 0)
HUNTER_RADIUS_ALPHA = 30

# Trophy colors
TROPHY_COLOR = (255, 215, 0)
TROPHY_GLOW_ALPHA = 50

# Fog of war
# Unexplored: 40% visible (60% dark overlay)
FOG_UNEXPLORED_ALPHA = 153
# Explored: fully visible (no overlay)
FOG_EXPLORED_BASE_ALPHA = 0

# Grid
GRID_ALPHA = 25

# Agent pulse colors
AGENT_PULSE_MIN = (255, 220, 50)
AGENT_PULSE_MAX = (255, 255, 100)
