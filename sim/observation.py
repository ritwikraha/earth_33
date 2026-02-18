"""Build partial observations for the agent."""

from __future__ import annotations

import math

import numpy as np

from config_io.config import Config
from config_io.schema import Terrain, ActionType
from sim.physiology import OrganismState
from sim.world import WorldState


def build_observation(
    world: WorldState,
    organism: OrganismState,
    step: int,
    config: Config,
    recent_events: list[str],
    visible_hunters: list[dict] | None = None,
    hunter_sighting_history: list[list[dict]] | None = None,
    trophy_hint: dict | None = None,
) -> dict:
    """Build the observation dict that gets sent to the agent."""
    from sim.dynamics import get_time_info

    time_info = get_time_info(step, config)

    # Use fog of war radius if enabled, otherwise normal observation radius
    R = (
        config.fog_of_war.visibility_radius
        if config.fog_of_war.enabled
        else config.sim.observation_radius
    )
    x, y = organism.x, organism.y

    # Local cell
    terrain = world.terrain_at(x, y)
    local = {
        "terrain": terrain.value,
        "air_temp_c": round(world.air_temperature_c[y, x], 1),
        "humidity": round(world.humidity[y, x], 2),
        "water_availability": round(world.water_availability[y, x], 2),
        "vegetation_biomass": round(world.vegetation_biomass[y, x], 2),
        "wildlife_risk": round(world.wildlife_risk[y, x], 2),
        "shelter_quality": round(world.shelter_quality[y, x], 2),
        "movement_cost": round(world.movement_cost[y, x], 1),
        "elevation_m": round(world.elevation[y, x], 0),
    }

    # Neighborhood analysis
    nearby = _analyze_neighborhood(world, x, y, R)

    # Action mask: valid actions
    action_mask = _get_action_mask(world, organism, x, y)

    obs = {
        "timestep": step,
        "time": {
            "day_of_year": time_info["day_of_year"],
            "hour": time_info["hour"],
        },
        "agent": {
            "pos": {"x": x, "y": y},
            "hydration": round(organism.hydration, 1),
            "energy": round(organism.energy, 1),
            "core_temp_c": round(organism.core_temp_c, 1),
            "fatigue": round(organism.fatigue, 1),
            "injury": round(organism.injury, 1),
            "infection": round(organism.infection, 1),
            "has_shelter": organism.has_shelter,
        },
        "local": local,
        "nearby": nearby,
        "recent_memory": recent_events[-5:] if recent_events else [],
        "action_mask": action_mask,
    }

    # Hunter awareness
    if visible_hunters is not None:
        obs["visible_hunters"] = visible_hunters
        # Compile sighting history for LLM radius estimation
        if hunter_sighting_history:
            obs["hunter_sighting_history"] = _compile_sighting_summary(
                hunter_sighting_history
            )

    # Trophy hints
    if trophy_hint:
        obs["trophy"] = trophy_hint

    return obs


def _compile_sighting_summary(history: list[list[dict]]) -> list[dict]:
    """Compile per-hunter sighting data for LLM radius estimation.

    Groups sightings by hunter ID and returns a summary showing
    when each hunter was seen, at what distance, and whether it
    was chasing -- allowing the LLM to estimate detection radii.
    """
    hunter_data: dict[int, list[dict]] = {}
    for step_idx, sightings in enumerate(history):
        for s in sightings:
            hid = s["id"]
            if hid not in hunter_data:
                hunter_data[hid] = []
            hunter_data[hid].append({
                "step_offset": step_idx - len(history),  # negative = N steps ago
                "pos": s["pos"],
                "distance": s["distance"],
                "is_chasing": s["is_chasing"],
            })
    return [
        {"hunter_id": hid, "sightings": sights}
        for hid, sights in hunter_data.items()
    ]


def _analyze_neighborhood(world: WorldState, cx: int, cy: int, radius: int) -> dict:
    """Summarize the neighborhood within radius."""
    temps = []
    veg_vals = []
    max_wildlife = 0.0
    nearest_water_dist = 9999
    nearest_water_dir = ""
    best_shelter_dist = 9999
    best_shelter_dir = ""
    best_shelter_q = 0.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if not world.in_bounds(nx, ny):
                continue
            if dx == 0 and dy == 0:
                continue

            temps.append(world.air_temperature_c[ny, nx])
            veg_vals.append(world.vegetation_biomass[ny, nx])

            wr = world.wildlife_risk[ny, nx]
            if wr > max_wildlife:
                max_wildlife = wr

            dist = abs(dx) + abs(dy)

            # Water
            if world.water_mask[ny, nx] and dist < nearest_water_dist:
                nearest_water_dist = dist
                nearest_water_dir = _direction_name(dx, dy)

            # Shelter
            sq = world.shelter_quality[ny, nx]
            if sq > best_shelter_q or (sq == best_shelter_q and dist < best_shelter_dist):
                best_shelter_q = sq
                best_shelter_dist = dist
                best_shelter_dir = _direction_name(dx, dy)

    # If no water in radius, use dist_to_water map
    if nearest_water_dist == 9999 and world.dist_to_water is not None:
        d = world.dist_to_water[cy, cx]
        nearest_water_dist = int(d)
        # Estimate direction by checking gradient
        best_d = 9999.0
        best_dir = ""
        for dy, dx, dname in [(-1, 0, "N"), (1, 0, "S"), (0, 1, "E"), (0, -1, "W")]:
            nx2, ny2 = cx + dx, cy + dy
            if world.in_bounds(nx2, ny2):
                dd = world.dist_to_water[ny2, nx2]
                if dd < best_d:
                    best_d = dd
                    best_dir = dname
        nearest_water_dir = best_dir

    return {
        "nearest_water": {
            "distance": nearest_water_dist,
            "direction": nearest_water_dir,
        },
        "best_shelter": {
            "distance": best_shelter_dist if best_shelter_dist < 9999 else -1,
            "direction": best_shelter_dir,
            "shelter_quality": round(best_shelter_q, 2),
        },
        "avg_temp_c": round(sum(temps) / len(temps), 1) if temps else 0.0,
        "avg_vegetation": round(sum(veg_vals) / len(veg_vals), 2) if veg_vals else 0.0,
        "max_wildlife_risk": round(max_wildlife, 2),
    }


def _direction_name(dx: int, dy: int) -> str:
    """Cardinal direction from delta."""
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def _get_action_mask(world: WorldState, organism: OrganismState, x: int, y: int) -> list[str]:
    """Return list of valid action names."""
    actions = [ActionType.REST, ActionType.HIDE, ActionType.SIGNAL]

    # Movement
    for act, (dx, dy) in [
        (ActionType.MOVE_N, (0, -1)),
        (ActionType.MOVE_S, (0, 1)),
        (ActionType.MOVE_E, (1, 0)),
        (ActionType.MOVE_W, (-1, 0)),
    ]:
        nx, ny = x + dx, y + dy
        if world.in_bounds(nx, ny):
            actions.append(act)

    # Drink: need some water
    if world.water_availability[y, x] > 0.1:
        actions.append(ActionType.DRINK)

    # Forage: need vegetation
    if world.vegetation_biomass[y, x] > 0.1:
        actions.append(ActionType.FORAGE)

    # Build shelter: not on water
    terrain = world.terrain_at(x, y)
    if terrain != Terrain.WATER:
        actions.append(ActionType.BUILD_SHELTER)

    return [a.value for a in actions]
