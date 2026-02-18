"""Environment dynamics: temperature, humidity, precipitation, water, vegetation."""

from __future__ import annotations

import math

import numpy as np

from config_io.config import Config
from config_io.schema import Terrain
from sim.world import WorldState


def update_environment(world: WorldState, step: int, config: Config) -> None:
    """Update all environmental fields for the current timestep."""
    dt = config.sim.dt_hours
    hour = (config.sim.start_hour + step * dt) % 24.0
    day = (config.sim.start_day + (config.sim.start_hour + step * dt) / 24.0) % 365.0

    _update_temperature(world, hour, day, config)
    _update_humidity(world, config)
    _update_precipitation(world, day, config)
    _update_water_availability(world, config)
    _update_vegetation(world, dt, config)


def get_time_info(step: int, config: Config) -> dict:
    """Return current time info."""
    dt = config.sim.dt_hours
    total_hours = config.sim.start_hour + step * dt
    hour = total_hours % 24.0
    day = (config.sim.start_day + total_hours / 24.0) % 365.0
    return {
        "step": step,
        "day_of_year": int(day),
        "hour": int(hour),
        "total_hours": total_hours,
    }


# ── Temperature ────────────────────────────────────────────────────────────

def _update_temperature(world: WorldState, hour: float, day: float, config: Config) -> None:
    terrains = list(Terrain)
    cc = config.climate
    h, w = world.h, world.w

    for y in range(h):
        for x in range(w):
            t = terrains[world.terrain_id[y, x]]
            tname = t.value

            base = cc.terrain_base_temp.get(tname, 15.0)
            # Lapse rate adjustment for elevation
            base -= cc.lapse_rate * (world.elevation[y, x] / 1000.0)

            temp_ext = config.difficulty.temperature_extremity
            seasonal = cc.seasonal_amp.get(tname, 10.0) * temp_ext * math.sin(
                2.0 * math.pi * day / 365.0
            )
            diurnal = cc.diurnal_amp.get(tname, 6.0) * temp_ext * math.sin(
                2.0 * math.pi * hour / 24.0 - math.pi / 2.0
            )
            noise = world.rng.normal(0.0, cc.temp_noise_sigma)

            world.air_temperature_c[y, x] = base + seasonal + diurnal + noise


def update_temperature_vectorized(world: WorldState, hour: float, day: float, config: Config) -> None:
    """Vectorized temperature update for headless performance."""
    terrains = list(Terrain)
    cc = config.climate
    h, w = world.h, world.w

    # Build per-cell base temp, seasonal amp, diurnal amp arrays
    base_arr = np.zeros((h, w), dtype=np.float64)
    seas_arr = np.zeros((h, w), dtype=np.float64)
    diur_arr = np.zeros((h, w), dtype=np.float64)

    for idx, t in enumerate(terrains):
        mask = world.terrain_id == idx
        base_arr[mask] = cc.terrain_base_temp.get(t.value, 15.0)
        seas_arr[mask] = cc.seasonal_amp.get(t.value, 10.0)
        diur_arr[mask] = cc.diurnal_amp.get(t.value, 6.0)

    base_arr -= cc.lapse_rate * (world.elevation / 1000.0)
    temp_ext = config.difficulty.temperature_extremity
    seasonal = seas_arr * temp_ext * math.sin(2.0 * math.pi * day / 365.0)
    diurnal = diur_arr * temp_ext * math.sin(2.0 * math.pi * hour / 24.0 - math.pi / 2.0)
    noise = world.rng.normal(0.0, cc.temp_noise_sigma, size=(h, w))

    world.air_temperature_c[:] = base_arr + seasonal + diurnal + noise


# ── Humidity ───────────────────────────────────────────────────────────────

def _update_humidity(world: WorldState, config: Config) -> None:
    terrains = list(Terrain)
    h, w = world.h, world.w

    # Base humidity by terrain
    base_humidity: dict[Terrain, float] = {
        Terrain.WATER: 0.90, Terrain.SWAMP: 0.80, Terrain.FOREST: 0.60,
        Terrain.PLAINS: 0.40, Terrain.TUNDRA: 0.35, Terrain.MOUNTAIN: 0.30,
        Terrain.DESERT: 0.15,
    }

    for y in range(h):
        for x in range(w):
            t = terrains[world.terrain_id[y, x]]
            base = base_humidity.get(t, 0.40)
            # Boost near water
            if world.dist_to_water is not None:
                dist = world.dist_to_water[y, x]
                water_boost = 0.3 * math.exp(-dist / 5.0)
            else:
                water_boost = 0.0
            world.humidity[y, x] = min(1.0, base + water_boost)


# ── Precipitation ──────────────────────────────────────────────────────────

def _update_precipitation(world: WorldState, day: float, config: Config) -> None:
    h, w = world.h, world.w
    # Seasonal factor: more rain in "summer" (day ~180)
    season_factor = 0.5 + 0.5 * math.sin(2.0 * math.pi * day / 365.0 - math.pi / 4.0)

    for y in range(h):
        for x in range(w):
            world.precip_prob[y, x] = world.humidity[y, x] * season_factor * 0.3


# ── Water availability ─────────────────────────────────────────────────────

def _update_water_availability(world: WorldState, config: Config) -> None:
    h, w = world.h, world.w
    for y in range(h):
        for x in range(w):
            if world.water_mask[y, x]:
                world.water_availability[y, x] = 1.0
                continue

            dist = world.dist_to_water[y, x] if world.dist_to_water is not None else 9999.0
            proximity = 0.6 * math.exp(-dist / 4.0)

            # Rain contribution
            rain_contrib = world.precip_prob[y, x] * 0.4

            # Evaporation (higher with temp)
            temp = world.air_temperature_c[y, x]
            evap = max(0.0, (temp - 15.0) / 40.0) * 0.3

            world.water_availability[y, x] = max(0.0, min(1.0, proximity + rain_contrib - evap))


# ── Vegetation ─────────────────────────────────────────────────────────────

_SOIL_FACTOR = {0: 0.6, 1: 1.0, 2: 0.8, 3: 0.3, 4: 0.7}  # indexed by Soil enum order


def _update_vegetation(world: WorldState, dt: float, config: Config) -> None:
    h, w = world.h, world.w
    for y in range(h):
        for x in range(w):
            biomass = world.vegetation_biomass[y, x]
            if biomass <= 0.0:
                # Small chance of regrowth in good conditions
                if world.water_availability[y, x] > 0.3:
                    biomass = 0.01
                else:
                    continue

            water_f = world.water_availability[y, x]
            temp = world.air_temperature_c[y, x]
            # Temperature factor: peaks at 20°C, Gaussian falloff
            temp_f = math.exp(-((temp - 20.0) ** 2) / (2.0 * 15.0 ** 2))
            soil_f = _SOIL_FACTOR.get(int(world.soil_id[y, x]), 0.5)

            growth_rate = soil_f * water_f * temp_f * 0.01
            biomass += growth_rate * biomass * (1.0 - biomass) * dt

            # Drought loss
            if water_f < 0.1 and temp > 30.0:
                biomass -= 0.005 * dt

            world.vegetation_biomass[y, x] = max(0.0, min(1.0, biomass))
