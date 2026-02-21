"""Configuration loading and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


# ── Sub-configs ────────────────────────────────────────────────────────────

class MapConfig(BaseModel):
    width: int = 80
    height: int = 60
    terrain_weights: dict[str, float] = Field(default_factory=lambda: {
        "PLAINS": 0.30, "FOREST": 0.25, "DESERT": 0.10,
        "TUNDRA": 0.05, "MOUNTAIN": 0.10, "WATER": 0.10, "SWAMP": 0.10,
    })
    num_lakes: int = 3
    num_rivers: int = 2
    elevation_octaves: int = 4
    elevation_scale: float = 0.05


class SimConfig(BaseModel):
    dt_hours: float = 1.0
    max_steps: int = 500
    observation_radius: int = 3
    start_hour: int = 6
    start_day: int = 80  # late March


class ClimateConfig(BaseModel):
    lapse_rate: float = 6.5
    diurnal_amp: dict[str, float] = Field(default_factory=lambda: {
        "DESERT": 10.0, "PLAINS": 6.0, "FOREST": 4.0,
        "TUNDRA": 5.0, "MOUNTAIN": 7.0, "WATER": 3.0, "SWAMP": 4.0,
    })
    seasonal_amp: dict[str, float] = Field(default_factory=lambda: {
        "DESERT": 8.0, "PLAINS": 10.0, "FOREST": 10.0,
        "TUNDRA": 15.0, "MOUNTAIN": 12.0, "WATER": 6.0, "SWAMP": 8.0,
    })
    terrain_base_temp: dict[str, float] = Field(default_factory=lambda: {
        "DESERT": 30.0, "PLAINS": 18.0, "FOREST": 15.0,
        "TUNDRA": -5.0, "MOUNTAIN": 8.0, "WATER": 16.0, "SWAMP": 20.0,
    })
    temp_noise_sigma: float = 1.0


class PhysiologyConfig(BaseModel):
    initial_hydration: float = 80.0
    initial_energy: float = 80.0
    initial_core_temp: float = 37.0
    initial_fatigue: float = 10.0
    initial_injury: float = 0.0
    initial_infection: float = 0.0
    # Drain rates
    hydration_base_drain: float = 0.8
    energy_base_drain: float = 0.6
    fatigue_base_gain: float = 0.7
    # Recovery
    rest_fatigue_recovery: float = 4.0
    injury_recovery_rate: float = 0.2
    # Thermal
    thermal_drift_k: float = 0.008
    # Death thresholds
    death_hydration: float = 0.0
    death_energy: float = 0.0
    death_core_temp_low: float = 30.0
    death_core_temp_high: float = 42.0
    death_injury: float = 100.0
    death_infection: float = 100.0


class AgentConfig(BaseModel):
    type: str = "heuristic"  # llm / random / heuristic
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    llm_max_retries: int = 2
    llm_temperature: float = 0.7


class RenderConfig(BaseModel):
    cell_size: int = 10
    fps: int = 30
    trail_length: int = 50
    hud_width: int = 280


class HunterConfig(BaseModel):
    """Configuration for hunter NPC entities."""
    enabled: bool = False
    count: int = 5
    min_detection_radius: int = 3
    max_detection_radius: int = 8
    chase_speed: int = 2
    patrol_speed: int = 1
    spawn_margin: int = 15


class TrophyConfig(BaseModel):
    """Configuration for the trophy objective."""
    enabled: bool = False
    min_distance_from_spawn: int = 25
    hint_interval: int = 10
    warm_cold_enabled: bool = True


class DifficultyConfig(BaseModel):
    """Global difficulty multipliers."""
    drain_multiplier: float = 1.0
    hazard_multiplier: float = 1.0
    wildlife_multiplier: float = 1.0
    temperature_extremity: float = 1.0


class FogOfWarConfig(BaseModel):
    """Fog of war visibility settings."""
    enabled: bool = False
    visibility_radius: int = 5
    explored_dim_factor: float = 0.4


class SwarmConfig(BaseModel):
    """Configuration for swarm/bio-inspired optimization agents."""
    population_size: int = 10
    search_radius: int = 30
    inner_iterations: int = 3


# ── Top-level config ───────────────────────────────────────────────────

class Config(BaseModel):
    map: MapConfig = Field(default_factory=MapConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    physiology: PhysiologyConfig = Field(default_factory=PhysiologyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    hunters: HunterConfig = Field(default_factory=HunterConfig)
    trophy: TrophyConfig = Field(default_factory=TrophyConfig)
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    fog_of_war: FogOfWarConfig = Field(default_factory=FogOfWarConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML file, applying optional overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
    if overrides:
        _deep_merge(data, overrides)
    return Config(**data)


def _deep_merge(base: dict, overlay: dict) -> None:
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
