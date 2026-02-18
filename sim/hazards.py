"""Wildlife encounters and environmental hazards."""

from __future__ import annotations

from numpy.random import Generator

from config_io.schema import ActionType, Terrain
from sim.physiology import OrganismState


def _hazard_time_multiplier(hour: int) -> float:
    """Wildlife more active at dawn/dusk."""
    if 5 <= hour <= 7 or 18 <= hour <= 20:
        return 1.5  # dawn/dusk
    elif 22 <= hour or hour <= 4:
        return 1.8  # night
    return 1.0


def _stealth_modifier(action: ActionType) -> float:
    """How visible the agent is."""
    if action == ActionType.HIDE:
        return 0.6
    elif action == ActionType.REST:
        return 0.3
    elif action in (ActionType.BUILD_SHELTER, ActionType.FORAGE):
        return -0.1  # noisy, increases risk
    return 0.0


def check_wildlife_encounter(
    state: OrganismState,
    wildlife_risk: float,
    terrain: Terrain,
    hour: int,
    action: ActionType,
    rng: Generator,
    hazard_multiplier: float = 1.0,
) -> dict:
    """Check for wildlife encounter. Returns event dict."""
    time_mult = _hazard_time_multiplier(hour)
    stealth = _stealth_modifier(action)
    # Scale down base probability so encounters aren't overwhelming
    p = wildlife_risk * time_mult * (1.0 - stealth) * 0.3 * hazard_multiplier
    p = max(0.0, min(1.0, p))

    encounter = rng.random() < p
    result = {"encounter": encounter, "injury_delta": 0.0, "energy_delta": 0.0, "infection_delta": 0.0}

    if encounter:
        severity = rng.uniform(3.0, 15.0)
        result["injury_delta"] = severity
        result["energy_delta"] = -rng.uniform(2.0, 6.0)
        result["infection_delta"] = rng.uniform(0.0, 3.0)

        state.injury += severity
        state.energy += result["energy_delta"]
        state.infection += result["infection_delta"]

        # Clamp
        state.injury = min(100.0, max(0.0, state.injury))
        state.energy = min(100.0, max(0.0, state.energy))
        state.infection = min(100.0, max(0.0, state.infection))

    return result


def check_environmental_hazards(
    state: OrganismState,
    air_temp: float,
    shelter_active: bool,
    terrain: Terrain,
    humidity: float,
    hazard_multiplier: float = 1.0,
) -> list[str]:
    """Check for heat stress, hypothermia risk, disease from swamp etc."""
    events: list[str] = []

    # Heat stress (threshold lowered by multiplier)
    heat_threshold = 38.0 / max(0.1, hazard_multiplier)
    if air_temp > heat_threshold and not shelter_active:
        extra_drain = (air_temp - heat_threshold) * 0.5 * hazard_multiplier
        state.hydration -= extra_drain
        state.hydration = max(0.0, state.hydration)
        events.append(f"Heat stress: extra hydration drain {extra_drain:.1f}")

    # Hypothermia acceleration
    if air_temp < 0.0 and not shelter_active and state.fatigue > 60:
        temp_drop = abs(air_temp) * 0.01
        state.core_temp_c -= temp_drop
        events.append(f"Hypothermia risk: core temp dropping extra {temp_drop:.2f}")

    # Swamp disease
    if terrain == Terrain.SWAMP and humidity > 0.7:
        disease_risk = 0.02 * humidity
        state.infection += disease_risk
        state.infection = min(100.0, state.infection)
        events.append(f"Swamp exposure: infection +{disease_risk:.2f}")

    return events
