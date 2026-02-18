"""Organism physiology: state, resource drain, recovery, death checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from config_io.config import PhysiologyConfig
from config_io.schema import ActionType, CauseOfDeath
from config_io.utils import clamp


# Activity multipliers per action
_ACTIVITY_MULT: dict[ActionType, float] = {
    ActionType.MOVE_N: 1.5,
    ActionType.MOVE_S: 1.5,
    ActionType.MOVE_E: 1.5,
    ActionType.MOVE_W: 1.5,
    ActionType.REST: 0.7,
    ActionType.DRINK: 0.9,
    ActionType.FORAGE: 1.3,
    ActionType.BUILD_SHELTER: 1.4,
    ActionType.HIDE: 0.8,
    ActionType.SIGNAL: 0.8,
}


@dataclass
class OrganismState:
    x: int = 0
    y: int = 0
    hydration: float = 80.0
    energy: float = 80.0
    core_temp_c: float = 37.0
    fatigue: float = 10.0
    injury: float = 0.0
    infection: float = 0.0
    alive: bool = True
    cause_of_death: CauseOfDeath = CauseOfDeath.ALIVE
    age_steps: int = 0
    has_shelter: bool = False
    shelter_durability: int = 0
    inventory: dict = field(default_factory=lambda: {"water": 0.0, "food": 0.0})

    def to_dict(self) -> dict:
        return {
            "pos": {"x": self.x, "y": self.y},
            "hydration": round(self.hydration, 1),
            "energy": round(self.energy, 1),
            "core_temp_c": round(self.core_temp_c, 1),
            "fatigue": round(self.fatigue, 1),
            "injury": round(self.injury, 1),
            "infection": round(self.infection, 1),
            "alive": self.alive,
            "cause_of_death": self.cause_of_death.value,
            "age_steps": self.age_steps,
            "has_shelter": self.has_shelter,
        }

    @staticmethod
    def from_config(cfg: PhysiologyConfig, x: int, y: int) -> "OrganismState":
        return OrganismState(
            x=x, y=y,
            hydration=cfg.initial_hydration,
            energy=cfg.initial_energy,
            core_temp_c=cfg.initial_core_temp,
            fatigue=cfg.initial_fatigue,
            injury=cfg.initial_injury,
            infection=cfg.initial_infection,
        )


def apply_physiology(
    state: OrganismState,
    action: ActionType,
    air_temp: float,
    shelter_active: bool,
    movement_cost: float,
    dt: float,
    cfg: PhysiologyConfig,
    difficulty_mult: float = 1.0,
) -> dict[str, float]:
    """Apply resource drain, recovery, thermal drift. Returns delta dict."""
    deltas: dict[str, float] = {}
    act_mult = _ACTIVITY_MULT.get(action, 1.0)

    # Movement cost scaling for move actions
    if action in (ActionType.MOVE_N, ActionType.MOVE_S, ActionType.MOVE_E, ActionType.MOVE_W):
        act_mult *= movement_cost

    # Heat multiplier
    heat_mult = max(0.0, (air_temp - 30.0) / 10.0) * 0.8

    # Hydration drain (scaled by difficulty)
    h_drain = (cfg.hydration_base_drain * act_mult + heat_mult) * dt * difficulty_mult
    state.hydration -= h_drain
    deltas["hydration"] = -h_drain

    # Energy drain (scaled by difficulty)
    e_drain = cfg.energy_base_drain * act_mult * dt * difficulty_mult
    state.energy -= e_drain
    deltas["energy"] = -e_drain

    # Fatigue
    if action == ActionType.REST:
        f_change = -cfg.rest_fatigue_recovery * dt
    else:
        f_change = cfg.fatigue_base_gain * act_mult * dt * difficulty_mult
    state.fatigue += f_change
    deltas["fatigue"] = f_change

    # Thermal drift
    shelter_k = cfg.thermal_drift_k * (0.3 if shelter_active else 1.0)
    temp_drift = (air_temp - state.core_temp_c) * shelter_k * dt
    state.core_temp_c += temp_drift
    deltas["core_temp"] = temp_drift

    # Injury recovery (small, when resting and low stress)
    if action == ActionType.REST and state.fatigue < 40:
        inj_heal = cfg.injury_recovery_rate * dt
        state.injury -= inj_heal
        deltas["injury"] = -inj_heal

    # Infection growth if injured and wet/cold
    if state.injury > 20 and (air_temp < 5 or air_temp > 38):
        inf_grow = 0.3 * (state.injury / 100.0) * dt
        state.infection += inf_grow
        deltas["infection"] = inf_grow

    # Clamp all values
    state.hydration = clamp(state.hydration, 0.0, 100.0)
    state.energy = clamp(state.energy, 0.0, 100.0)
    state.fatigue = clamp(state.fatigue, 0.0, 100.0)
    state.injury = clamp(state.injury, 0.0, 100.0)
    state.infection = clamp(state.infection, 0.0, 100.0)

    return deltas


def check_death(state: OrganismState, cfg: PhysiologyConfig) -> bool:
    """Check death conditions. Sets cause_of_death and alive=False if dead."""
    if state.hydration <= cfg.death_hydration:
        state.alive = False
        state.cause_of_death = CauseOfDeath.DEHYDRATION
    elif state.energy <= cfg.death_energy:
        state.alive = False
        state.cause_of_death = CauseOfDeath.STARVATION
    elif state.core_temp_c <= cfg.death_core_temp_low:
        state.alive = False
        state.cause_of_death = CauseOfDeath.HYPOTHERMIA
    elif state.core_temp_c >= cfg.death_core_temp_high:
        state.alive = False
        state.cause_of_death = CauseOfDeath.HYPERTHERMIA
    elif state.injury >= cfg.death_injury:
        state.alive = False
        state.cause_of_death = CauseOfDeath.TRAUMA
    elif state.infection >= cfg.death_infection:
        state.alive = False
        state.cause_of_death = CauseOfDeath.INFECTION
    return not state.alive
