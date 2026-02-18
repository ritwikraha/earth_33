"""Main environment: ties together world, dynamics, physiology, hazards."""

from __future__ import annotations

from config_io.config import Config
from config_io.schema import ActionType, Terrain, CauseOfDeath, EpisodeOutcome, ACTION_DIRECTION
from config_io.utils import clamp
from sim.world import WorldState
from sim.dynamics import update_environment, get_time_info
from sim.physiology import OrganismState, apply_physiology, check_death
from sim.hazards import check_wildlife_encounter, check_environmental_hazards
from sim.observation import build_observation
from sim.replay import ReplayLogger
from sim.hunters import HunterManager
from sim.trophy import TrophyManager


class Environment:
    """Full simulation environment."""

    def __init__(self, config: Config, seed: int):
        self.config = config
        self.seed = seed
        self.world = WorldState(config, seed)
        self.step_count = 0
        self.recent_events: list[str] = []

        # Spawn organism
        sx, sy = self.world.find_spawn_point()
        self.organism = OrganismState.from_config(config.physiology, sx, sy)

        # Hunter NPC manager
        self.hunter_manager = HunterManager(
            config, self.world, self.world.rng,
            player_spawn=(sx, sy),
        )
        self.hunter_sighting_history: list[list[dict]] = []

        # Trophy manager
        self.trophy_manager = TrophyManager(
            config, self.world, self.world.rng,
            player_spawn=(sx, sy),
        )
        self._trophy_won: bool = False

        # Replay logger
        self.replay = ReplayLogger(seed, config.model_dump())

        # Initial environment update
        update_environment(self.world, 0, config)

    @property
    def done(self) -> bool:
        """Episode is over if dead or trophy found."""
        return not self.organism.alive or self._trophy_won

    @property
    def trophy_won(self) -> bool:
        return self._trophy_won

    def get_observation(self) -> dict:
        """Build current observation for the agent."""
        vis_r = (
            self.config.fog_of_war.visibility_radius
            if self.config.fog_of_war.enabled
            else self.config.sim.observation_radius
        )
        visible_hunters = self.hunter_manager.get_visible_hunters(
            self.organism.x, self.organism.y, vis_r,
        )
        trophy_hint = self.trophy_manager.get_hint(
            self.organism.x, self.organism.y, self.step_count,
        )
        return build_observation(
            self.world, self.organism, self.step_count,
            self.config, self.recent_events,
            visible_hunters=visible_hunters,
            hunter_sighting_history=self.hunter_sighting_history,
            trophy_hint=trophy_hint,
        )

    def step(self, action: ActionType) -> dict:
        """Advance one timestep with the given action. Returns step result dict."""
        org = self.organism
        world = self.world
        cfg = self.config
        dt = cfg.sim.dt_hours
        diff = cfg.difficulty

        time_info = get_time_info(self.step_count, cfg)
        events: dict = {"encounter": False, "rain": False, "hazard_events": [], "delta": {}}
        event_messages: list[str] = []

        # ── Apply action ───────────────────────────────────────────────
        movement_cost = world.movement_cost[org.y, org.x]

        if action in ACTION_DIRECTION:
            dx, dy = ACTION_DIRECTION[action]
            nx, ny = org.x + dx, org.y + dy
            if world.in_bounds(nx, ny):
                org.x, org.y = nx, ny
                event_messages.append(f"Moved {action.value[-1]}")
            else:
                event_messages.append("Move blocked: out of bounds")

        elif action == ActionType.DRINK:
            wa = world.water_availability[org.y, org.x]
            if wa > 0.1:
                gain = min(20.0, wa * 25.0) * dt
                org.hydration = clamp(org.hydration + gain, 0, 100)
                events["delta"]["hydration_gain"] = gain
                event_messages.append(f"Drank water: +{gain:.1f} hydration")
            else:
                event_messages.append("Tried to drink but no water available")

        elif action == ActionType.FORAGE:
            bm = world.vegetation_biomass[org.y, org.x]
            if bm > 0.1:
                gain = min(15.0, bm * 20.0) * dt
                org.energy = clamp(org.energy + gain, 0, 100)
                # Reduce biomass
                world.vegetation_biomass[org.y, org.x] -= 0.1
                world.vegetation_biomass[org.y, org.x] = max(
                    0.0, world.vegetation_biomass[org.y, org.x]
                )
                events["delta"]["energy_gain"] = gain
                event_messages.append(f"Foraged: +{gain:.1f} energy")
            else:
                event_messages.append("Foraged but insufficient vegetation")

        elif action == ActionType.BUILD_SHELTER:
            org.has_shelter = True
            org.shelter_durability = 12  # lasts 12 steps
            event_messages.append("Built shelter")

        elif action == ActionType.REST:
            event_messages.append("Resting")

        elif action == ActionType.HIDE:
            event_messages.append("Hiding")

        elif action == ActionType.SIGNAL:
            event_messages.append("Signaled (no effect)")

        # ── Shelter durability ─────────────────────────────────────────
        if org.has_shelter:
            org.shelter_durability -= 1
            if org.shelter_durability <= 0:
                org.has_shelter = False
                event_messages.append("Shelter collapsed")

        shelter_active = org.has_shelter or world.shelter_quality[org.y, org.x] > 0.3

        # ── Physiology ─────────────────────────────────────────────────
        air_temp = world.air_temperature_c[org.y, org.x]
        deltas = apply_physiology(
            org, action, air_temp, shelter_active,
            movement_cost, dt, cfg.physiology,
            difficulty_mult=diff.drain_multiplier,
        )
        events["delta"].update(deltas)

        # ── Wildlife encounter ─────────────────────────────────────────
        terrain = world.terrain_at(org.x, org.y)
        hour = time_info["hour"]
        encounter_result = check_wildlife_encounter(
            org, world.wildlife_risk[org.y, org.x],
            terrain, hour, action, world.rng,
            hazard_multiplier=diff.wildlife_multiplier,
        )
        events["encounter"] = encounter_result["encounter"]
        if encounter_result["encounter"]:
            event_messages.append(
                f"Wildlife encounter! Injury +{encounter_result['injury_delta']:.1f}"
            )

        # ── Environmental hazards ──────────────────────────────────────
        hazard_events = check_environmental_hazards(
            org, air_temp, shelter_active, terrain,
            world.humidity[org.y, org.x],
            hazard_multiplier=diff.hazard_multiplier,
        )
        events["hazard_events"] = hazard_events
        event_messages.extend(hazard_events)

        # ── Hunter NPC update ──────────────────────────────────────────
        killer_id = self.hunter_manager.update(org.x, org.y)
        if killer_id is not None:
            org.alive = False
            org.cause_of_death = CauseOfDeath.HUNTED
            event_messages.append(f"Killed by hunter #{killer_id}!")

        # Track visible hunter sightings for LLM observation
        vis_r = (
            cfg.fog_of_war.visibility_radius
            if cfg.fog_of_war.enabled
            else cfg.sim.observation_radius
        )
        visible_hunters = self.hunter_manager.get_visible_hunters(
            org.x, org.y, vis_r,
        )
        self.hunter_sighting_history.append(visible_hunters)
        if len(self.hunter_sighting_history) > 20:
            self.hunter_sighting_history = self.hunter_sighting_history[-20:]

        # ── Trophy check ───────────────────────────────────────────────
        if self.trophy_manager.check_found(org.x, org.y):
            self._trophy_won = True
            event_messages.append("TROPHY FOUND! You win!")

        # ── Precipitation ──────────────────────────────────────────────
        if world.rng.random() < world.precip_prob[org.y, org.x]:
            events["rain"] = True
            event_messages.append("Rain")

        # ── Death check ────────────────────────────────────────────────
        dead = check_death(org, cfg.physiology)

        # ── Update age ─────────────────────────────────────────────────
        org.age_steps = self.step_count + 1

        # ── Log to replay ──────────────────────────────────────────────
        local_state = {
            "terrain": terrain.value,
            "air_temp_c": round(air_temp, 1),
            "water_availability": round(world.water_availability[org.y, org.x], 2),
            "vegetation_biomass": round(world.vegetation_biomass[org.y, org.x], 2),
        }

        # Extend events with hunter/trophy data for replay
        if cfg.hunters.enabled:
            events["hunters"] = [h.to_dict() for h in self.hunter_manager.hunters]
        events["trophy_found"] = self._trophy_won

        self.replay.log_step(
            step=self.step_count,
            time_info={"day": time_info["day_of_year"], "hour": time_info["hour"]},
            agent_state=org.to_dict(),
            local_state=local_state,
            action={"action": action.value},
            events=events,
        )

        # ── Store events ───────────────────────────────────────────────
        self.recent_events = event_messages

        # ── Advance world ──────────────────────────────────────────────
        self.step_count += 1
        update_environment(self.world, self.step_count, self.config)

        return {
            "step": self.step_count,
            "alive": org.alive,
            "events": event_messages,
            "dead": dead,
            "trophy_won": self._trophy_won,
        }

    def get_summary(self) -> dict:
        """Generate episode summary."""
        dt = self.config.sim.dt_hours
        steps = self.organism.age_steps
        hours = steps * dt

        if self._trophy_won:
            outcome = EpisodeOutcome.TROPHY_FOUND.value
        elif not self.organism.alive:
            outcome = EpisodeOutcome.DIED.value
        else:
            outcome = EpisodeOutcome.RUNNING.value

        return {
            "survived_steps": steps,
            "survived_hours": round(hours, 1),
            "survived_days": round(hours / 24.0, 1),
            "cause_of_death": self.organism.cause_of_death.value,
            "final_pos": {"x": self.organism.x, "y": self.organism.y},
            "trophy_found": self._trophy_won,
            "outcome": outcome,
        }

    def finalize_replay(self) -> None:
        """Set summary on replay logger."""
        self.replay.set_summary(self.get_summary())
