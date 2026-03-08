# Configuration

All scenarios are defined via YAML configs in `configs/`. Four presets are included:

| Config | Biome | Hunters | Trophy | Fog | Difficulty | Steps |
|--------|-------|---------|--------|-----|------------|-------|
| `forest.yaml` | Temperate forest | 4 NPCs | Yes | Yes (r=6) | Normal (1.0x) | 500 |
| `desert.yaml` | Arid desert | 5 NPCs | Yes | Yes (r=7) | Medium (1.2x drain) | 500 |
| `tundra.yaml` | Frozen tundra | 3 NPCs | Yes | Yes (r=5) | Medium (1.3x temp) | 500 |
| `hunt.yaml` | Mixed terrain | 6 NPCs | Yes | Yes (r=6) | Hard (1.3x drain) | 800 |

## Configuration Reference

All config fields with their defaults:

```yaml
# ── Map Generation ──────────────────────────────────────────
map:
  width: 80                    # Grid width in cells
  height: 60                   # Grid height in cells
  terrain_weights:             # Relative biome weights (normalized)
    PLAINS: 0.30
    FOREST: 0.25
    DESERT: 0.10
    TUNDRA: 0.05
    MOUNTAIN: 0.10
    WATER: 0.10
    SWAMP: 0.10
  num_lakes: 3                 # Number of procedural lakes
  num_rivers: 2                # Number of procedural rivers
  elevation_octaves: 4         # Fractal noise octaves
  elevation_scale: 0.05        # Fractal noise scale

# ── Simulation ──────────────────────────────────────────────
sim:
  dt_hours: 1.0                # Simulated hours per step
  max_steps: 500               # Maximum episode length
  observation_radius: 3        # Cells visible around organism
  start_hour: 6                # Starting hour of day (0-23)
  start_day: 80                # Starting day of year (1-365)

# ── Hunters ─────────────────────────────────────────────────
hunters:
  enabled: false               # Toggle hunter NPCs
  count: 5                     # Number of hunters on map
  min_detection_radius: 3      # Minimum detection radius (cells)
  max_detection_radius: 8      # Maximum detection radius (cells)
  chase_speed: 2               # Hunter speed when chasing (cells/step)
  patrol_speed: 1              # Hunter speed when patrolling
  spawn_margin: 15             # Min distance from organism spawn

# ── Trophy ──────────────────────────────────────────────────
trophy:
  enabled: false               # Toggle trophy objective
  min_distance_from_spawn: 25  # Min distance from organism spawn
  hint_interval: 10            # Steps between directional hints
  warm_cold_enabled: true      # Enable warmer/colder feedback

# ── Fog of War ──────────────────────────────────────────────
fog_of_war:
  enabled: false               # Toggle fog of war
  visibility_radius: 5         # Cells visible around organism
  explored_dim_factor: 0.4     # Dimming of explored cells (0-1)

# ── Difficulty ──────────────────────────────────────────────
difficulty:
  drain_multiplier: 1.0        # Hydration/energy drain rate
  hazard_multiplier: 1.0       # Environmental stress
  wildlife_multiplier: 1.0     # Wildlife encounter probability
  temperature_extremity: 1.0   # Temperature amplitude

# ── Physiology ──────────────────────────────────────────────
physiology:
  initial_hydration: 80.0      # Starting hydration (0-100)
  initial_energy: 80.0         # Starting energy (0-100)
  initial_core_temp: 37.0      # Starting core temperature (C)
  initial_fatigue: 10.0        # Starting fatigue (0-100)
  initial_injury: 0.0          # Starting injury (0-100)
  initial_infection: 0.0       # Starting infection (0-100)
  hydration_base_drain: 0.8    # Hydration loss per step
  energy_base_drain: 0.6       # Energy loss per step
  fatigue_base_gain: 0.7       # Fatigue gain per step
  rest_fatigue_recovery: 4.0   # Fatigue recovered per REST action

# ── Swarm Agents ────────────────────────────────────────────
swarm:
  population_size: 10          # Number of virtual clones
  search_radius: 30            # Clone search area (cells)
  inner_iterations: 3          # Algorithm iterations per step

# ── Rendering ───────────────────────────────────────────────
render:
  cell_size: 10                # Pixels per grid cell
  fps: 30                      # Target frame rate
  trail_length: 50             # Agent trail breadcrumb length
  hud_width: 280               # HUD panel width (pixels)

# ── Agent ───────────────────────────────────────────────────
agent:
  type: heuristic              # Default agent type
  llm_provider: gemini         # LLM provider (gemini/openai)
  llm_model: gemini-2.0-flash  # LLM model name
  llm_max_retries: 2           # Retry count for invalid responses
  llm_temperature: 0.7         # LLM sampling temperature
```

## Creating Custom Configs

```yaml
# configs/my_scenario.yaml
map:
  width: 80
  height: 60
  terrain_weights:
    FOREST: 40
    PLAINS: 30
    DESERT: 10
    WATER: 15
    MOUNTAIN: 5

sim:
  max_steps: 600
  start_day: 180        # Mid-summer
  start_hour: 8

hunters:
  enabled: true
  count: 4
  min_detection_radius: 4
  max_detection_radius: 10
  chase_speed: 2
  spawn_margin: 20

trophy:
  enabled: true
  min_distance_from_spawn: 25
  hint_interval: 10

fog_of_war:
  enabled: true
  visibility_radius: 8

difficulty:
  drain_multiplier: 1.5   # Brutal survival drain
  hazard_multiplier: 1.3
  wildlife_multiplier: 1.2
  temperature_extremity: 1.4

swarm:
  population_size: 15      # More clones for better coverage
  search_radius: 40        # Wider search area
  inner_iterations: 5      # More iterations per step

physiology:
  initial_hydration: 70
  initial_energy: 65
```

All config fields are **backward compatible** — omitting any field uses the default value, so minimal configs work fine.
