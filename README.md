# Earth2D Survival — LLM-as-Organism in Realistic 2D Earth Environments

```
    ███████╗ █████╗ ██████╗ ████████╗██╗  ██╗ ██████╗ ██████╗
    ██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║╚════██╗╚════██╗
    █████╗  ███████║██████╔╝   ██║   ███████║ █████╔╝ █████╔╝
    ██╔══╝  ██╔══██║██╔══██╗   ██║   ██╔══██║ ╚═══██╗ ╚═══██╗
    ███████╗██║  ██║██║  ██║   ██║   ██║  ██║██████╔╝██████╔╝
    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═════╝
             S  U  R  V  I  V  A  L
```

> **Can an LLM survive in the wild?**
> Drop a large language model into a procedurally generated 2D world.
> It must drink, eat, regulate body temperature, avoid predators,
> and find a hidden trophy — all by reasoning over text observations.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Game Mechanics](#game-mechanics)
- [Configuration](#configuration)
- [Agents](#agents)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

Earth2D Survival is a research-grade simulation that tests whether LLMs and bio-inspired
optimization algorithms can function as autonomous survival agents. The organism (controlled
by an LLM, heuristic rules, swarm optimization, or random baseline) is dropped onto an 80x60
procedurally generated grid with realistic biomes, dynamic climate, hostile wildlife, roaming
hunter NPCs, and a hidden trophy to find.

**7 agent types** are available:

| Agent | Type | Description |
|-------|------|-------------|
| `heuristic` | Rule-based | 12-tier priority engine with survival logic |
| `llm` | LLM-powered | Gemini/OpenAI with 5-step planning protocol |
| `random` | Baseline | Uniform random action selection |
| `pso` | Swarm | Particle Swarm Optimization |
| `gwo` | Swarm | Grey Wolf Optimization |
| `woa` | Swarm | Whale Optimization Algorithm |
| `aco` | Swarm | Ant Colony Optimization |

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        THE CHALLENGE                                │
 │                                                                     │
 │   ┌───────────┐    ┌────────────┐    ┌──────────────┐              │
 │   │  SURVIVE  │───>│  NAVIGATE  │───>│ FIND TROPHY  │              │
 │   │           │    │            │    │              │              │
 │   │ - Hydrate │    │ - Fog      │    │ - Follow     │              │
 │   │ - Eat     │    │ - Hunters  │    │   hints      │              │
 │   │ - Shelter │    │ - Terrain  │    │ - Win!       │              │
 │   └───────────┘    └────────────┘    └──────────────┘              │
 └─────────────────────────────────────────────────────────────────────┘
```

The LLM receives a JSON observation each turn — vitals, nearby terrain, visible threats,
trophy hints — and must return a structured action with reasoning. Swarm agents instead
deploy virtual clones (particles/wolves/whales/ants) that explore the grid and converge
on optimal positions to guide the organism's movement.

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                         GAME LOOP                                    │
 │                                                                      │
 │  ┌─────────┐   observation   ┌─────────────┐   action   ┌────────┐ │
 │  │         │ ──────────────> │             │ ─────────> │        │ │
 │  │   ENV   │                 │    AGENT    │            │  ENV   │ │
 │  │  (sim/) │ <────────────── │  (agents/)  │ <───────── │ .step()│ │
 │  │         │    reward/done  │             │   events   │        │ │
 │  └────┬────┘                 └──────┬──────┘            └────┬───┘ │
 │       │                             │                        │      │
 │       v                             v                        v      │
 │  ┌─────────┐                 ┌────────────┐          ┌───────────┐ │
 │  │ RENDER  │                 │  DECISION  │          │  REPLAY   │ │
 │  │(render/)│                 │  ENGINE    │          │  (JSON)   │ │
 │  │ Pygame  │                 │            │          │           │ │
 │  │ +Clones │                 │ LLM APIs   │          └───────────┘ │
 │  └─────────┘                 │ Heuristic  │                         │
 │                              │ PSO/GWO    │                         │
 │                              │ WOA/ACO    │                         │
 │                              └────────────┘                         │
 └──────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────┐
 │                     SIMULATION SYSTEMS                               │
 │                                                                      │
 │  world.py ─── Procedural terrain, elevation, water, vegetation      │
 │  dynamics.py ─ Climate engine: seasonal + diurnal temperature       │
 │  physiology.py Organism state: hydration, energy, temp, fatigue     │
 │  hazards.py ── Wildlife encounters, environmental stress            │
 │  hunters.py ── NPC patrol/chase AI with variable detection radii    │
 │  trophy.py ─── Hidden objective with graduated proximity hints      │
 │  observation.py Agent-facing observation builder (fog-of-war)       │
 │  replay.py ─── Deterministic episode logging for reproducibility    │
 └──────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────┐
 │                     SWARM AGENT PIPELINE                             │
 │                                                                      │
 │  observation ──> SwarmAgentBase ──> _run_swarm_iteration()           │
 │       │              │                    │                          │
 │       │         ┌────┴─────┐        ┌─────┴──────────┐              │
 │       │         │ fitness  │        │ PSO: velocity   │              │
 │       │         │ evaluate │        │ GWO: wolf pack  │              │
 │       │         │ + cache  │        │ WOA: spiral     │              │
 │       │         └──────────┘        │ ACO: pheromone  │              │
 │       │                             └────────┬───────┘              │
 │       │                                      │                      │
 │       v                                      v                      │
 │  world_snapshot ──>              clone_positions + best_target       │
 │  (numpy arrays)                        │             │              │
 │                                        v             v              │
 │                                   renderer      MOVE_N/S/E/W       │
 │                               (cyan circles)   (one step)          │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Simulation
- **Procedural World Generation** — 7 terrain types (Plains, Forest, Desert, Tundra, Mountain, Water, Swamp) with fractal noise, rivers, and lakes
- **Dynamic Climate** — Seasonal (365-day) and diurnal (24-hour) temperature cycles, elevation lapse rate, per-biome parameters
- **Realistic Physiology** — Hydration, energy, core temperature, fatigue, injury, infection with interconnected drain/recovery mechanics
- **Deterministic Replay** — Seeded RNG ensures every episode is perfectly reproducible

### Hunter NPCs
- **Variable Detection Radii** — Each hunter has a random radius (3-8 cells); the LLM must *estimate* it from observations
- **Patrol + Chase AI** — Random patrol patterns that switch to pursuit when the organism enters detection range
- **Information Asymmetry** — Hunters are visible, but their detection radius is hidden. The LLM infers it from sighting distances and chase activation

```
                     Detection Radius (hidden)
                    ┌─────────────────────┐
                    │         . . .       │
                    │       .       .     │
                    │     .    [H]    .   │   H = Hunter (patrolling)
                    │       .       .     │   @ = Agent
                    │         . . .       │
                    └─────────────────────┘

                          @ ← Agent is outside radius
                              (hunter hasn't noticed yet)

     When agent enters radius:
                    ┌─────────────────────┐
                    │         . . .       │
                    │       .       .     │
                    │     .   [H]→→  .    │   Hunter starts chasing!
                    │       .     ↘ .     │   Speed: 2 cells/step
                    │         . . @ .     │   Kill: Manhattan dist <= 1
                    └─────────────────────┘
```

### Trophy Objective
- **Hidden Goal** — A trophy is placed 22-30 cells from spawn (varies by config); find it to win
- **Graduated Hints** — Distance buckets (very close / close / moderate / far), warmer/colder feedback, directional hints every N steps
- **Risk-Reward Tradeoff** — Pursuing the trophy means venturing into unknown, potentially hunter-infested territory

### Fog of War
- **Limited Visibility** — Agent can only see within a configurable radius (default: 5-7 cells)
- **Three-State Cells** — Visible (full detail), Explored (dimmed), Unexplored (dark)
- **Exploration Incentive** — Must physically explore to reveal the map

```
    Fog of War Visualization:

    ░░░░░░░░░░░░░░░░░░░░░░░
    ░░░░░░░░░░░░░░░░░░░░░░░    ░ = Unexplored (dark)
    ░░░░░▒▒▒▒▒▒▒▒▒░░░░░░░░    ▒ = Explored (dimmed)
    ░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░    █ = Visible (full detail)
    ░░░▒▒▒█████████▒▒░░░░░░
    ░░░▒▒██████@████▒▒░░░░░    @ = Agent (center)
    ░░░▒▒█████████▒▒▒░░░░░░
    ░░░░▒▒▒▒▒▒▒▒▒▒▒░░░░░░░
    ░░░░░▒▒▒▒▒▒▒▒▒░░░░░░░░
    ░░░░░░░░░░░░░░░░░░░░░░░
    ░░░░░░░░░░░░░░░░░░░░░░░
```

### Difficulty Scaling

Global multipliers that scale the challenge:

| Multiplier | What It Affects |
|-----------|-----------------|
| `drain_multiplier` | Hydration/energy loss rate |
| `hazard_multiplier` | Environmental stress |
| `wildlife_multiplier` | Wildlife encounter probability |
| `temperature_extremity` | Seasonal/diurnal temperature amplitude |

### Bio-Inspired Swarm Agents
- **4 Optimization Algorithms** — PSO, GWO, WOA, ACO explore the grid using virtual clone populations
- **Clone Visualization** — Faded pulsing copies of the organism scattered across the map
- **Visibility Radius Circles** — Each clone has a cyan semi-transparent search radius circle (similar to hunter detection radius in red)
- **Convergence Behavior** — Clones converge toward high-fitness positions (near trophy, away from hunters, near resources)
- **Survival Override** — When vitals are critical, swarm agents automatically delegate to the heuristic agent

```
    Swarm Clone Visualization:

       ╭─────╮                         ╭─────╮
      │  .  │ ← cyan search           │  .  │
      │  c  │    radius circle         │  c  │
       ╰─────╯                         ╰─────╯
                    ╭─────╮
                   │  .  │
                   │  c  │ ← clones converge
                   ╰─────╯   toward best
                                  ╭─────╮
            ╭─────╮              │  .  │
           │  .  │              │  c  │
           │  c  │              ╰─────╯
           ╰─────╯
                        [@] ← real organism follows
                             global best position

    c = clone (faded)    @ = organism    . = search area
```

### Visuals
- **Layered Pygame Renderer** — Terrain, grid, overlays, hunters, trophy, fog, trail, clones, agent, HUD
- **7 Overlay Modes** — Toggle temperature, water, vegetation, wildlife, shelter, elevation heatmaps
- **Pulsing Animations** — Agent glow, trophy shimmer, hunter chase indicators, clone pulse
- **Rich HUD** — Vitals bars, time, event log, hunter proximity, trophy status, swarm clone count
- **Video Recording** — Export episodes as MP4 or GIF

---

## Installation

### Prerequisites
- Python >= 3.11
- A Gemini API key (primary) and/or OpenAI API key (fallback) for LLM agents

### Setup

```bash
# Clone the repository
git clone https://github.com/ritwikraha/earth2d-survival.git
cd earth2d-survival

# Install core dependencies
pip install -e .

# Install with all extras
pip install -e ".[dev,recording,llm-openai]"

# Configure API keys (for LLM agent only)
cp .env.example .env
# Edit .env with your API keys:
#   GEMINI_API_KEY=your-key-here
#   OPENAI_API_KEY=your-key-here   (optional fallback)
```

### Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `numpy>=1.26` | Numerical computing, terrain generation | Yes |
| `pydantic>=2.0` | Configuration validation | Yes |
| `pygame>=2.5` | Real-time visualization | Yes |
| `typer>=0.9` | CLI framework | Yes |
| `rich>=13.0` | Rich terminal output | Yes |
| `pyyaml>=6.0` | Config file parsing | Yes |
| `python-dotenv>=1.0` | Environment variable loading | Yes |
| `google-genai>=1.0` | Gemini LLM provider | Yes |
| `openai>=1.0` | OpenAI fallback provider | Optional (`pip install -e ".[llm-openai]"`) |
| `imageio>=2.9` | Video/GIF recording | Optional (`pip install -e ".[recording]"`) |
| `imageio-ffmpeg>=0.4` | MP4 encoding | Optional (`pip install -e ".[recording]"`) |
| `pytest>=8.0` | Testing | Optional (`pip install -e ".[dev]"`) |

---

## Quick Start

### Run a Visual Episode (Heuristic Agent)

```bash
# Forest biome — normal difficulty
python -m cli run_episode --config configs/forest.yaml --agent heuristic --seed 42

# Hunt mode — 6 hunters, trophy, fog of war, hard difficulty
python -m cli run_episode --config configs/hunt.yaml --agent heuristic --seed 42
```

### Run Headless (No Window)

```bash
python -m cli run_episode --config configs/hunt.yaml --agent heuristic --headless --seed 42
```

### Run with LLM Agent

```bash
# Uses Gemini by default, falls back to OpenAI if quota exhausted
python -m cli run_episode --config configs/hunt.yaml --agent llm --seed 42

# Force OpenAI provider
python -m cli run_episode --config configs/hunt.yaml --agent llm --provider openai --seed 42
```

### Run with Swarm Optimization Agents

```bash
# Particle Swarm Optimization — 10 particles exploring the grid
python -m cli run_episode --config configs/hunt.yaml --agent pso --seed 42

# Grey Wolf Optimization — 8 wolves in alpha/beta/delta hierarchy
python -m cli run_episode --config configs/hunt.yaml --agent gwo --seed 42

# Whale Optimization Algorithm — 6 whales with spiral bubble-net
python -m cli run_episode --config configs/hunt.yaml --agent woa --seed 42

# Ant Colony Optimization — 12 ants laying pheromone trails
python -m cli run_episode --config configs/hunt.yaml --agent aco --seed 42
```

### Batch Evaluation

```bash
# Evaluate across 10 seeds, outputs CSV + summary JSON
python -m cli evaluate --config configs/hunt.yaml --agent heuristic --seeds 0-9 --output runs/eval

# Compare swarm agents
python -m cli evaluate --config configs/hunt.yaml --agent pso --seeds 0-4 --output runs/pso_eval
python -m cli evaluate --config configs/hunt.yaml --agent gwo --seeds 0-4 --output runs/gwo_eval
```

### Replay a Saved Episode

```bash
python -m cli replay runs/20260222_001132_seed42.json
```

### Record Video

```bash
# Record as MP4 (requires: pip install -e ".[recording]")
python -m cli run_episode --config configs/hunt.yaml --agent heuristic --record runs/gameplay.mp4

# Record as GIF
python -m cli run_episode --config configs/hunt.yaml --agent pso --record runs/swarm_demo.gif

# Auto-timestamped filename
python -m cli run_episode --config configs/hunt.yaml --agent gwo --record
```

### Keyboard Controls (Visual Mode)

| Key | Action |
|-----|--------|
| `0` | No overlay (terrain only) |
| `1` | Temperature heatmap |
| `2` | Water availability |
| `3` | Vegetation biomass |
| `4` | Wildlife risk |
| `5` | Shelter quality |
| `6` | Elevation map |
| `ESC` | Quit episode |

---

## CLI Reference

### `run_episode` — Run a Single Episode

```bash
python -m cli run_episode [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | `None` | Path to YAML config file. Uses built-in defaults if omitted. |
| `--seed` | `int` | `42` | Random seed for reproducible episodes. |
| `--agent` | `str` | `heuristic` | Agent type: `random`, `heuristic`, `llm`, `pso`, `gwo`, `woa`, `aco` |
| `--headless` | flag | off | Run without the pygame window (terminal output only). |
| `--max-steps` | `int` | config value | Override the max steps from the config file. |
| `--provider` | `str` | auto | Force LLM provider: `gemini` or `openai`. Only used with `--agent llm`. |
| `--record` | `str` | `None` | Record video to path (e.g. `runs/demo.mp4`). Pass without value for auto-timestamped filename. Requires visual mode (incompatible with `--headless`). Supports `.mp4`, `.gif`, `.avi`, `.webm`. |

**Examples:**

```bash
# Basic visual run
python -m cli run_episode --config configs/forest.yaml

# Headless with custom seed
python -m cli run_episode --config configs/hunt.yaml --headless --seed 123

# LLM agent with forced OpenAI provider
python -m cli run_episode --config configs/hunt.yaml --agent llm --provider openai

# Swarm agent with video recording
python -m cli run_episode --config configs/hunt.yaml --agent pso --record runs/pso.mp4

# Override max steps
python -m cli run_episode --config configs/hunt.yaml --max-steps 200 --headless
```

### `evaluate` — Batch Evaluation Across Seeds

```bash
python -m cli evaluate [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | `None` | Path to YAML config file. |
| `--agent` | `str` | `heuristic` | Agent type: `random`, `heuristic`, `llm`, `pso`, `gwo`, `woa`, `aco` |
| `--seeds` | `str` | `0-9` | Seed specification. Supports ranges (`0-9`), lists (`1,3,5`), or mixed (`0-4,10,20`). |
| `--output` | `str` | `runs` | Output directory for CSV metrics and JSON summary. |

**Output files:**
- `evaluation_metrics.csv` — Per-seed metrics (survival steps, cause of death, cells explored, etc.)
- `evaluation_summary.json` — Aggregate statistics (averages, death cause distribution)
- `eval_seed{N}.json` — Individual episode replay files

**Examples:**

```bash
# Evaluate heuristic agent across 10 seeds
python -m cli evaluate --config configs/hunt.yaml --agent heuristic --seeds 0-9

# Compare agents with specific seeds
python -m cli evaluate --config configs/forest.yaml --agent pso --seeds 0-4 --output runs/pso
python -m cli evaluate --config configs/forest.yaml --agent gwo --seeds 0-4 --output runs/gwo

# Large-scale evaluation
python -m cli evaluate --config configs/hunt.yaml --agent llm --seeds 0-99 --output runs/llm_100
```

### `replay` — Replay a Saved Episode

```bash
python -m cli replay <replay_file.json>
```

Replays a previously saved episode JSON file in the pygame window.

---

## Game Mechanics

### Organism Physiology

The organism has six vital stats that interact and drain over time:

```
    VITALS DASHBOARD
    ┌────────────────────────────────────────┐
    │  Hydration  [████████████░░░░░░░] 65%  │  ← Drains 0.8/step baseline
    │  Energy     [██████████░░░░░░░░░] 55%  │  ← Drains faster when moving
    │  Core Temp  [========37.0°C========]   │  ← Drifts toward air temp
    │  Fatigue    [███░░░░░░░░░░░░░░░░] 18%  │  ← Accumulates with activity
    │  Injury     [░░░░░░░░░░░░░░░░░░░]  0%  │  ← Wildlife attacks (5-20 dmg)
    │  Infection  [░░░░░░░░░░░░░░░░░░░]  0%  │  ← Follows untreated injury
    └────────────────────────────────────────┘

    DEATH CONDITIONS:
    ╔══════════════════╦══════════════════════╗
    ║  Dehydration     ║  Hydration <= 0      ║
    ║  Starvation      ║  Energy <= 0         ║
    ║  Hypothermia     ║  Core Temp < 30°C    ║
    ║  Hyperthermia    ║  Core Temp > 42°C    ║
    ║  Trauma          ║  Injury >= 100       ║
    ║  Infection       ║  Infection >= 100    ║
    ║  Hunted          ║  Hunter adjacent     ║
    ╚══════════════════╩══════════════════════╝
```

### Available Actions (10 total)

```
    MOVEMENT                    SURVIVAL
    ┌───┐                      ┌───────────────┐
    │ N │                      │ REST          │  Recover fatigue (-4.0/step)
    ┌─┴───┴─┐                  │ DRINK         │  +hydration (needs water tile)
    │W     E│                  │ FORAGE        │  +energy (needs vegetation)
    └─┬───┬─┘                  │ BUILD_SHELTER │  Protection from elements
    │ S │                      │ HIDE          │  Avoid wildlife/hunters
    └───┘                      │ SIGNAL        │  (reserved for future use)
                               └───────────────┘
```

### Terrain Types (7 biomes)

| Terrain | Movement Cost | Water | Food | Temperature | Notes |
|---------|:------------:|:-----:|:----:|:-----------:|-------|
| Plains | 1.0x | Low | Moderate | Moderate | Fast, open ground |
| Forest | 1.5x | Moderate | High | Mild | Good shelter + food |
| Desert | 1.0x | Very Low | Very Low | Extreme heat | Fast but harsh |
| Tundra | 1.8x | Low | Low | Extreme cold | Sparse resources |
| Mountain | 3.0x | Low | Low | Cold (lapse rate) | Very slow |
| Water | Impassable | N/A | N/A | Moderate | Can DRINK at edge |
| Swamp | 2.5x | High | Low | Warm | Disease risk |

### The Hunt

In hunt mode, the organism must balance three competing pressures:

```
              ┌─────────────────────┐
              │    STAY ALIVE       │
              │  (manage vitals)    │
              └────────┬────────────┘
                       │
            ┌──────────┼──────────┐
            │          │          │
            v          v          v
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │  AVOID   │ │  EXPLORE │ │ FIND TROPHY  │
    │ HUNTERS  │ │  THE MAP │ │  (to win)    │
    │          │ │          │ │              │
    │ Estimate │ │ Fog of   │ │ Follow hints │
    │ radii,   │ │ war      │ │ warmer/      │
    │ flee or  │ │ limits   │ │ colder,      │
    │ hide     │ │ sight    │ │ directional  │
    └──────────┘ └──────────┘ └──────────────┘
```

---

## Configuration

All scenarios are defined via YAML configs in `configs/`. Four presets are included:

| Config | Biome | Hunters | Trophy | Fog | Difficulty | Steps |
|--------|-------|---------|--------|-----|------------|-------|
| `forest.yaml` | Temperate forest | 4 NPCs | Yes | Yes (r=6) | Normal (1.0x) | 500 |
| `desert.yaml` | Arid desert | 5 NPCs | Yes | Yes (r=7) | Medium (1.2x drain) | 500 |
| `tundra.yaml` | Frozen tundra | 3 NPCs | Yes | Yes (r=5) | Medium (1.3x temp) | 500 |
| `hunt.yaml` | Mixed terrain | 6 NPCs | Yes | Yes (r=6) | Hard (1.3x drain) | 800 |

### Configuration Reference

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

### Creating Custom Configs

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

---

## Agents

### LLM Agent (`--agent llm`)

The LLM agent uses a multi-provider fallback chain:

```
    ┌──────────────┐     quota      ┌──────────────┐     quota      ┌────────────┐
    │   Gemini     │ ───exhausted──>│   OpenAI     │ ───exhausted──>│ Heuristic  │
    │  2.0 Flash   │                │  GPT-4o-mini │                │  Fallback  │
    └──────────────┘                └──────────────┘                └────────────┘
```

**Planning Protocol**: The LLM follows a 5-step reasoning chain each turn:

1. **ASSESS** — Check vitals, identify critical needs
2. **ANALYZE HUNTERS** — Review sighting history, estimate detection radii
3. **PLAN PATH** — Consider 2-3 options with risk assessment
4. **TROPHY PURSUIT** — Evaluate trophy hints against survival state
5. **DECIDE** — Choose action with confidence score

**Hunter Radius Estimation**: The LLM infers each hunter's hidden detection radius from:
- Distance at first sighting (not chasing = radius > distance)
- Distance when chase activates (radius ~ that distance)
- Safety margin: estimated radius + 2 cells

### Heuristic Agent (`--agent heuristic`)

A priority-based rule engine with 12 tiers:

```
    Priority 1:  Critical dehydration (<25%)  →  DRINK
    Priority 2:  Critical energy (<25%)       →  FORAGE
    Priority 3:  Temperature danger           →  BUILD_SHELTER / REST
    Priority 4:  High fatigue (>75%)          →  REST
    Priority 5:  Hunter nearby (<=6 cells)    →  FLEE (opposite direction)
    Priority 6:  Hunter approaching (<=10)    →  HIDE
    Priority 7:  Wildlife risk (>0.3)         →  HIDE
    Priority 8:  Low hydration (<50%)         →  Move toward water
    Priority 9:  Low energy (<50%)            →  FORAGE / seek vegetation
    Priority 10: Moderate fatigue (>50%)      →  REST
    Priority 11: Trophy hint available        →  Follow direction
    Priority 12: Default                      →  EXPLORE (random move)
```

### Random Agent (`--agent random`)

Uniform random action selection from the valid action mask. Serves as a baseline for evaluation.

### Swarm Optimization Agents

All four swarm agents share a common base (`SwarmAgentBase`) that provides:
- **World data caching** — Numpy arrays for terrain cost, water, vegetation, wildlife risk, elevation
- **Hunter memory** — Tracks visible hunters, forgets after 30 steps of no sighting
- **Trophy estimation** — Blends directional hints and warm/cold feedback into a direction vector
- **Fitness function** — Evaluates candidate positions: trophy attraction + hunter avoidance + resource urgency + terrain cost
- **Survival override** — Delegates to the heuristic agent when vitals are critical (hydration < 20, energy < 20, fatigue > 85, injury > 70)
- **Clone rendering** — Each clone is drawn as a faded pulsing copy with a cyan search radius circle

#### PSO — Particle Swarm Optimization (`--agent pso`)

```
    ┌──────────────────────────────────────────────────────┐
    │            PARTICLE SWARM OPTIMIZATION                │
    │                                                      │
    │  Each particle has:                                  │
    │    - Position (x, y)                                 │
    │    - Velocity (vx, vy)                               │
    │    - Personal best position                          │
    │    - Personal best fitness                           │
    │                                                      │
    │  Global best: best position found by any particle    │
    │                                                      │
    │  Velocity update:                                    │
    │    v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest-pos)│
    │                                                      │
    │  Parameters: w=0.7  c1=1.5  c2=2.0  max_vel=3.0    │
    │  Population: 10 particles                            │
    └──────────────────────────────────────────────────────┘
```

#### GWO — Grey Wolf Optimization (`--agent gwo`)

```
    ┌──────────────────────────────────────────────────────┐
    │           GREY WOLF OPTIMIZATION                      │
    │                                                      │
    │  Pack hierarchy (ranked by fitness):                  │
    │    Alpha (best) ──> Beta (2nd) ──> Delta (3rd)      │
    │                         │                            │
    │                    Omega (rest)                       │
    │                                                      │
    │  Each wolf moves toward the weighted average of      │
    │  alpha, beta, and delta positions.                   │
    │                                                      │
    │  Parameter 'a' decays 2 → 0 over the episode:       │
    │    a=2: exploration (wide search)                    │
    │    a→0: exploitation (tight convergence)             │
    │                                                      │
    │  Population: 8 wolves                                │
    └──────────────────────────────────────────────────────┘
```

#### WOA — Whale Optimization Algorithm (`--agent woa`)

```
    ┌──────────────────────────────────────────────────────┐
    │          WHALE OPTIMIZATION ALGORITHM                  │
    │                                                      │
    │  Two hunting mechanisms (50/50 probability):          │
    │                                                      │
    │  1. Shrinking Encirclement          2. Spiral        │
    │     ╭─ ─ ─ ─╮                        ↗ ↗            │
    │     │ .→.→. │                      ↗ .              │
    │     │  [B]  │ ← toward best      . ← bubble-net     │
    │     │ .→.→. │                    . spiral toward      │
    │     ╰─ ─ ─ ─╯                   [B]  best whale      │
    │                                                      │
    │  Exploration: when |A| >= 1, move toward random      │
    │  whale instead of best (built-in diversity)          │
    │                                                      │
    │  Population: 6 whales                                │
    └──────────────────────────────────────────────────────┘
```

#### ACO — Ant Colony Optimization (`--agent aco`)

```
    ┌──────────────────────────────────────────────────────┐
    │          ANT COLONY OPTIMIZATION                      │
    │                                                      │
    │  Pheromone grid (80x60) — updated each step          │
    │                                                      │
    │  Each ant walks from organism:                       │
    │    - Choose neighbors probabilistically               │
    │    - prob ~ pheromone^alpha * fitness^beta            │
    │    - Walk for search_radius steps                    │
    │                                                      │
    │  After all ants walk:                                │
    │    - Best ant's path gets pheromone deposit           │
    │    - All pheromone evaporates (decay = 0.1)          │
    │    - Clamp pheromone to [0.1, 10.0]                  │
    │                                                      │
    │  Parameters: alpha=1.0  beta=2.0  decay=0.1          │
    │  Population: 12 ants                                 │
    └──────────────────────────────────────────────────────┘
```

---

## Demo

### Running Your First Episode

```bash
# 1. Start with the forest (normal mode)
python -m cli run_episode --config configs/forest.yaml --agent heuristic

# 2. Try the hunt (hard mode — 6 hunters + trophy + fog)
python -m cli run_episode --config configs/hunt.yaml --agent heuristic

# 3. Watch the LLM think and plan
python -m cli run_episode --config configs/hunt.yaml --agent llm --seed 7

# 4. Watch swarm clones converge on optimal positions
python -m cli run_episode --config configs/hunt.yaml --agent pso --seed 42
```

### What You'll See

```
    ┌──────────────────────────────────────┬──────────────────────┐
    │                                      │   EARTH2D SURVIVAL   │
    │   ░░░░░░░░░░░░░░░░░░░               │                      │
    │   ░░░░▒▒▒▒▒▒▒▒▒░░░░░░               │   Day 3  14:00       │
    │   ░░▒▒▒███████▒▒▒░░░░               │                      │
    │   ░░▒██ . . T . ██▒░░               │   Hydration ████░ 62% │
    │   ░░▒█ . H . . . █▒░░  ← hunter     │   Energy    ███░░ 48% │
    │   ░░▒█ . . @ . . █▒░░  ← you        │   Temp      37.1°C   │
    │   ░░▒██ . c . c ██▒░░  ← clones     │   Fatigue   ██░░░ 35% │
    │   ░░▒▒▒███████▒▒▒░░░░               │                      │
    │   ░░░░▒▒▒▒▒▒▒▒▒░░░░░░               │   Hunters: 6 alive   │
    │   ░░░░░░░░░░░░░░░░░░░               │   Nearest: 5 cells   │
    │                                      │   Swarm: 10 clones   │
    │   T = Trophy   H = Hunter            │   Trophy:  moderate   │
    │   @ = You      c = Clone             │                      │
    │   ░ = Fog                            │   [Events...]        │
    └──────────────────────────────────────┴──────────────────────┘
```

### Sample Episode Output

```
$ python -m cli run_episode --config configs/hunt.yaml --agent pso --headless --seed 42

INFO: Running episode: agent=pso, seed=42, map=80x60, max_steps=800

=== Episode Summary ===
  Survived: 87 steps (87.0 hours / 3.62 days)
  Cause of death: HUNTED
  Cells explored: 17
  Exploration rate: 0.515
  Near-death events: 3
  Wildlife encounters: 0
  Replay saved: runs/20260222_001132_seed42.json
```

### Batch Evaluation Results

```
$ python -m cli evaluate --config configs/hunt.yaml --agent heuristic --seeds 0-4

=== Evaluation Summary ===
  Agent: heuristic
  Seeds: 5
  Avg survived steps: 87.6
  Avg survived hours: 87.6
  Death causes: {'HUNTED': 1, 'STARVATION': 4}
  Trophy wins: 0
```

```
    ┌──────────────────────────────────────────────────────────┐
    │                  VIDEO RECORDING                         │
    │                                                          │
    │   run_episode ──┬── pygame render ──> screen             │
    │                 └── capture frame ──> numpy array         │
    │                                          │               │
    │                          on exit:        v               │
    │                                    ┌──────────┐          │
    │                                    │ imageio  │          │
    │                                    │ encode   │          │
    │                                    └──┬───┬───┘          │
    │                                 .mp4  │   │  .gif        │
    │                                       v   v              │
    │                                   saved to disk          │
    └──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
earth_33/
│
├── configs/                    # Scenario configuration files
│   ├── forest.yaml            #   Temperate forest (normal)
│   ├── desert.yaml            #   Arid desert (medium)
│   ├── tundra.yaml            #   Frozen tundra (medium)
│   └── hunt.yaml              #   Full hunt mode (hard)
│
├── sim/                        # Simulation engine
│   ├── env.py                 #   Main environment (orchestrator)
│   ├── world.py               #   Procedural world generation
│   ├── dynamics.py            #   Climate & temperature engine
│   ├── physiology.py          #   Organism vital stats & drain
│   ├── hazards.py             #   Wildlife & environmental hazards
│   ├── hunters.py             #   Hunter NPC patrol/chase/kill AI
│   ├── trophy.py              #   Hidden trophy objective & hints
│   ├── observation.py         #   Agent observation builder
│   └── replay.py              #   Episode replay logger
│
├── agents/                     # Agent implementations
│   ├── base.py                #   Abstract base agent interface
│   ├── llm_agent.py           #   LLM agent (Gemini/OpenAI/fallback)
│   ├── heuristic_agent.py     #   Rule-based priority agent
│   ├── random_agent.py        #   Random baseline
│   ├── swarm_base.py          #   Shared swarm agent base class
│   ├── pso_agent.py           #   Particle Swarm Optimization
│   ├── gwo_agent.py           #   Grey Wolf Optimization
│   ├── woa_agent.py           #   Whale Optimization Algorithm
│   └── aco_agent.py           #   Ant Colony Optimization
│
├── eval/                       # Evaluation framework
│   ├── runner.py              #   Single & batch episode runners
│   └── metrics.py             #   Metrics computation
│
├── render/                     # Visualization
│   ├── pygame_renderer.py     #   Pygame renderer (terrain/fog/clones/HUD)
│   └── palettes.py            #   Color palettes & constants
│
├── config_io/                  # Configuration management
│   ├── schema.py              #   Pydantic models & enums
│   ├── config.py              #   Config loading & defaults
│   └── utils.py               #   Utility functions
│
├── cli/                        # Command-line interface
│   ├── __main__.py            #   Entry point & dispatcher
│   ├── run_episode.py         #   Single episode command
│   ├── evaluate.py            #   Batch evaluation command
│   └── replay.py              #   Replay viewer command
│
├── data/                       # Data ingestion (optional GIS)
│   ├── ingest.py
│   └── tiles.py
│
├── tests/                      # Test suite (pytest)
│   ├── test_schema.py         #   Config/schema validation (6 tests)
│   ├── test_death.py          #   Death condition logic (7 tests)
│   ├── test_hunters.py        #   Hunter NPC behavior (11 tests)
│   ├── test_trophy.py         #   Trophy placement & hints (9 tests)
│   ├── test_determinism.py    #   RNG reproducibility (2 tests)
│   └── test_replay.py         #   Replay round-trip (1 test)
│
├── runs/                       # Episode outputs (gitignored)
├── .env.example                # API key template
├── .gitignore
├── pyproject.toml              # Project metadata & dependencies
└── README.md                   # You are here
```

---

## Testing

```bash
# Run the full test suite (36 tests)
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_hunters.py -v    # 11 hunter tests
python -m pytest tests/test_trophy.py -v     # 9 trophy tests
python -m pytest tests/test_death.py -v      # 7 death condition tests

# Run with coverage
python -m pytest tests/ --cov=sim --cov=agents --cov-report=term-missing
```

### Test Categories

```
    ┌──────────────────────────────────────────────────────────┐
    │                    TEST SUITE (36)                        │
    │                                                          │
    │  Schema (6)        Death (7)         Determinism (2)     │
    │  ├─ valid action   ├─ dehydration    ├─ world seeds      │
    │  ├─ from string    ├─ starvation     └─ episode replay   │
    │  ├─ defaults       ├─ hypothermia                        │
    │  ├─ invalid        ├─ hyperthermia   Replay (1)          │
    │  ├─ confidence     ├─ trauma         └─ round trip       │
    │  └─ action mask    ├─ infection                          │
    │                    └─ alive          Hunters (11)         │
    │                                     ├─ spawn count       │
    │  Trophy (9)                         ├─ spawn distance    │
    │  ├─ placement dist                  ├─ no water spawn    │
    │  ├─ not on water                    ├─ detection range   │
    │  ├─ found adjacent                  ├─ patrol movement   │
    │  ├─ found same cell                 ├─ chase activates   │
    │  ├─ not found far                   ├─ kill on contact   │
    │  ├─ warmer/colder                   ├─ no kill far       │
    │  ├─ direction hint                  ├─ disabled mode     │
    │  ├─ disabled mode                   ├─ visible in radius │
    │  └─ found only once                 └─ no radius leak    │
    └──────────────────────────────────────────────────────────┘
```

---

## Contributing

Contributions are welcome! Here's how to get started:

### Getting Set Up

```bash
# Fork and clone the repo
git clone https://github.com/<your-username>/earth2d-survival.git
cd earth2d-survival

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run the test suite to verify your setup
python -m pytest tests/ -v
```

### Development Workflow

```
    1. Fork & Clone          2. Create Branch
    ┌──────────────┐        ┌──────────────────────┐
    │  git clone   │  ───>  │  git checkout -b      │
    │  your-fork   │        │  feature/my-feature   │
    └──────────────┘        └──────────────────────┘
                                      │
    4. Open PR               3. Make Changes + Test
    ┌──────────────┐        ┌──────────────────────┐
    │  gh pr create│  <───  │  pytest tests/ -v     │
    │              │        │  (all 36 must pass)   │
    └──────────────┘        └──────────────────────┘
```

### Areas for Contribution

- **New Agent Strategies** — Implement new agent types (RL, tree search, multi-agent)
- **Swarm Algorithms** — Add more bio-inspired algorithms (firefly, cuckoo search, harmony search)
- **Biome Expansion** — Add new terrain types (volcanic, coastal, urban)
- **Hunter AI** — Smarter pathfinding (A*), coordinated pack behavior
- **Visualization** — Additional overlays, minimap, agent thought bubbles
- **Real GIS Data** — Integration with elevation/climate datasets via `data/`
- **Benchmarking** — Compare LLM providers, prompt engineering experiments
- **Multi-Agent** — Multiple organisms cooperating or competing on the same map

### Guidelines

1. **Tests required** — All PRs must include tests for new functionality
2. **Backward compatible** — New config fields must have defaults (see `enabled: bool = False` pattern)
3. **Deterministic** — All randomness must flow through the seeded `np.random.default_rng`
4. **Type hints** — Use type annotations for all function signatures
5. **Docstrings** — Document classes and public methods

---

## Citation

If you use Earth2D Survival in your research, please cite:

```bibtex
@software{earth2d_survival_2025,
  title     = {Earth2D Survival: LLM-as-Organism in Realistic 2D Earth Environments},
  author    = {Ritwik Raha},
  year      = {2025},
  url       = {https://github.com/ritwikraha/earth2d-survival},
  note      = {A simulation framework for evaluating LLM autonomous survival
               and planning capabilities in procedurally generated environments
               with adversarial NPC hunters, fog of war, bio-inspired swarm
               optimization agents, and objective-driven trophy pursuit.}
}
```

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---

<p align="center">

```
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   "Can it survive? Can it plan? Can it win?"          ║
    ║                                                       ║
    ║          Drop an LLM into the wild and find out.      ║
    ║    Or let a swarm of wolves hunt the trophy for you.  ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
```

</p>
