# Earth33 Survival — LLM-as-Organism in Realistic 2D Earth Environments

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

Earth2D Survival is a research-grade simulation that tests whether LLMs can function as
autonomous survival agents. The organism (controlled by an LLM, heuristic rules, or random
baseline) is dropped onto an 80x60 procedurally generated grid with realistic biomes, dynamic
climate, hostile wildlife, roaming hunter NPCs, and a hidden trophy to find.

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
trophy hints — and must return a structured action with reasoning. It plans paths, estimates
hunter detection radii from sighting history, and balances survival needs against trophy pursuit.

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
 │  │ RENDER  │                 │  LLM APIs  │          │  REPLAY   │ │
 │  │(render/)│                 │  Gemini    │          │  (JSON)   │ │
 │  │ Pygame  │                 │  OpenAI    │          │           │ │
 │  └─────────┘                 │  Heuristic │          └───────────┘ │
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
```

---

## Features

### Core Simulation
- **Procedural World Generation** — 7 terrain types (Plains, Forest, Desert, Tundra, Mountain, Water, Swamp) with fractal noise, rivers, and lakes
- **Dynamic Climate** — Seasonal (365-day) and diurnal (24-hour) temperature cycles, elevation lapse rate, per-biome parameters
- **Realistic Physiology** — Hydration, energy, core temperature, fatigue, injury, infection with interconnected drain/recovery mechanics
- **Deterministic Replay** — Seeded RNG ensures every episode is perfectly reproducible

### Hunter NPCs
- **Variable Detection Radii** — Each hunter has a random radius (3–8 cells); the LLM must *estimate* it from observations
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
- **Hidden Goal** — A trophy is placed 30+ cells from spawn; find it to win
- **Graduated Hints** — Distance buckets (very close / close / moderate / far), warmer/colder feedback, directional hints every N steps
- **Risk-Reward Tradeoff** — Pursuing the trophy means venturing into unknown, potentially hunter-infested territory

### Fog of War
- **Limited Visibility** — Agent can only see within a configurable radius (default: 6 cells)
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
- **Drain Multiplier** — 1.3x hydration/energy loss rate
- **Hazard Multiplier** — 1.2x environmental stress
- **Wildlife Multiplier** — 1.4x encounter probability
- **Temperature Extremity** — 1.2x seasonal/diurnal amplitude

### Visuals
- **Layered Pygame Renderer** — Terrain, grid, overlays, hunters, trophy, fog, trail, agent, HUD
- **7 Overlay Modes** — Toggle temperature, water, vegetation, wildlife, shelter, elevation heatmaps
- **Pulsing Animations** — Agent glow, trophy shimmer, hunter chase indicators
- **Rich HUD** — Vitals bars, time, event log, hunter proximity, trophy status

---

## Installation

### Prerequisites
- Python >= 3.9
- A Gemini API key (primary) and/or OpenAI API key (fallback) for LLM agents

### Setup

```bash
# Clone the repository
git clone https://github.com/ritwikraha/earth2d-survival.git
cd earth2d-survival

# Install dependencies
pip install -e ".[dev]"

# Configure API keys (for LLM agent)
cp .env.example .env
# Edit .env with your API keys:
#   GEMINI_API_KEY=your-key-here
#   OPENAI_API_KEY=your-key-here   (optional fallback)
```

### Dependencies
| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing, terrain generation |
| `pydantic` | Configuration validation |
| `pygame` | Real-time visualization |
| `typer` + `rich` | CLI framework |
| `pyyaml` | Config file parsing |
| `google-genai` | Gemini LLM provider |
| `openai` | OpenAI fallback provider (optional) |

---

## Quick Start

### Run a Visual Episode (Heuristic Agent)

```bash
# Forest biome — gentle difficulty
python -m cli run_episode --config configs/forest.yaml --agent heuristic --seed 42

# Hunt mode — hunters, trophy, fog of war, hard difficulty
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
```

### Batch Evaluation

```bash
# Evaluate across 10 seeds, outputs CSV + summary JSON
python -m cli evaluate --config configs/hunt.yaml --agent heuristic --seeds 0-9 --output runs/eval
```

### Replay a Saved Episode

```bash
python -m cli replay runs/20260219_001132_seed42.json
```

### Keyboard Controls (Visual Mode)
| Key | Action |
|-----|--------|
| `0` | No overlay |
| `1` | Temperature heatmap |
| `2` | Water availability |
| `3` | Vegetation biomass |
| `4` | Wildlife risk |
| `5` | Shelter quality |
| `6` | Elevation map |
| `ESC` | Quit |

---

## Game Mechanics

### Organism Physiology

The organism has six vital stats that interact and drain over time:

```
    VITALS DASHBOARD
    ┌────────────────────────────────────────┐
    │  Hydration  [████████████░░░░░░░] 65%  │  ← Drains every step
    │  Energy     [██████████░░░░░░░░░] 55%  │  ← Drains faster when moving
    │  Core Temp  [========37.0°C========]   │  ← Drifts toward air temp
    │  Fatigue    [███░░░░░░░░░░░░░░░░] 18%  │  ← Accumulates with activity
    │  Injury     [░░░░░░░░░░░░░░░░░░░]  0%  │  ← Wildlife attacks
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

### Available Actions

```
    MOVEMENT                    SURVIVAL
    ┌───┐                      ┌───────────────┐
    │ N │                      │ REST          │  Recover fatigue
    ┌─┴───┴─┐                  │ DRINK         │  +hydration (needs water)
    │W     E│                  │ FORAGE        │  +energy (needs vegetation)
    └─┬───┬─┘                  │ BUILD_SHELTER │  Protection from elements
    │ S │                      │ HIDE          │  Avoid wildlife/hunters
    └───┘                      │ SIGNAL        │  (reserved)
                               └───────────────┘
```

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

| Config | Biome | Hunters | Trophy | Fog | Difficulty |
|--------|-------|---------|--------|-----|------------|
| `forest.yaml` | Temperate forest | No | No | No | Normal |
| `desert.yaml` | Arid desert | No | No | No | Normal |
| `tundra.yaml` | Frozen tundra | No | No | No | Normal |
| `hunt.yaml` | Mixed terrain | 6 NPCs | Yes | Yes | Hard (1.3x) |

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
  explored_dim_factor: 0.4

difficulty:
  drain_multiplier: 1.5   # Brutal survival drain
  hazard_multiplier: 1.3
  wildlife_multiplier: 1.2
  temperature_extremity: 1.4

physiology:
  initial_hydration: 70
  initial_energy: 65
```

All new config fields (hunters, trophy, fog_of_war, difficulty) are **backward compatible** — omitting them defaults to disabled/normal, so old configs work without modification.

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
3. **PLAN PATH** — Consider 2–3 options with risk assessment
4. **TROPHY PURSUIT** — Evaluate trophy hints against survival state
5. **DECIDE** — Choose action with confidence score

**Hunter Radius Estimation**: The LLM infers each hunter's hidden detection radius from:
- Distance at first sighting (not chasing → radius > distance)
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

Uniform random action selection. Serves as a baseline for evaluation.

---

## Demo

### Running Your First Episode

```bash
# 1. Start with the forest (easy mode)
python -m cli run_episode --config configs/forest.yaml --agent heuristic

# 2. Try the hunt (hard mode — hunters + trophy + fog)
python -m cli run_episode --config configs/hunt.yaml --agent heuristic

# 3. Watch the LLM think and plan
python -m cli run_episode --config configs/hunt.yaml --agent llm --seed 7
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
    │   ░░▒██ . . . . ██▒░░               │   Fatigue   ██░░░ 35% │
    │   ░░▒▒▒███████▒▒▒░░░░               │                      │
    │   ░░░░▒▒▒▒▒▒▒▒▒░░░░░░               │   Hunters: 6 alive   │
    │   ░░░░░░░░░░░░░░░░░░░               │   Nearest: 5 cells   │
    │                                      │   Trophy:  moderate   │
    │   ★ = Trophy   H = Hunter            │                      │
    │   @ = You      ░ = Fog               │   [Events...]        │
    └──────────────────────────────────────┴──────────────────────┘
```

### Sample Episode Output

```
$ python -m cli run_episode --config configs/hunt.yaml --agent heuristic --headless --seed 42

INFO: Running episode: agent=heuristic, seed=42, map=80x60, max_steps=800
INFO: Replay saved to runs/20260219_001144_seed42.json

=== Episode Summary ===
  Survived: 56 steps (56.0 hours / 2.33 days)
  Cause of death: HUNTED
  Cells explored: 30
  Exploration rate: 0.536
  Near-death events: 0
  Wildlife encounters: 2
  Replay saved: runs/20260219_001144_seed42.json
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

---

## Project Structure

```
earth_33/
│
├── configs/                    # Scenario configuration files
│   ├── forest.yaml            #   Temperate forest (easy)
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
│   └── random_agent.py        #   Random baseline
│
├── eval/                       # Evaluation framework
│   ├── runner.py              #   Single & batch episode runners
│   └── metrics.py             #   Metrics computation
│
├── render/                     # Visualization
│   ├── pygame_renderer.py     #   Pygame renderer (terrain/fog/HUD)
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
               with adversarial NPC hunters, fog of war, and objective-driven
               trophy pursuit.}
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
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
```

</p>
