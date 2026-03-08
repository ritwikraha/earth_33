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

## Overview

Earth33 Survival is a research-grade simulation that tests whether LLMs and bio-inspired
optimization algorithms can function as autonomous survival agents. The organism is dropped
onto an 80x60 procedurally generated grid with realistic biomes, dynamic climate, hostile
wildlife, roaming hunter NPCs, and a hidden trophy to find.

## Features

- **Procedural World** — 7 terrain types with fractal noise, rivers, and lakes
- **Dynamic Climate** — Seasonal + diurnal temperature cycles, elevation lapse rate
- **Realistic Physiology** — Hydration, energy, core temperature, fatigue, injury, infection
- **Hunter NPCs** — Patrol/chase AI with hidden detection radii
- **Trophy Objective** — Hidden goal with graduated proximity hints
- **Fog of War** — Limited visibility with three-state cells
- **Difficulty Scaling** — Global multipliers for drain, hazards, wildlife, temperature
- **7 Agent Types** — Heuristic, LLM (Gemini/OpenAI), Random, PSO, GWO, WOA, ACO
- **Swarm Visualization** — Pulsing clone copies with cyan search radius circles
- **Layered Renderer** — Pygame with 7 overlay modes, rich HUD, video recording
- **Deterministic Replay** — Seeded RNG for perfect reproducibility

---

## Installation

### Prerequisites
- Python >= 3.11
- A Gemini API key (primary) and/or OpenAI API key (fallback) for LLM agents

### Setup

```bash
git clone https://github.com/ritwikraha/earth-33.git
cd earth-33

pip install -e .

# Install with all extras
pip install -e ".[dev,recording,llm-openai]"

# Configure API keys (for LLM agent only)
cp .env.example .env
# Edit .env with your API keys
```

---

## Quick Start

```bash
# Heuristic agent — forest biome
python -m cli run_episode --config configs/forest.yaml --agent heuristic --seed 42

# Hunt mode — 6 hunters, trophy, fog of war
python -m cli run_episode --config configs/hunt.yaml --agent heuristic --seed 42

# LLM agent
python -m cli run_episode --config configs/hunt.yaml --agent llm --seed 42

# Swarm agents
python -m cli run_episode --config configs/hunt.yaml --agent pso --seed 42
python -m cli run_episode --config configs/hunt.yaml --agent gwo --seed 42

# Headless (no window)
python -m cli run_episode --config configs/hunt.yaml --agent heuristic --headless --seed 42

# Batch evaluation
python -m cli evaluate --config configs/hunt.yaml --agent heuristic --seeds 0-9 --output runs/eval

# Record video
python -m cli run_episode --config configs/hunt.yaml --agent pso --record runs/demo.mp4

# Replay a saved episode
python -m cli replay runs/20260222_001132_seed42.json
```

---

## Project Structure

```
earth_33/
├── configs/              # Scenario YAML configs (forest, desert, tundra, hunt)
├── sim/                  # Simulation engine (world, climate, physiology, hunters, trophy)
├── agents/               # Agent implementations (LLM, heuristic, random, swarm)
├── eval/                 # Batch evaluation framework
├── render/               # Pygame renderer + color palettes
├── config_io/            # Pydantic config schemas + loading
├── cli/                  # Typer CLI (run_episode, evaluate, replay)
├── data/                 # Data ingestion (optional GIS)
├── tests/                # Test suite (36 tests)
├── docs/                 # Detailed documentation
├── runs/                 # Episode outputs (gitignored)
├── pyproject.toml        # Project metadata & dependencies
├── CONTRIBUTING.md       # Contributing guide & test info
└── README.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Game loop, simulation systems, and swarm agent pipeline diagrams |
| [Game Mechanics](docs/game-mechanics.md) | Physiology, actions, terrain, hunters, trophy, fog of war, hunt mode |
| [Configuration](docs/configuration.md) | YAML reference, preset configs, custom config examples |
| [Agents](docs/agents.md) | All 7 agent types: LLM, heuristic, random, PSO, GWO, WOA, ACO |
| [CLI Reference](docs/cli-reference.md) | Full command-line interface with all flags and examples |
| [Contributing](CONTRIBUTING.md) | Setup, workflow, areas for contribution, testing |

---

## Citation

If you use Earth33 Survival in your research, please cite:

```bibtex
@software{earth33_survival_2025,
  title     = {Earth33 Survival: LLM-as-Organism in Realistic 2D Earth Environments},
  author    = {Ritwik Raha},
  year      = {2025},
  url       = {https://github.com/ritwikraha/earth-33},
  note      = {A simulation framework for evaluating LLM autonomous survival
               and planning capabilities in procedurally generated environments
               with adversarial NPC hunters, fog of war, bio-inspired swarm
               optimization agents, and objective-driven trophy pursuit.}
}
```

## License

This project is open source. See [LICENSE](LICENSE) for details.
