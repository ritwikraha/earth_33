# Contributing

Contributions are welcome! Here's how to get started.

## Getting Set Up

```bash
# Fork and clone the repo
git clone https://github.com/<your-username>/earth-33.git
cd earth-33

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run the test suite to verify your setup
python -m pytest tests/ -v
```

## Development Workflow

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

## Areas for Contribution

- **New Agent Strategies** — Implement new agent types (RL, tree search, multi-agent)
- **Swarm Algorithms** — Add more bio-inspired algorithms (firefly, cuckoo search, harmony search)
- **Biome Expansion** — Add new terrain types (volcanic, coastal, urban)
- **Hunter AI** — Smarter pathfinding (A*), coordinated pack behavior
- **Visualization** — Additional overlays, minimap, agent thought bubbles
- **Real GIS Data** — Integration with elevation/climate datasets via `data/`
- **Benchmarking** — Compare LLM providers, prompt engineering experiments
- **Multi-Agent** — Multiple organisms cooperating or competing on the same map

## Guidelines

1. **Tests required** — All PRs must include tests for new functionality
2. **Backward compatible** — New config fields must have defaults (see `enabled: bool = False` pattern)
3. **Deterministic** — All randomness must flow through the seeded `np.random.default_rng`
4. **Type hints** — Use type annotations for all function signatures
5. **Docstrings** — Document classes and public methods

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
