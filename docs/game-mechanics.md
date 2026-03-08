# Game Mechanics

## Organism Physiology

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

## Available Actions (10 total)

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

## Terrain Types (7 biomes)

| Terrain | Movement Cost | Water | Food | Temperature | Notes |
|---------|:------------:|:-----:|:----:|:-----------:|-------|
| Plains | 1.0x | Low | Moderate | Moderate | Fast, open ground |
| Forest | 1.5x | Moderate | High | Mild | Good shelter + food |
| Desert | 1.0x | Very Low | Very Low | Extreme heat | Fast but harsh |
| Tundra | 1.8x | Low | Low | Extreme cold | Sparse resources |
| Mountain | 3.0x | Low | Low | Cold (lapse rate) | Very slow |
| Water | Impassable | N/A | N/A | Moderate | Can DRINK at edge |
| Swamp | 2.5x | High | Low | Warm | Disease risk |

## Hunter NPCs

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

## Trophy Objective

- **Hidden Goal** — A trophy is placed 22-30 cells from spawn (varies by config); find it to win
- **Graduated Hints** — Distance buckets (very close / close / moderate / far), warmer/colder feedback, directional hints every N steps
- **Risk-Reward Tradeoff** — Pursuing the trophy means venturing into unknown, potentially hunter-infested territory

## Fog of War

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

## Difficulty Scaling

Global multipliers that scale the challenge:

| Multiplier | What It Affects |
|-----------|-----------------|
| `drain_multiplier` | Hydration/energy loss rate |
| `hazard_multiplier` | Environmental stress |
| `wildlife_multiplier` | Wildlife encounter probability |
| `temperature_extremity` | Seasonal/diurnal temperature amplitude |

## The Hunt

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
