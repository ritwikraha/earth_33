# Architecture

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
```

## Simulation Systems

```
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

## Swarm Agent Pipeline

```
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
