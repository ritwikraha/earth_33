# Agents

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

---

## LLM Agent (`--agent llm`)

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

---

## Heuristic Agent (`--agent heuristic`)

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

---

## Random Agent (`--agent random`)

Uniform random action selection from the valid action mask. Serves as a baseline for evaluation.

---

## Swarm Optimization Agents

All four swarm agents share a common base (`SwarmAgentBase`) that provides:
- **World data caching** — Numpy arrays for terrain cost, water, vegetation, wildlife risk, elevation
- **Hunter memory** — Tracks visible hunters, forgets after 30 steps of no sighting
- **Trophy estimation** — Blends directional hints and warm/cold feedback into a direction vector
- **Fitness function** — Evaluates candidate positions: trophy attraction + hunter avoidance + resource urgency + terrain cost
- **Survival override** — Delegates to the heuristic agent when vitals are critical (hydration < 20, energy < 20, fatigue > 85, injury > 70)
- **Clone rendering** — Each clone is drawn as a faded pulsing copy with a cyan search radius circle

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

### PSO — Particle Swarm Optimization (`--agent pso`)

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

### GWO — Grey Wolf Optimization (`--agent gwo`)

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

### WOA — Whale Optimization Algorithm (`--agent woa`)

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

### ACO — Ant Colony Optimization (`--agent aco`)

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
