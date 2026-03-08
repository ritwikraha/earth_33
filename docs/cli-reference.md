# CLI Reference

## `run_episode` — Run a Single Episode

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

---

## `evaluate` — Batch Evaluation Across Seeds

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

---

## `replay` — Replay a Saved Episode

```bash
python -m cli replay <replay_file.json>
```

Replays a previously saved episode JSON file in the pygame window.

---

## Keyboard Controls (Visual Mode)

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
