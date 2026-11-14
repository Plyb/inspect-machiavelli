# inspect-machiavelli

The [MACHIAVELLI benchmark](https://github.com/aypan17/machiavelli) implemented as an [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evaluation.

MACHIAVELLI measures how a model behaves while pursuing goals in 30 narrative, choice-driven text games. As the agent plays, the eval tracks not just achievement completion but the ethical and power-seeking cost of getting there (harm, deception, resource accumulation, etc.).

## How it works

`machiavelli/tasks.py` defines an Inspect `@task` that, per game:

1. **`game_data.py`** — downloads and caches the game data from HuggingFace (`kobyjl/machiavelli_game_data`) into `./outputs/game_data/` on first run.
2. **`solver.py`** — instantiates a `MachiavelliEnv` and steps an LM-driven `InspectAgent` through it for up to `max_traj_length` actions, recording the trajectory.
3. **`scorer.py`** — scores the trajectory against the MACHIAVELLI harm/power metrics, normalized by the benchmark's coefficients and the configured thresholds.

## Setup

Requires Python ≥3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                        # core eval dependencies
uv sync --extra hf-models      # add transformers + accelerate to run HF models locally
```

## Running

```bash
uv run inspect eval src/machiavelli/tasks.py \
  --model hf/Qwen/Qwen2.5-3B \
  -M dtype=float16 \
  --limit 5 \
  --task-config config.yaml
```

Any Inspect-supported model provider works via `--model`; `hf/<repo>` runs a HuggingFace model locally.

## Layout

```
src/
  machiavelli/
    tasks.py                 # Inspect @task entry point + game list
    dataset.py               # games -> Inspect samples
    game_data.py             # fetch/cache game data from HuggingFace
    solver.py                # game loop driving the agent through the env
    scorer.py                # MACHIAVELLI harm/power scoring
    agent/                   # agent protocol + LM-based InspectAgent
    game/                    # game env, player, types (adapted from original)
```

## Licensing

Files under `src/machiavelli/game/` and a few items are derived from the original MACHIAVELLI benchmark (MIT). See `THIRD_PARTY_LICENSES`.
