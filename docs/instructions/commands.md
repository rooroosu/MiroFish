# Commands

Run from repo root unless noted. `uv` required for backend; Node ≥18 for frontend.

```bash
# Install (root + frontend + backend uv sync)
npm run setup:all

# Dev — backend (5001) + frontend (3000) concurrently
npm run dev
npm run backend           # cd backend && uv run python run.py
npm run frontend          # vite --host

# Frontend build
npm run build

# Docker (reads root .env, exposes 3000/5001)
docker compose up -d
```

## Backend tests

pytest is a dev dep; no suite wired up yet.

```bash
cd backend && uv run pytest
cd backend && uv run pytest tests/test_x.py::test_name -v
```

## Simulation preset scripts

Invoked by `SimulationRunner` as subprocesses — usually not run by hand.

```bash
cd backend && uv run python scripts/run_parallel_simulation.py --config <path>
cd backend && uv run python scripts/run_twitter_simulation.py  --config <path>
cd backend && uv run python scripts/run_reddit_simulation.py   --config <path>
# flags: --no-wait (exit after sim), --twitter-only / --reddit-only
```
