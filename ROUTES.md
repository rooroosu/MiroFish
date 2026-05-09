# MiroFish — Local Access & Routes

## Starting the App

```bash
# Terminal 1 — Backend (Flask, port 5001)
cd backend
uv run python run.py

# Terminal 2 — Frontend (Vite, port 3000)
cd frontend
npm run dev
```

Open: **http://localhost:3000**

---

## Frontend Routes (localhost:3000)

| URL | Page |
|-----|------|
| `http://localhost:3000/` | Home — create or load a project |
| `http://localhost:3000/process/<projectId>` | 5-stage pipeline (Graph → Env → Sim → Report → Interact) |
| `http://localhost:3000/simulation/<simulationId>` | Simulation config |
| `http://localhost:3000/simulation/<simulationId>/start` | Running simulation (live output) |
| `http://localhost:3000/report/<reportId>` | Report view |
| `http://localhost:3000/interaction/<reportId>` | Chat with report (Step 5) |

> **Note:** Replace `<projectId>` etc. with the actual ID — no colon.
> Example: `/process/sim_585b8bc7d6e1` not `/process/:sim_585b8bc7d6e1`

---

## Backend API Routes (localhost:5001)

All routes proxied through Vite at `/api/*` → `localhost:5001/api/*`

### Health
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Server health check |

### Graph (`/api/graph`)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/graph/project/list` | List all projects |
| GET | `/api/graph/project/<project_id>` | Get project details |
| DELETE | `/api/graph/project/<project_id>` | Delete project |
| POST | `/api/graph/project/<project_id>/reset` | Reset project state |
| POST | `/api/graph/ontology/generate` | Generate ontology from topic |
| POST | `/api/graph/build` | Build graph (async — returns task_id) |
| GET | `/api/graph/task/<task_id>` | Poll build task status |
| GET | `/api/graph/tasks` | List all tasks |
| GET | `/api/graph/data/<graph_id>` | Get graph node/edge data |
| DELETE | `/api/graph/delete/<graph_id>` | Delete a graph |

### Simulation (`/api/simulation`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/simulation/create` | Create simulation config |
| GET | `/api/simulation/<sim_id>` | Get simulation |
| POST | `/api/simulation/<sim_id>/run` | Start simulation (async) |
| GET | `/api/simulation/<sim_id>/status` | Poll simulation status |

### Report (`/api/report`)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/report/generate` | Generate report (async) |
| POST | `/api/report/generate/status` | Poll generation status |
| GET | `/api/report/<report_id>` | Get report |
| GET | `/api/report/by-simulation/<sim_id>` | Get report by simulation |
| GET | `/api/report/list` | List all reports |
| GET | `/api/report/<report_id>/download` | Download report file |
| DELETE | `/api/report/<report_id>` | Delete report |
| POST | `/api/report/chat` | Chat with report (Step 5) |
| GET | `/api/report/<report_id>/progress` | Poll report generation progress |
| GET | `/api/report/<report_id>/sections` | Get report sections |
| GET | `/api/report/<report_id>/section/<index>` | Get single section |
| GET | `/api/report/check/<sim_id>` | Check if report exists for simulation |
| GET | `/api/report/<report_id>/agent-log` | Get agent action log |
| GET | `/api/report/<report_id>/agent-log/stream` | Stream agent log (SSE) |
| GET | `/api/report/<report_id>/console-log` | Get console log |

---

## Pipeline Flow

```
Step 1: Graph Build    →  POST /api/graph/build          (poll /api/graph/task/<id>)
Step 2: Env Setup      →  (frontend config, no API call)
Step 3: Simulation     →  POST /api/simulation/<id>/run  (poll /status)
Step 4: Report         →  POST /api/report/generate      (poll /progress)
Step 5: Interaction    →  POST /api/report/chat
```
