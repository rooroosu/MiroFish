# Architecture

## Backend (`backend/app/`)

Flask app factory in `app/__init__.py` registers three blueprints under `/api/`:

- `graph_bp` (`/api/graph`) — `api/graph.py` — project CRUD, file upload, ontology generation, Zep/local graph build. Uses `services/ontology_generator.py`, `services/graph_builder.py`, `services/text_processor.py`, `utils/file_parser.py` (PDF/MD/TXT).
- `simulation_bp` (`/api/simulation`) — `api/simulation.py` — entity filtering, persona generation, OASIS config generation, start/stop/monitor, agent interview. Uses `services/zep_entity_reader.py`, `services/oasis_profile_generator.py`, `services/simulation_config_generator.py`, `services/simulation_manager.py`, `services/simulation_runner.py`.
- `report_bp` (`/api/report`) — `api/report.py` — ReACT report generation with tool-calling agent. Uses `services/report_agent.py` + `services/zep_tools.py`.

### State

- `ProjectManager` / `TaskManager` (`app/models/`) — in-memory project + async task tracking (tasks power long-running graph build / simulation / report jobs; clients poll status endpoints).
- Uploads and simulation artifacts live under `backend/app/uploads/` (configured in `Config.UPLOAD_FOLDER`, `OASIS_SIMULATION_DATA_DIR`).

### Memory backend switch (`Config.is_local_memory_backend()`)

- `MIROFISH_MEMORY_BACKEND=local_qmd` (default) → `services/local_qmd_backend.LocalQMDGraphStore` (filesystem JSON under `uploads/local_graphs/`).
- Else → Zep Cloud (`zep_cloud` SDK, `zep-cloud==3.13.0`). Use `utils/zep_paging.fetch_all_nodes/edges` for full pagination; don't hand-roll cursors.

### Simulation IPC

`services/simulation_ipc.py`: Flask and the preset simulation scripts are separate processes. They communicate via a filesystem command/response directory pattern (`CommandType.INTERVIEW`, `BATCH_INTERVIEW`, `CLOSE_ENV`). `SimulationRunner` spawns `scripts/run_parallel_simulation.py` and drives it through `SimulationIPCClient`. Cleanup registered via `atexit` (`SimulationRunner.register_cleanup()` in `create_app`) so Flask shutdown also kills sim subprocesses.

### LLM access

Always go through `utils/llm_client.LLMClient` and `model_routing.resolve_model_route()` — not direct `openai` imports — so Qwen-only enforcement, fallbacks, and OpenAI-compat env exposure (`apply_openai_compatible_env`, used by CAMEL/OASIS) stay consistent.

### Windows

`run.py` reconfigures stdio to UTF-8 before importing app. Keep console output UTF-8-safe.

## Frontend (`frontend/src/`)

Vue 3 SFCs. Router (`router/index.js`) maps the five pipeline stages to views: `Home` → `MainView` (Process) → `SimulationView` / `SimulationRunView` → `ReportView` → `InteractionView`. Step components (`components/Step1GraphBuild.vue` … `Step5Interaction.vue`) are reused inside `MainView`. API wrappers in `src/api/{graph,simulation,report}.js` hit the Flask blueprints. i18n via `vue-i18n` (`src/i18n/index.js`, locale files in `frontend/src/i18n/` and top-level `locales/`). Graph visualisation uses d3 (`components/GraphPanel.vue`). Dev server is Vite on port 3000 with `--host`.

## Five-stage pipeline (matches UI steps)

1. **Graph build** — upload seed docs, chunk, extract ontology, build graph (Zep or local_qmd).
2. **Env setup** — read entities from graph, filter by type, generate OASIS agent personas + simulation config params.
3. **Simulation** — spawn parallel Twitter/Reddit OASIS runs as subprocess, stream status, allow mid-run interviews.
4. **Report** — `ReportAgent` plans TOC, generates sections with ReACT loop (search / insight-forge / panorama / interview tools), writes `agent_log.jsonl` alongside the report.
5. **Interaction** — chat with individual sim agents or with the ReportAgent post-run.
