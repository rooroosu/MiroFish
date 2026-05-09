# CLAUDE.md

MiroFish — swarm-intelligence prediction engine. Fork of BettaFish using OASIS (camel-ai). Five-stage pipeline: Graph Build → Env Setup → Simulation → Report → Interaction. Flask backend (uv, Python 3.11–3.12) + Vue 3/Vite frontend. AGPL-3.0.

## Must-know

- **LLM model ids must be `qwen/...`** — `_require_qwen` in `app/model_routing.py` raises otherwise. Always route via `utils/llm_client.LLMClient` + `model_routing.resolve_model_route()`, not direct `openai` imports.
- **Memory backend**: default `MIROFISH_MEMORY_BACKEND=local_qmd` (filesystem under `backend/app/uploads/local_graphs/`). Set to anything else → Zep Cloud; `ZEP_API_KEY` then required. Use `utils/zep_paging.fetch_all_nodes/edges` for pagination.
- **Secrets**: never echo values, never commit, never in MCP env blocks. Reference by `path:line`.
- **Long-running work**: `TaskManager` + background thread + status endpoint. See `api/graph.py`, `api/report.py`.
- **Logging**: `utils/logger.get_logger('mirofish.<area>')`. No `print`.
- **i18n**: backend `utils/locale.t(...)`; frontend `vue-i18n` with keys in `locales/{en,zh}.json` + `frontend/src/i18n/`.
- **Korean → English on ingest**: `FileParser` auto-translates `.md`/`.txt`/`.pdf` whose Hangul ratio ≥ `MIROFISH_TRANSLATION_KO_THRESHOLD` (default `0.05`) via Qwen (`MIROFISH_TRANSLATION_MODEL`, default `qwen/qwen-turbo`). Cached as `<name>.en.<ext>` next to source, keyed by SHA-256 of source content. Disable via `MIROFISH_AUTO_TRANSLATE=false`. Sub-agents read English (~3× fewer tokens than Hangul under BPE). Pre-warm: `uv run python backend/scripts/translate_korean_docs.py <path>`.
- **Stock scenarios layout**: `backend/uploads/stock_scenarios/<TICKER>/inputs/<scenario>/` (source docs + `catalyst.txt`) and `<TICKER>/results/<scenario>/` (run snapshots). Scripts use `scripts/_scenario_paths.py`; legacy flat `<TICKER>_<scenario>/` still resolves.

## References

Load only when relevant:

- [docs/instructions/commands.md](docs/instructions/commands.md) — dev/build/test/docker/sim scripts.
- [docs/instructions/environment.md](docs/instructions/environment.md) — `.env` keys (required + optional).
- [docs/instructions/architecture.md](docs/instructions/architecture.md) — blueprints, services, IPC, pipeline stages.
- [docs/instructions/conventions.md](docs/instructions/conventions.md) — full conventions detail.
