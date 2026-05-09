# Environment

Root `.env` loaded by `backend/app/config.py` via python-dotenv. Copy from `.env.example`. Never echo values.

## Required

- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` — OpenAI-format LLM. Defaults point at OpenRouter Qwen in `app/model_routing.py`. **Only `qwen/...` model ids accepted** — `_require_qwen` raises `ValueError` otherwise.
- `ZEP_API_KEY` — required unless `MIROFISH_MEMORY_BACKEND=local_qmd` (default). Local backend writes graphs to `backend/app/uploads/local_graphs/`.

## Optional

- Boost/review routing: `LLM_BOOST_*`, `MIROFISH_REVIEW_MODEL`, `MIROFISH_FALLBACK_MODELS` (csv of qwen ids).
- Tuning: `OASIS_DEFAULT_MAX_ROUNDS`, `REPORT_AGENT_MAX_TOOL_CALLS`, `REPORT_AGENT_MAX_REFLECTION_ROUNDS`, `REPORT_AGENT_TEMPERATURE`.
- Flask: `FLASK_HOST`, `FLASK_PORT` (default 0.0.0.0:5001), `FLASK_DEBUG`.
- Translation (Korean → English on ingest, default on):
  - `MIROFISH_AUTO_TRANSLATE` (default `true`) — disable to read raw Korean.
  - `MIROFISH_TRANSLATION_MODEL` (default `qwen/qwen-turbo`) — must be a `qwen/...` id; `qwen-turbo` is the cheapest production option on OpenRouter at the time of writing.
  - `MIROFISH_TRANSLATION_KO_THRESHOLD` (default `0.05`) — minimum Hangul-character ratio that triggers translation. English-only docs pass through.
  - Cache: `<name>.en.<ext>` written next to source, keyed by SHA-256 of source content (auto-regenerated when source changes).
