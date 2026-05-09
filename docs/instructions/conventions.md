# Conventions

- **Localization**: backend uses `utils/locale.t(...)` / `get_locale` / `set_locale` for error/status strings; don't hardcode user-visible Chinese/English in API responses. Frontend uses `vue-i18n` — add keys to `locales/{en,zh}.json` and `frontend/src/i18n/` accordingly.
- **Logging**: `utils/logger.get_logger('mirofish.<area>')`. Don't `print`.
- **Long-running work**: create a `Task` via `TaskManager`, run in a background thread, expose a status endpoint. Follow patterns in `api/graph.py` and `api/report.py`.
- **Model IDs**: must be `qwen/...` in code/config. Different provider = policy change — don't bypass `_require_qwen` silently.
- **Secrets**: never appear in code, commits, MCP env blocks, or log lines. Reference by file+line.
