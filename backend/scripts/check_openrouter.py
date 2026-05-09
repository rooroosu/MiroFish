"""Phase 0 smoke check for MiroFish → OpenRouter Qwen routing.

Lists the current Qwen model catalog on OpenRouter, then exercises the primary
and review models via LLMClient with a trivial JSON prompt.

Throwaway utility — delete once Phase 0 passes.
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config  # noqa: E402  (imports after sys.path mutation)
from app.utils.llm_client import LLMClient  # noqa: E402


def list_qwen_catalog(api_key: str) -> list[str]:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = [m["id"] for m in payload.get("data", []) if m.get("id", "").startswith("qwen/")]
    return sorted(ids)


def smoke_model(label: str, model: str) -> dict:
    client = LLMClient(model=model)
    result = client.chat_json(
        messages=[
            {"role": "system",
             "content": "Return json only. Output exactly this json object and nothing else: {\"ok\": true}"},
            {"role": "user", "content": "Return the json object now."},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    return {"label": label, "model": model, "result": result}


def main() -> int:
    errors = Config.validate()
    if errors:
        print("Config errors:", errors, file=sys.stderr)
        print("Paste real OPENROUTER key into MiroFish/.env "
              "(LLM_API_KEY and LLM_BOOST_API_KEY lines) and re-run.",
              file=sys.stderr)
        return 2

    api_key = Config.LLM_API_KEY or ""

    print("== OpenRouter Qwen catalog (current) ==")
    try:
        catalog = list_qwen_catalog(api_key)
    except Exception as exc:
        print(f"catalog fetch failed: {exc}", file=sys.stderr)
        return 3
    for mid in catalog:
        print(f"  {mid}")
    print(f"  ({len(catalog)} qwen/* models)")

    pinned = {
        "LLM_MODEL_NAME (primary)": Config.LLM_MODEL_NAME,
        "LLM_BOOST_MODEL_NAME (boost)": Config.LLM_BOOST_MODEL_NAME,
        "MIROFISH_REVIEW_MODEL (review)": Config.MIROFISH_REVIEW_MODEL,
    }
    print("\n== Pinned models in .env ==")
    for label, mid in pinned.items():
        status = "OK" if mid in catalog else "MISSING from catalog"
        print(f"  {label}: {mid}  [{status}]")

    missing = [mid for label, mid in pinned.items() if mid not in catalog]
    if missing:
        print("\nRepin .env to IDs from the catalog above, then re-run.", file=sys.stderr)
        return 4

    print("\n== Live JSON smoke ==")
    failures = []
    for label, model in [
        ("primary", Config.LLM_MODEL_NAME),
        ("review",  Config.MIROFISH_REVIEW_MODEL),
    ]:
        try:
            out = smoke_model(label, model)
            print(f"  {label} ({model}) -> {out['result']}")
            if out["result"] != {"ok": True}:
                failures.append((label, model, f"unexpected payload: {out['result']}"))
        except Exception as exc:
            failures.append((label, model, str(exc)))
            print(f"  {label} ({model}) FAILED: {exc}", file=sys.stderr)

    if failures:
        print("\nOne or more live checks failed. Do not proceed past Phase 0.", file=sys.stderr)
        return 5

    print("\nPhase 0 OK. Safe to proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
