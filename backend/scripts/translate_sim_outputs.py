"""Translate Chinese agent posts/interviews/comments from a simulation snapshot
into English markdown for human reading.

Usage:
    uv run python backend/scripts/translate_sim_outputs.py \
        backend/uploads/stock_scenarios/NU/results/bear_miss \
        backend/uploads/stock_scenarios/NU/results/bull_beat \
        backend/uploads/stock_scenarios/NU/results/no_catalyst

Per scenario dir, reads twitter_simulation.db trace table (action in
{create_post, interview, comment_post, quote_post, repost}), filters out
catalyst rows (per Bug 1's catalyst_markers.json), batch-translates Chinese
content via Qwen, and writes:
    <scenario_dir>/agent_interactions_en.md

Always routes through utils.llm_client.LLMClient + model_routing.resolve_model_route()
— never bypass _require_qwen.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config  # noqa: E402
from app.utils.llm_client import LLMClient  # noqa: E402

# Reuse aggregator's catalyst-marker logic for consistency.
from app.services.stock_sentiment_aggregator import (  # noqa: E402
    _is_catalyst,
    _load_catalyst_markers,
)


CJK_RANGE = "　-〿一-鿿＀-￯"
import re

CJK_RE = re.compile(f"[{CJK_RANGE}]")


def has_chinese(s: str) -> bool:
    return bool(s and CJK_RE.search(s))


def db_path_for(scenario_dir: str) -> str:
    for fname in ("twitter_simulation.db", "reddit_simulation.db"):
        p = os.path.join(scenario_dir, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no DB under {scenario_dir}")


def agent_names(config_path: str) -> Dict[int, Tuple[str, float]]:
    """Return {agent_id: (entity_name, sentiment_bias)} from simulation_config.json."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out: Dict[int, Tuple[str, float]] = {}
    for ac in cfg.get("agent_configs", []):
        aid = ac.get("agent_id")
        if aid is None:
            continue
        out[int(aid)] = (
            ac.get("entity_name", f"Agent_{aid}"),
            float(ac.get("sentiment_bias", 0.0)),
        )
    return out


ACTIONS_TO_INCLUDE = ("create_post", "interview", "comment_post", "quote_post", "repost")


def load_actions(db_path: str, sim_dir: str) -> List[Dict]:
    """Returns list of dicts: {rowid, action, user_id, text, created_at, is_catalyst}."""
    markers = _load_catalyst_markers(sim_dir)
    out: List[Dict] = []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT rowid, action, user_id, info, created_at FROM trace "
            f"WHERE action IN ({','.join('?' * len(ACTIONS_TO_INCLUDE))}) "
            f"ORDER BY created_at, rowid",
            ACTIONS_TO_INCLUDE,
        ).fetchall()
    finally:
        conn.close()
    for rowid, action, user_id, info, created_at in rows:
        try:
            data = json.loads(info) if info else {}
        except json.JSONDecodeError:
            continue
        text = ""
        if action in ("create_post", "quote_post"):
            text = (data.get("content") or "").strip()
        elif action == "interview":
            resp = data.get("response")
            if isinstance(resp, (dict, list)):
                resp = json.dumps(resp, ensure_ascii=False)
            text = (resp or "").strip()
        elif action == "comment_post":
            text = (data.get("content") or data.get("comment") or "").strip()
        elif action == "repost":
            text = (data.get("content") or "").strip()
            if not text:
                continue  # bare repost, no original text — skip
        if not text:
            continue
        is_cat = _is_catalyst(text, markers) if action == "create_post" else False
        out.append({
            "rowid": rowid,
            "action": action,
            "user_id": int(user_id),
            "text": text,
            "created_at": created_at,
            "is_catalyst": is_cat,
        })
    return out


def translate_batch(client: LLMClient, batch: List[str]) -> List[str]:
    """Translate a list of Chinese strings to English. Preserves indexing."""
    if not batch:
        return []
    numbered = "\n\n".join(f"[[{i+1}]]\n{t}" for i, t in enumerate(batch))
    system = (
        "You are a precise translator. Translate each numbered input from Chinese "
        "(or any non-English source language) to clear, natural English. Preserve "
        "the original meaning, tone, list structure, line breaks, and any technical "
        "terms (ticker symbols, financial terms, proper nouns). If a passage is "
        "already in English, return it unchanged. Output a JSON object: "
        '{"translations": ["...", "...", ...]} with exactly one English string per '
        "input, in the same order. No prose outside the JSON, no code fence."
    )
    user = f"Translate these {len(batch)} passages:\n\n{numbered}"
    try:
        # Use chat() and parse JSON manually so we can request response_format JSON
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=min(4096, 256 + 600 * len(batch)),
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
        out = data.get("translations") or []
        if len(out) != len(batch):
            # Pad / truncate defensively.
            while len(out) < len(batch):
                out.append(batch[len(out)])  # fall back to original
            out = out[: len(batch)]
        return [str(s) for s in out]
    except Exception as exc:
        print(f"  [warn] translate batch failed ({exc}); falling back to originals", flush=True)
        return list(batch)


def translate_all(client: LLMClient, texts: List[str], batch_size: int = 6,
                  max_workers: int = 4) -> List[str]:
    """Translate texts preserving order. Splits into batches; runs batches in parallel."""
    if not texts:
        return []
    # Only translate ones that contain CJK; pass through pure-English unchanged.
    needs_idx = [i for i, t in enumerate(texts) if has_chinese(t)]
    out = list(texts)
    if not needs_idx:
        return out

    def chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    batches = list(chunk(needs_idx, batch_size))
    print(f"  translating {len(needs_idx)} of {len(texts)} passages in {len(batches)} batch(es)...", flush=True)

    def run(batch_idxs):
        batch_texts = [texts[i] for i in batch_idxs]
        translated = translate_batch(client, batch_texts)
        return list(zip(batch_idxs, translated))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run, b) for b in batches]
        for fut in as_completed(futures):
            for idx, t in fut.result():
                out[idx] = t
    return out


def write_markdown(scenario_dir: str, scenario_label: str, ticker: str,
                   actions: List[Dict], translations: List[str],
                   names: Dict[int, Tuple[str, float]]) -> str:
    out_path = os.path.join(scenario_dir, "agent_interactions_en.md")
    by_agent: Dict[int, List[Tuple[Dict, str]]] = {}
    catalyst_block: List[Tuple[Dict, str]] = []
    for act, trans in zip(actions, translations):
        if act.get("is_catalyst"):
            catalyst_block.append((act, trans))
            continue
        by_agent.setdefault(act["user_id"], []).append((act, trans))

    def bias_label(b: float) -> str:
        # Match stock_sentiment_aggregator.STANCE_THRESHOLD = 0.2
        if b > 0.2:
            return f"bullish-bias ({b:+.2f})"
        if b < -0.2:
            return f"bearish-bias ({b:+.2f})"
        return f"neutral ({b:+.2f})"

    lines: List[str] = []
    lines.append(f"# Agent Interactions — {ticker} / {scenario_label}\n")
    lines.append(f"_Translated from source language to English. Source DB: `{os.path.basename(db_path_for(scenario_dir))}`._\n")

    if catalyst_block:
        lines.append("## Injected Catalyst (excluded from sentiment count)\n")
        for act, trans in catalyst_block:
            lines.append(f"- **{act['action']}** at t={act['created_at']}: {trans}")
            if has_chinese(act["text"]):
                lines.append(f"  <details><summary>original</summary>\n\n  {act['text']}\n  </details>")
        lines.append("")

    lines.append("## Per-Agent Timeline\n")
    for aid in sorted(by_agent):
        name, bias = names.get(aid, (f"Agent_{aid}", 0.0))
        n_post = sum(1 for a, _ in by_agent[aid] if a["action"] in ("create_post", "quote_post"))
        n_iv = sum(1 for a, _ in by_agent[aid] if a["action"] == "interview")
        n_com = sum(1 for a, _ in by_agent[aid] if a["action"] in ("comment_post", "repost"))
        lines.append(f"### Agent {aid} — {name}  _(bias: {bias_label(bias)}; posts={n_post}, interviews={n_iv}, comments/reposts={n_com})_\n")
        for act, trans in by_agent[aid]:
            tag = act["action"].replace("_", " ")
            lines.append(f"**[{tag} @ t={act['created_at']}]** {trans}")
            if has_chinese(act["text"]) and act["text"] != trans:
                # collapsible original
                lines.append(f"<details><summary>original</summary>\n\n```\n{act['text']}\n```\n</details>")
            lines.append("")
        lines.append("---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def process_scenario(scenario_dir: str) -> Optional[str]:
    if not os.path.isdir(scenario_dir):
        print(f"[skip] not a directory: {scenario_dir}", flush=True)
        return None
    db = db_path_for(scenario_dir)
    cfg_path = os.path.join(scenario_dir, "simulation_config.json")
    names = agent_names(cfg_path)

    # Pull scenario_label / ticker either from config or from dir name
    label = "scenario"
    ticker = "?"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        label = cfg.get("scenario_label", os.path.basename(scenario_dir))
        ticker = cfg.get("stock_ticker", "?")
    else:
        base = os.path.basename(scenario_dir)
        if base.endswith("_results"):
            base = base[: -len("_results")]
        parts = base.split("_", 1)
        if len(parts) == 2:
            ticker, label = parts

    print(f"\n=== {ticker} / {label} ({scenario_dir}) ===", flush=True)
    actions = load_actions(db, scenario_dir)
    print(f"  {len(actions)} translatable actions ({sum(1 for a in actions if a['is_catalyst'])} catalyst-tagged)", flush=True)
    if not actions:
        return None

    client = LLMClient(model=Config.LLM_MODEL_NAME)
    texts = [a["text"] for a in actions]
    translations = translate_all(client, texts, batch_size=6, max_workers=4)
    out_path = write_markdown(scenario_dir, label, ticker, actions, translations, names)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dirs", nargs="+", help="One or more <ticker>_<scenario>_results dirs")
    args = ap.parse_args()

    written = []
    for d in args.scenario_dirs:
        try:
            p = process_scenario(d)
            if p:
                written.append(p)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[error] {d}: {exc}", flush=True)

    print("\n=== summary ===", flush=True)
    for p in written:
        print(f"  ✓ {p}", flush=True)


if __name__ == "__main__":
    main()
