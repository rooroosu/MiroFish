"""Build human-readable interaction summaries from stock-scenario snapshots.

C1 — per-scenario `interactions_summary.md`: per-agent timeline (posts +
     interviews + comments + reposts), with sentiment_bias, sentiment label
     (from sentiment_report.json), and engagement counts.

C2 — cross-scenario `cross_scenario_summary.md`: same agent_id under each
     scenario side-by-side. Highlights stance-shift cases.

If `agent_interactions_en.md` exists in a scenario dir (from
translate_sim_outputs.py), the summary uses the English translations inline;
otherwise falls back to original text.

Pure SQL + JSON reads — no LLM calls.

Usage:
    uv run python backend/scripts/summarize_interactions.py \
        backend/uploads/stock_scenarios/NU/results/bear_miss \
        backend/uploads/stock_scenarios/NU/results/bull_beat \
        backend/uploads/stock_scenarios/NU/results/no_catalyst
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.stock_sentiment_aggregator import (  # noqa: E402
    _is_catalyst,
    _load_catalyst_markers,
    bucket_from_bias,
)


ACTION_ORDER = ("create_post", "quote_post", "interview", "comment_post", "repost", "like_post")
# Skip these in the human-readable timeline — they are mechanical (sign_up = persona blob;
# refresh = feed-poll noise; do_nothing = no-op).
TIMELINE_SKIP_ACTIONS = {"sign_up", "refresh", "do_nothing"}


def db_path_for(scenario_dir: str) -> str:
    for fname in ("twitter_simulation.db", "reddit_simulation.db"):
        p = os.path.join(scenario_dir, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no DB under {scenario_dir}")


def load_config(sim_dir: str) -> Dict:
    p = os.path.join(sim_dir, "simulation_config.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentiment_report(sim_dir: str) -> Dict:
    p = os.path.join(sim_dir, "sentiment_report.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_actions(db_path: str, sim_dir: str) -> List[Dict]:
    """Returns enriched action records from trace."""
    markers = _load_catalyst_markers(sim_dir)
    actions: List[Dict] = []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT rowid, action, user_id, info, created_at FROM trace ORDER BY created_at, rowid"
        ).fetchall()
    finally:
        conn.close()
    for rowid, action, user_id, info, created_at in rows:
        try:
            data = json.loads(info) if info else {}
        except json.JSONDecodeError:
            data = {}
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
        elif action == "like_post":
            text = ""
        else:
            text = json.dumps(data, ensure_ascii=False)[:200]
        is_cat = _is_catalyst(text, markers) if action == "create_post" else False
        actions.append({
            "rowid": rowid,
            "action": action,
            "user_id": int(user_id),
            "text": text,
            "created_at": created_at,
            "is_catalyst": is_cat,
            "post_id": data.get("post_id") or data.get("new_post_id"),
            "target_post_id": data.get("post_id") or data.get("like_id"),
        })
    return actions


def parse_translations(scenario_dir: str) -> Dict[Tuple[int, str], str]:
    """If agent_interactions_en.md exists, parse its per-action English text.

    Markdown produced by translate_sim_outputs.py format:
        ### Agent N — name ...
        **[create post @ t=4]** <english line 1>
        <english line 2>
        ...
        <english line K>
        <details><summary>original</summary>
        ```
        <chinese ...>
        ```
        </details>

    Returns dict keyed by (user_id, original_text) → english_text. Falls back to
    {} if file missing.
    """
    path = os.path.join(scenario_dir, "agent_interactions_en.md")
    if not os.path.exists(path):
        return {}
    out: Dict[Tuple[int, str], str] = {}
    current_uid: Optional[int] = None
    en_lines: List[str] = []
    in_details = False
    orig_lines: List[str] = []
    capturing_en = False

    def flush_pair():
        nonlocal en_lines, orig_lines
        if current_uid is None or not en_lines or not orig_lines:
            en_lines = []
            orig_lines = []
            return
        orig = "\n".join(orig_lines).strip()
        if orig.startswith("```"):
            orig = "\n".join(orig.splitlines()[1:])
        if orig.endswith("```"):
            orig = "\n".join(orig.splitlines()[:-1])
        orig = orig.strip()
        en = "\n".join(en_lines).strip()
        if orig and en:
            out[(current_uid, orig)] = en
        en_lines = []
        orig_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = re.match(r"### Agent (\d+)", line)
            if m:
                # Reset on agent boundary
                current_uid = int(m.group(1))
                en_lines = []
                orig_lines = []
                in_details = False
                capturing_en = False
                continue
            m2 = re.match(r"\*\*\[[^\]]+\]\*\*\s*(.*)", line)
            if m2:
                # New action begins — finalize any prior pair (if a details block hadn't closed it)
                en_lines = [m2.group(1)]
                orig_lines = []
                capturing_en = True
                in_details = False
                continue
            if line.startswith("<details>"):
                in_details = True
                capturing_en = False
                orig_lines = []
                continue
            if line.startswith("</details>"):
                in_details = False
                flush_pair()
                continue
            if in_details:
                orig_lines.append(line)
            elif capturing_en:
                # Stop capturing on horizontal rule (agent separator) just in case
                if line.strip() == "---":
                    capturing_en = False
                else:
                    en_lines.append(line)
    return out


def bias_label(b: float) -> str:
    if b > 0.2:
        return f"bullish ({b:+.2f})"
    if b < -0.2:
        return f"bearish ({b:+.2f})"
    return f"neutral ({b:+.2f})"


def per_post_label_from_report(report: Dict) -> Dict[int, str]:
    """sentiment_report.json doesn't include per-rowid labels (only top-K quotes).
    Approximate via top_bullish/top_bearish quotes — match content to label."""
    out: Dict[int, str] = {}
    # The report has top_bullish_quotes and top_bearish_quotes (list of strings).
    # We can't reliably map back to rowid without a join. Skip and let summary
    # show "—" for label per post; aggregate label is in interview_excerpts.
    return out


def parse_dir_label(scenario_dir: str) -> Tuple[str, str]:
    """Parse `<TICKER>_<label>_results` directory name into (ticker, label).
    Falls back to ('?', dirname) if it doesn't match."""
    base = os.path.basename(scenario_dir.rstrip("/"))
    if base.endswith("_results"):
        base = base[:-len("_results")]
    parts = base.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "?", base


def write_per_scenario_summary(scenario_dir: str) -> str:
    db = db_path_for(scenario_dir)
    cfg = load_config(scenario_dir)
    report = load_sentiment_report(scenario_dir)
    translations = parse_translations(scenario_dir)
    actions = load_actions(db, scenario_dir)

    # Prefer dir-name when scenarios share a config (3-way gate reuses sim_id, so
    # simulation_config.scenario_label only reflects the LAST scenario run, not THIS dir).
    dir_ticker, dir_label = parse_dir_label(scenario_dir)
    ticker = cfg.get("stock_ticker") or dir_ticker
    label = dir_label if dir_label else cfg.get("scenario_label", os.path.basename(scenario_dir))
    agent_configs = cfg.get("agent_configs", [])
    bias_by_id = {int(ac.get("agent_id", -1)): float(ac.get("sentiment_bias", 0.0)) for ac in agent_configs if ac.get("agent_id") is not None}
    name_by_id = {int(ac.get("agent_id", -1)): ac.get("entity_name", f"Agent_{ac.get('agent_id')}") for ac in agent_configs if ac.get("agent_id") is not None}
    archetype_by_id = {int(ac.get("agent_id", -1)): ac.get("synthetic_archetype", "") for ac in agent_configs if ac.get("agent_id") is not None}

    # Group by agent
    by_agent: Dict[int, List[Dict]] = defaultdict(list)
    catalyst_actions: List[Dict] = []
    like_counts: Dict[int, int] = defaultdict(int)  # by post_id (target)
    for a in actions:
        if a["action"] == "like_post" and a["target_post_id"] is not None:
            like_counts[int(a["target_post_id"])] += 1
        if a["is_catalyst"]:
            catalyst_actions.append(a)
            continue
        by_agent[a["user_id"]].append(a)

    # Build markdown
    lines: List[str] = []
    lines.append(f"# Interaction Summary — {ticker} / {label}\n")
    sd = report.get("sentiment_distribution", {})
    id_ = report.get("interview_distribution", {})
    pre = report.get("pre_bucket_distribution", {})
    post = report.get("post_bucket_distribution", {})
    lines.append("## Aggregate (from sentiment_report.json)\n")
    lines.append(f"- post_count: **{report.get('post_count', '?')}** | interview_count: **{report.get('interview_count', '?')}**")
    lines.append(f"- sentiment_distribution (posts): {sd}")
    lines.append(f"- interview_distribution: {id_}")
    lines.append(f"- pre_bucket: {pre} → post_bucket: {post}")
    if report.get("stance_shift"):
        lines.append(f"- stance_shift: {report['stance_shift']}")
    lines.append("")

    if catalyst_actions:
        lines.append("## Injected Catalyst (excluded from sentiment count)\n")
        for a in catalyst_actions:
            lines.append(f"- t={a['created_at']} agent={a['user_id']}: {a['text'][:300]}{'…' if len(a['text']) > 300 else ''}")
        lines.append("")

    lines.append("## Per-Agent Timeline\n")
    for aid in sorted(by_agent):
        bias = bias_by_id.get(aid, 0.0)
        name = name_by_id.get(aid, f"Agent_{aid}")
        archetype = archetype_by_id.get(aid, "")
        # Counters
        n_post = sum(1 for a in by_agent[aid] if a["action"] in ("create_post", "quote_post"))
        n_iv = sum(1 for a in by_agent[aid] if a["action"] == "interview")
        n_com = sum(1 for a in by_agent[aid] if a["action"] in ("comment_post", "repost"))
        n_like = sum(1 for a in by_agent[aid] if a["action"] == "like_post")
        archetype_tag = f" _[archetype: {archetype}]_" if archetype else ""
        lines.append(f"### Agent {aid} — {name}{archetype_tag}")
        lines.append(f"_bias: {bias_label(bias)} (pre-bucket: {bucket_from_bias(bias)}); posts={n_post} interviews={n_iv} comments+reposts={n_com} likes_given={n_like}_\n")
        for a in by_agent[aid]:
            if a["action"] == "like_post" or a["action"] in TIMELINE_SKIP_ACTIONS:
                continue  # too noisy for timeline
            tag = a["action"].replace("_", " ")
            text = a["text"]
            engagement = ""
            if a["post_id"] is not None and a["action"] in ("create_post", "quote_post"):
                lc = like_counts.get(int(a["post_id"]), 0)
                if lc:
                    engagement = f" _(likes={lc})_"
            # Translation lookup
            en = translations.get((aid, text), None)
            display_text = en if en else text
            lines.append(f"**[{tag} @ t={a['created_at']}]**{engagement} {display_text}")
            if en and en != text:
                lines.append(f"<details><summary>original</summary>\n\n```\n{text}\n```\n</details>")
            lines.append("")
        lines.append("---\n")

    out_path = os.path.join(scenario_dir, "interactions_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def write_cross_scenario(scenario_dirs: List[str], out_path: str) -> str:
    """C2 — same agent_id under each scenario side-by-side."""
    scenarios = []
    for d in scenario_dirs:
        cfg = load_config(d)
        report = load_sentiment_report(d)
        actions = load_actions(db_path_for(d), d)
        translations = parse_translations(d)
        agent_configs = cfg.get("agent_configs", [])
        bias_by_id = {int(ac.get("agent_id", -1)): float(ac.get("sentiment_bias", 0.0)) for ac in agent_configs if ac.get("agent_id") is not None}
        name_by_id = {int(ac.get("agent_id", -1)): ac.get("entity_name", f"Agent_{ac.get('agent_id')}") for ac in agent_configs if ac.get("agent_id") is not None}
        per_agent_iv: Dict[int, str] = {}  # latest interview text per agent
        for a in actions:
            if a["action"] == "interview" and not a["is_catalyst"]:
                en = translations.get((a["user_id"], a["text"]))
                per_agent_iv[a["user_id"]] = en if en else a["text"]
        dir_ticker, dir_label = parse_dir_label(d)
        scenarios.append({
            "dir": d,
            "label": dir_label,
            "ticker": cfg.get("stock_ticker") or dir_ticker,
            "report": report,
            "name_by_id": name_by_id,
            "bias_by_id": bias_by_id,
            "iv_by_id": per_agent_iv,
        })

    if not scenarios:
        return ""

    all_ids = sorted(set().union(*(set(s["bias_by_id"].keys()) for s in scenarios)))

    lines: List[str] = []
    lines.append("# Cross-Scenario Interaction Summary\n")
    lines.append("## Aggregate Comparison\n")
    lines.append("| scenario | post_count | iv_count | sentiment_dist (posts) | iv_dist | pre_bucket | post_bucket |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for s in scenarios:
        r = s["report"]
        lines.append(f"| **{s['label']}** | {r.get('post_count','?')} | {r.get('interview_count','?')} | {r.get('sentiment_distribution', {})} | {r.get('interview_distribution', {})} | {r.get('pre_bucket_distribution', {})} | {r.get('post_bucket_distribution', {})} |")
    lines.append("")

    # Bullish/bearish gap diagnostic
    def share(d, k):
        total = sum(d.values()) or 1
        return d.get(k, 0) / total * 100
    by_label = {s["label"]: s["report"].get("sentiment_distribution", {}) for s in scenarios}
    if "bear_miss" in by_label and "bull_beat" in by_label:
        bull_share_bull = share(by_label["bull_beat"], "bullish")
        bull_share_bear = share(by_label["bear_miss"], "bullish")
        bear_share_bear = share(by_label["bear_miss"], "bearish")
        bear_share_bull = share(by_label["bull_beat"], "bearish")
        lines.append("## Catalyst-Reactivity Gap\n")
        lines.append(f"- **Bullish-share gap (bull_beat − bear_miss): {bull_share_bull - bull_share_bear:+.1f}pp** (criterion: ≥ 25pp)")
        lines.append(f"- **Bearish-share gap (bear_miss − bull_beat): {bear_share_bear - bear_share_bull:+.1f}pp**")
        lines.append("")

    lines.append("## Per-Agent Stance Shift Across Scenarios\n")
    lines.append("Compares the latest interview answer per agent across scenarios. ⚠ = stance differs from pre-bucket.\n")
    for aid in all_ids:
        # Try to get a name from the first scenario that has it
        name = next((s["name_by_id"].get(aid, "") for s in scenarios if s["name_by_id"].get(aid)), f"Agent_{aid}")
        bias = next((s["bias_by_id"].get(aid, 0.0) for s in scenarios if aid in s["bias_by_id"]), 0.0)
        lines.append(f"### Agent {aid} — {name}  _(bias: {bias_label(bias)}, pre-bucket: {bucket_from_bias(bias)})_\n")
        for s in scenarios:
            iv = s["iv_by_id"].get(aid, "")
            if iv:
                preview = iv if len(iv) <= 600 else iv[:600] + "…"
                lines.append(f"**[{s['label']}]** {preview}")
            else:
                lines.append(f"**[{s['label']}]** _(no interview)_")
            lines.append("")
        lines.append("---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dirs", nargs="+")
    ap.add_argument("--cross-out", default=None,
                    help="Output path for cross-scenario report (default: alongside scenarios)")
    args = ap.parse_args()

    written = []
    for d in args.scenario_dirs:
        try:
            p = write_per_scenario_summary(d)
            print(f"  wrote {p}", flush=True)
            written.append(p)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[error] {d}: {exc}", flush=True)

    if len(args.scenario_dirs) >= 2:
        cross_out = args.cross_out or os.path.join(
            os.path.dirname(args.scenario_dirs[0].rstrip("/")), "cross_scenario_summary.md"
        )
        try:
            p = write_cross_scenario(args.scenario_dirs, cross_out)
            if p:
                print(f"  wrote {p}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[error] cross-scenario: {exc}", flush=True)


if __name__ == "__main__":
    main()
