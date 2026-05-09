"""Run the bear/bull/no_catalyst three-way gate on an existing prepared sim.

Reuses a single simulation_id (same personas, same graph) and varies only
the catalyst. Wipes DB on each restart via /start force=true.

For each scenario: patch config -> force-start -> wait alive -> interview
-> aggregator -> save sentiment_report.json and results JSON under
backend/uploads/stock_scenarios/<TICKER>/results/<scenario>/. Legacy flat
layout `<TICKER>_<scenario>_results/` is still resolved if present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services import stock_sentiment_aggregator as agg  # noqa: E402
from app.services.simulation_runner import SimulationRunner  # noqa: E402
from scripts.run_stock_scenario import DEFAULT_CATALYST, log  # noqa: E402


def sim_dir(sim_id: str) -> str:
    return os.path.join(SimulationRunner.RUN_STATE_DIR, sim_id)


def patch_config(sim_id: str, ticker: str, scenario: str, catalyst_file: str | None):
    d = sim_dir(sim_id)
    cfg_path = os.path.join(d, "simulation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if catalyst_file and os.path.exists(catalyst_file):
        with open(catalyst_file, "r", encoding="utf-8") as f:
            content = f.read().strip().format(ticker=ticker)
    else:
        content = DEFAULT_CATALYST[scenario].format(ticker=ticker)

    ec = cfg.get("event_config", {})
    if content:
        poster = cfg.get("agent_configs", [{}])[0].get("agent_id", 0)
        ec["initial_posts"] = [{"poster_agent_id": poster, "content": content, "is_catalyst": True}]
    else:
        ec["initial_posts"] = []
    ec["hot_topics"] = [ticker, "earnings", "guidance"]
    ec["narrative_direction"] = f"{scenario} reaction to {ticker}"
    cfg["event_config"] = ec
    cfg["scenario_label"] = scenario
    cfg["stock_ticker"] = ticker

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Bug 1 sidecar: aggregator excludes any create_post whose content matches.
    markers_path = os.path.join(d, "catalyst_markers.json")
    markers = {"contents": [content] if content else []}
    with open(markers_path, "w", encoding="utf-8") as f:
        json.dump(markers, f, ensure_ascii=False, indent=2)

    log(f"  patched initial_posts={len(ec['initial_posts'])} content[:80]={content[:80]!r}")
    log(f"  wrote catalyst_markers.json with {len(markers['contents'])} marker(s)")


def wait_for_interviews(sim_id: str, expected: int, timeout_s: int = 180, poll: int = 5) -> int:
    """Bug 2 fix: API may return before all interview rows persist to trace.
    Poll the DB until count >= expected or timeout. Returns observed count."""
    import sqlite3
    d = sim_dir(sim_id)
    db_path = os.path.join(d, "twitter_simulation.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(d, "reddit_simulation.db")
    if not os.path.exists(db_path):
        log(f"  wait_for_interviews: no DB found under {d}")
        return 0
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout_s:
        try:
            c = sqlite3.connect(db_path)
            n = c.execute("SELECT COUNT(*) FROM trace WHERE action='interview'").fetchone()[0]
            c.close()
        except Exception as exc:
            log(f"  wait_for_interviews: query error {exc}")
            return last if last >= 0 else 0
        if n != last:
            log(f"  interview rows in trace: {n} (expected {expected})")
            last = n
        if n >= expected:
            return n
        time.sleep(poll)
    log(f"  wait_for_interviews: timeout at {last} rows (expected {expected})")
    return last if last >= 0 else 0


def start(base: str, sim_id: str, platform: str, rounds: int):
    r = requests.post(
        f"{base}/api/simulation/start",
        json={
            "simulation_id": sim_id, "platform": platform,
            "max_rounds": rounds, "enable_graph_memory_update": False,
            "force": True,
        }, timeout=60,
    )
    r.raise_for_status()
    log(f"  start: {r.json().get('data', {}).get('runner_status')}")


def wait_alive(sim_id: str, timeout_s: int = 1200, poll: int = 15) -> bool:
    d = sim_dir(sim_id)
    t0 = time.time()
    env_path = os.path.join(d, "env_status.json")
    last = ""
    while time.time() - t0 < timeout_s:
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    status = json.load(f).get("status")
            except Exception:
                status = None
            if status and status != last:
                log(f"  env_status={status}")
                last = status
            if status == "alive":
                return True
            if status in {"stopped", "error"}:
                return False
        time.sleep(poll)
    return False


def interview(base: str, sim_id: str, ticker: str, platform: str) -> dict:
    plat = None if platform == "parallel" else platform
    r = requests.post(
        f"{base}/api/simulation/interview/all",
        json={
            "simulation_id": sim_id,
            "prompt": f"Summarize your current view of {ticker}.",
            "platform": plat, "timeout": 300,
        }, timeout=400,
    )
    r.raise_for_status()
    return r.json()


def save_results(sim_id: str, ticker: str, scenario: str, out_dir: str):
    from scripts._scenario_paths import results_dir as _results_dir
    src = os.path.join(sim_dir(sim_id), "sentiment_report.json")
    dst_dir = _results_dir(out_dir, ticker, scenario, create=True)
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "sentiment_report.json"))
    # Also snapshot trace DB and catalyst markers + simulation_config so re-aggregation works standalone
    for fname in ("twitter_simulation.db", "reddit_simulation.db", "catalyst_markers.json", "simulation_config.json"):
        fsrc = os.path.join(sim_dir(sim_id), fname)
        if os.path.exists(fsrc):
            shutil.copy2(fsrc, os.path.join(dst_dir, fname))
    # Bug 3 fix: rewrite sim_dir + db_path in the copied JSON to point at dst_dir,
    # not the shared source sim_id.
    dst_json = os.path.join(dst_dir, "sentiment_report.json")
    if os.path.exists(dst_json):
        try:
            with open(dst_json, "r", encoding="utf-8") as f:
                rep = json.load(f)
            rep["sim_dir"] = dst_dir
            db_basename = os.path.basename(rep.get("db_path", "twitter_simulation.db"))
            rep["db_path"] = os.path.join(dst_dir, db_basename)
            rep["source_sim_id"] = sim_id
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log(f"  warning: failed to rewrite sim_dir field: {exc}")
    log(f"  saved results -> {dst_dir}")


def run_scenario(base: str, sim_id: str, ticker: str, scenario: str, catalyst_file: str | None,
                 rounds: int, platform: str, out_dir: str):
    log(f"=== scenario: {scenario} ===")
    patch_config(sim_id, ticker, scenario, catalyst_file)
    start(base, sim_id, platform, rounds)
    if not wait_alive(sim_id):
        log("  env never went alive — aborting this scenario")
        return None
    try:
        iv = interview(base, sim_id, ticker, platform)
        log(f"  interviews returned count={iv.get('data', {}).get('interviews_count')}")
    except Exception as exc:
        log(f"  interview error: {exc}")

    # Bug 2 fix: API may return before all rows land in trace; poll until persisted.
    cfg_path = os.path.join(sim_dir(sim_id), "simulation_config.json")
    expected_iv = 0
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            expected_iv = len(json.load(f).get("agent_configs", []))
    except Exception:
        pass
    if expected_iv:
        wait_for_interviews(sim_id, expected=expected_iv, timeout_s=180, poll=5)

    result, path = agg.aggregate_to_file(
        sim_dir=sim_dir(sim_id), scenario=scenario, ticker=ticker,
        platform=platform if platform != "parallel" else "",
    )
    log(f"  sentiment_distribution: {result['sentiment_distribution']}")
    log(f"  pre_bucket: {result['pre_bucket_distribution']}")
    log(f"  post_bucket: {result['post_bucket_distribution']}")
    save_results(sim_id, ticker, scenario, out_dir)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-id", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--scenarios-dir", required=True,
                    help="backend/uploads/stock_scenarios root; resolves <T>/inputs/<scen>/ "
                         "or legacy <T>_<scen>/ for inputs and writes results to <T>/results/<scen>/")
    ap.add_argument("--scenarios", default="bear_miss,bull_beat,no_catalyst")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--platform", default="twitter")
    ap.add_argument("--backend-url", default="http://localhost:5001")
    args = ap.parse_args()

    base = args.backend_url.rstrip("/")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    from scripts._scenario_paths import inputs_dir as _inputs_dir, catalyst_file as _catalyst_file
    summary = {}
    for scen in scenarios:
        scen_dir = _inputs_dir(args.scenarios_dir, args.ticker, scen)
        catalyst = _catalyst_file(args.scenarios_dir, args.ticker, scen)
        result = run_scenario(
            base=base, sim_id=args.simulation_id, ticker=args.ticker,
            scenario=scen, catalyst_file=catalyst,
            rounds=args.rounds, platform=args.platform, out_dir=args.scenarios_dir,
        )
        if result:
            summary[scen] = {
                "post_distribution": result["sentiment_distribution"],
                "post_bucket_distribution": result["post_bucket_distribution"],
                "stance_shift": result["stance_shift"],
            }

    log("\n=== three-way gate summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Gate check
    if all(k in summary for k in ("bear_miss", "bull_beat", "no_catalyst")):
        def share(d, k):
            total = sum(d.values()) or 1
            return d.get(k, 0) / total * 100
        bull_share_bear = share(summary["bear_miss"]["post_distribution"], "bullish")
        bull_share_bull = share(summary["bull_beat"]["post_distribution"], "bullish")
        gap = bull_share_bull - bull_share_bear
        log(f"gate: bull%[bull_beat]={bull_share_bull:.1f}  bull%[bear_miss]={bull_share_bear:.1f}  gap={gap:.1f}pp (criterion: >=25pp)")


if __name__ == "__main__":
    main()
