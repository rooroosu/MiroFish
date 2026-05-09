"""End-to-end stock-scenario orchestrator for MiroFish.

Drives the existing HTTP API + on-disk simulation config to run one
(ticker, scenario) experiment: upload docs -> ontology -> graph -> simulation
create/prepare -> patch initial_posts catalyst -> start -> poll -> interview
-> sentiment aggregator -> report.

Usage:
    uv run python backend/scripts/run_stock_scenario.py \
        --ticker AAPL \
        --scenario bear_miss \
        --docs-dir backend/uploads/stock_scenarios/AAPL/inputs/bear_miss \
        --catalyst-file backend/uploads/stock_scenarios/AAPL/inputs/bear_miss/catalyst.txt \
        --agents 12 --rounds 5 --platform twitter

Requires the backend server to be running on --backend-url (default
http://localhost:5000). Run `cd backend && uv run python run.py` first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import Config  # noqa: E402
from app.services.simulation_runner import SimulationRunner  # noqa: E402
from app.services import stock_sentiment_aggregator as agg  # noqa: E402


SCENARIO_REQUIREMENTS = {
    "bear_miss": (
        "Simulate social-media reaction of retail investors, momentum traders, "
        "value investors, short-sellers, sell-side analysts, and financial media "
        "to {ticker} reporting a quarterly earnings miss with a guidance cut. "
        "Investor agents differ by holding period, stance, risk tolerance, and "
        "prior exposure to the stock."
    ),
    "bull_beat": (
        "Simulate social-media reaction of retail investors, momentum traders, "
        "value investors, short-sellers, sell-side analysts, and financial media "
        "to {ticker} reporting a quarterly earnings beat with raised guidance. "
        "Investor agents differ by holding period, stance, risk tolerance, and "
        "prior exposure to the stock."
    ),
    "neutral_inline": (
        "Simulate social-media reaction of retail investors, momentum traders, "
        "value investors, short-sellers, sell-side analysts, and financial media "
        "to {ticker} reporting a mixed, roughly in-line quarter with unchanged "
        "guidance. Investor agents differ by holding period and stance."
    ),
    "no_catalyst": (
        "Simulate the same investor population discussing {ticker} during a "
        "quiet trading week with no material news. Conversation is "
        "business-as-usual — positioning, valuation debate, macro comments."
    ),
}

DEFAULT_CATALYST = {
    "bear_miss": "{ticker} quarterly report: EPS below consensus, guidance lowered. Shares trading lower after-hours.",
    "bull_beat": "{ticker} quarterly report: EPS above consensus, guidance raised. Shares trading higher after-hours.",
    "neutral_inline": "{ticker} quarterly report: results roughly in line with consensus, guidance unchanged. Shares little changed after-hours.",
    "no_catalyst": "",
}


def log(msg: str) -> None:
    print(f"[stock-scenario] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[stock-scenario] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


class Client:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.sess = requests.Session()

    def _unwrap(self, r: requests.Response) -> Dict[str, Any]:
        if not r.ok:
            die(f"HTTP {r.status_code} {r.request.method} {r.request.url}: {r.text[:500]}")
        payload = r.json()
        if not payload.get("success", True):
            die(f"API failure: {payload.get('error')}")
        return payload.get("data", payload)

    def upload_docs(self, docs_dir: str, simulation_requirement: str, project_name: str) -> Dict:
        files = []
        opened = []
        for fn in sorted(os.listdir(docs_dir)):
            path = os.path.join(docs_dir, fn)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            if ext not in {"pdf", "md", "txt", "markdown"}:
                continue
            fh = open(path, "rb")
            opened.append(fh)
            files.append(("files", (fn, fh, "application/octet-stream")))
        if not files:
            die(f"no .pdf/.md/.txt found in {docs_dir}")
        try:
            r = self.sess.post(
                f"{self.base}/api/graph/ontology/generate",
                files=files,
                data={
                    "simulation_requirement": simulation_requirement,
                    "project_name": project_name,
                },
                timeout=600,
            )
            return self._unwrap(r)
        finally:
            for fh in opened:
                fh.close()

    def build_graph(self, project_id: str) -> Dict:
        r = self.sess.post(
            f"{self.base}/api/graph/build",
            json={"project_id": project_id},
            timeout=60,
        )
        return self._unwrap(r)

    def get_task(self, task_id: str) -> Dict:
        r = self.sess.get(f"{self.base}/api/graph/task/{task_id}", timeout=30)
        return self._unwrap(r)

    def wait_task(self, task_id: str, label: str, timeout_s: int = 1800, poll: int = 5) -> Dict:
        t0 = time.time()
        while True:
            data = self.get_task(task_id)
            status = data.get("status")
            log(f"  {label}: {status} ({data.get('progress', 0)}%)")
            if status in {"completed", "success"}:
                return data
            if status in {"failed", "error"}:
                die(f"{label} failed: {data.get('error') or data}")
            if time.time() - t0 > timeout_s:
                die(f"{label} timed out after {timeout_s}s")
            time.sleep(poll)

    def create_simulation(self, project_id: str, graph_id: str, platform: str) -> Dict:
        r = self.sess.post(
            f"{self.base}/api/simulation/create",
            json={
                "project_id": project_id,
                "graph_id": graph_id,
                "enable_twitter": platform in {"twitter", "parallel"},
                "enable_reddit": platform in {"reddit", "parallel"},
            },
            timeout=30,
        )
        return self._unwrap(r)

    def prepare(self, simulation_id: str, entity_types: Optional[List[str]] = None) -> Dict:
        payload: Dict[str, Any] = {"simulation_id": simulation_id, "use_llm_for_profiles": True}
        if entity_types:
            payload["entity_types"] = entity_types
        r = self.sess.post(
            f"{self.base}/api/simulation/prepare",
            json=payload,
            timeout=60,
        )
        return self._unwrap(r)

    def prepare_status(self, simulation_id: str, task_id: Optional[str]) -> Dict:
        payload = {"simulation_id": simulation_id}
        if task_id:
            payload["task_id"] = task_id
        r = self.sess.post(
            f"{self.base}/api/simulation/prepare/status",
            json=payload,
            timeout=120,  # Flask dev server is single-threaded; bump for long-running prepare
        )
        return self._unwrap(r)

    def wait_prepare(self, simulation_id: str, task_id: Optional[str], timeout_s: int = 2400, poll: int = 5) -> Dict:
        t0 = time.time()
        while True:
            data = self.prepare_status(simulation_id, task_id)
            status = data.get("status")
            log(f"  prepare: {status} ({data.get('progress', 0)}%)")
            if status in {"ready", "completed"}:
                return data
            if status in {"failed", "error"}:
                die(f"prepare failed: {data.get('error') or data}")
            if time.time() - t0 > timeout_s:
                die(f"prepare timed out after {timeout_s}s")
            time.sleep(poll)

    def start(self, simulation_id: str, platform: str, max_rounds: int) -> Dict:
        r = self.sess.post(
            f"{self.base}/api/simulation/start",
            json={
                "simulation_id": simulation_id,
                "platform": platform,
                "max_rounds": max_rounds,
                "enable_graph_memory_update": False,
            },
            timeout=60,
        )
        return self._unwrap(r)

    def run_status(self, simulation_id: str) -> Dict:
        r = self.sess.get(
            f"{self.base}/api/simulation/{simulation_id}/run-status",
            timeout=120,
        )
        return self._unwrap(r)

    def wait_run(self, simulation_id: str, timeout_s: int = 5400, poll: int = 10) -> Dict:
        t0 = time.time()
        last_round = -1
        while True:
            data = self.run_status(simulation_id)
            status = data.get("runner_status")
            cur = data.get("current_round", 0)
            total = data.get("total_rounds", 0)
            if cur != last_round:
                log(f"  sim: {status} round {cur}/{total} actions={data.get('total_actions_count', 0)}")
                last_round = cur
            if status in {"completed", "finished", "idle"}:
                return data
            if status in {"failed", "error"}:
                die(f"simulation failed: {data.get('error') or data}")
            if time.time() - t0 > timeout_s:
                die(f"simulation timed out after {timeout_s}s")
            time.sleep(poll)

    def interview_all(self, simulation_id: str, prompt: str, platform: Optional[str] = None, timeout: int = 300) -> Dict:
        r = self.sess.post(
            f"{self.base}/api/simulation/interview/all",
            json={
                "simulation_id": simulation_id,
                "prompt": prompt,
                "platform": platform,
                "timeout": timeout,
            },
            timeout=timeout + 30,
        )
        return self._unwrap(r)

    def report_generate(self, simulation_id: str) -> Dict:
        r = self.sess.post(
            f"{self.base}/api/report/generate",
            json={"simulation_id": simulation_id},
            timeout=60,
        )
        return self._unwrap(r)

    def report_status(self, task_id: Optional[str], simulation_id: str) -> Dict:
        payload = {"simulation_id": simulation_id}
        if task_id:
            payload["task_id"] = task_id
        r = self.sess.post(
            f"{self.base}/api/report/generate/status",
            json=payload,
            timeout=30,
        )
        return self._unwrap(r)


def resolve_sim_dir(simulation_id: str) -> str:
    return os.path.join(SimulationRunner.RUN_STATE_DIR, simulation_id)


def _truncate_local_graph(graph_id: str, max_nodes: int) -> None:
    """For local_qmd graphs, cap the node/edge count before prepare fans out."""
    app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
    path = os.path.join(app_root, "uploads", "local_graphs", f"{graph_id}.json")
    if not os.path.exists(path):
        log(f"  (local graph file not found at {path}, skipping truncate)")
        return
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    before_n, before_e = len(g.get("nodes", [])), len(g.get("edges", []))
    if before_n <= max_nodes:
        log(f"  (graph has {before_n} nodes, below cap {max_nodes}; no truncate)")
        return
    kept_nodes = g["nodes"][:max_nodes]
    kept_ids = {n["uuid"] for n in kept_nodes}
    kept_edges = [e for e in g.get("edges", [])
                  if e.get("source_node_uuid") in kept_ids and e.get("target_node_uuid") in kept_ids]
    g["nodes"] = kept_nodes
    g["edges"] = kept_edges
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, indent=2)
    log(f"  truncated local graph: nodes {before_n}->{len(kept_nodes)}, edges {before_e}->{len(kept_edges)}")


def load_catalyst(scenario: str, ticker: str, catalyst_file: Optional[str]) -> str:
    if catalyst_file:
        with open(catalyst_file, "r", encoding="utf-8") as f:
            return f.read().strip().format(ticker=ticker)
    return DEFAULT_CATALYST[scenario].format(ticker=ticker)


def patch_simulation_config(
    sim_dir: str,
    ticker: str,
    scenario: str,
    catalyst_content: str,
    agents: int,
) -> None:
    cfg_path = os.path.join(sim_dir, "simulation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    agent_configs = cfg.get("agent_configs", [])
    if agents and len(agent_configs) > agents:
        cfg["agent_configs"] = agent_configs[:agents]
        log(f"  truncated agent_configs to {agents} agents (was {len(agent_configs)})")

    event_cfg = cfg.get("event_config", {})
    if catalyst_content:
        poster_agent_id = 0
        if cfg.get("agent_configs"):
            poster_agent_id = cfg["agent_configs"][0].get("agent_id", 0)
        event_cfg["initial_posts"] = [
            {"poster_agent_id": poster_agent_id, "content": catalyst_content}
        ]
    else:
        event_cfg["initial_posts"] = []
    event_cfg["hot_topics"] = [ticker, "earnings", "guidance"]
    event_cfg["narrative_direction"] = f"{scenario} reaction to {ticker}"
    cfg["event_config"] = event_cfg

    cfg["scenario_label"] = scenario
    cfg["stock_ticker"] = ticker

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"  patched simulation_config.json (initial_posts x{len(event_cfg['initial_posts'])})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--scenario", required=True, choices=list(SCENARIO_REQUIREMENTS.keys()))
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--catalyst-file", default=None)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--platform", default="twitter", choices=["twitter", "reddit", "parallel"])
    ap.add_argument("--backend-url", default="http://localhost:5000")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-interview", action="store_true")
    ap.add_argument(
        "--entity-types",
        default="RetailInvestor,InvestmentAnalyst,InvestmentBank,FundManager",
        help="Comma-separated entity types to include in prepare. "
             "Empty string means include all (expensive on large ontologies).",
    )
    ap.add_argument("--max-entities", type=int, default=0,
                    help="If >0, truncate the local graph nodes to this many before prepare.")
    args = ap.parse_args()

    errors = Config.validate()
    if errors:
        die(f"config errors: {errors}. Add OpenRouter key to .env before running.")

    c = Client(args.backend_url)
    requirement = SCENARIO_REQUIREMENTS[args.scenario].format(ticker=args.ticker)
    project_name = f"{args.ticker}_{args.scenario}"
    catalyst = load_catalyst(args.scenario, args.ticker, args.catalyst_file)

    log(f"scenario: ticker={args.ticker} scenario={args.scenario} agents={args.agents} rounds={args.rounds} platform={args.platform}")
    log(f"docs: {args.docs_dir}")
    log(f"catalyst: {catalyst[:120]!r}")

    log("step 1/7: upload docs + generate ontology")
    ontology = c.upload_docs(args.docs_dir, requirement, project_name)
    project_id = ontology["project_id"]
    log(f"  project_id={project_id}")
    log(f"  entity_types={[et.get('name') for et in ontology['ontology'].get('entity_types', [])]}")

    log("step 2/7: build graph")
    build = c.build_graph(project_id)
    task_id = build.get("task_id")
    c.wait_task(task_id, "graph-build")
    graph_id = c.get_task(task_id).get("result", {}).get("graph_id") or build.get("graph_id")
    log(f"  graph_id={graph_id}")

    log("step 3/7: create + prepare simulation")
    sim = c.create_simulation(project_id, graph_id, args.platform)
    simulation_id = sim["simulation_id"]
    log(f"  simulation_id={simulation_id}")
    if args.max_entities > 0:
        _truncate_local_graph(graph_id, args.max_entities)
    entity_types = [t.strip() for t in args.entity_types.split(",") if t.strip()] if args.entity_types else None
    if entity_types:
        log(f"  prepare entity_types filter: {entity_types}")
    prep = c.prepare(simulation_id, entity_types=entity_types)
    prep_task_id = prep.get("task_id")
    c.wait_prepare(simulation_id, prep_task_id)

    sim_dir = resolve_sim_dir(simulation_id)
    log(f"  sim_dir={sim_dir}")

    log("step 4/7: patch simulation_config.json (catalyst + topics + ticker)")
    patch_simulation_config(sim_dir, args.ticker, args.scenario, catalyst, args.agents)

    log("step 5/7: start simulation")
    c.start(simulation_id, args.platform, args.rounds)
    run = c.wait_run(simulation_id)
    log(f"  final: round={run.get('current_round')}/{run.get('total_rounds')} actions={run.get('total_actions_count')}")

    if not args.skip_interview:
        log("step 6/7: interview all agents (scenario-blind prompt)")
        try:
            interview_platform = None if args.platform == "parallel" else args.platform
            c.interview_all(
                simulation_id,
                prompt=f"Summarize your current view of {args.ticker}.",
                platform=interview_platform,
                timeout=300,
            )
        except SystemExit:
            raise
        except Exception as exc:
            log(f"  interview skipped: {exc}")

    log("step 7a/7: run sentiment aggregator")
    result, out_path = agg.aggregate_to_file(
        sim_dir=sim_dir,
        scenario=args.scenario,
        ticker=args.ticker,
        platform=args.platform if args.platform != "parallel" else "",
    )
    log(f"  sentiment_report.json -> {out_path}")
    log(f"  distribution: {result['sentiment_distribution']}  stance_shift_keys: {list(result['stance_shift'].keys())}")

    template_path = agg.sample_gold_template(sim_dir, platform=args.platform if args.platform != "parallel" else "")
    log(f"  gold template: {template_path}  (hand-label, then rename to gold_labels.jsonl and rerun aggregator)")

    if not args.skip_report:
        log("step 7b/7: generate report")
        rep = c.report_generate(simulation_id)
        rep_task_id = rep.get("task_id")
        t0 = time.time()
        while True:
            status_data = c.report_status(rep_task_id, simulation_id)
            s = status_data.get("status")
            log(f"  report: {s} ({status_data.get('progress', 0)}%)")
            if s in {"completed", "success"}:
                log(f"  report_id={status_data.get('report_id') or rep.get('report_id')}")
                break
            if s in {"failed", "error"}:
                log(f"  report failed: {status_data.get('error')}")
                break
            if time.time() - t0 > 2400:
                log("  report timed out")
                break
            time.sleep(10)

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
