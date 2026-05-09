"""Resume a MiroFish stock scenario from an existing, prepared simulation_id.

Use after run_stock_scenario.py dies mid-pipeline (e.g. on prepare-status
timeout) but the simulation itself has finished preparing on the backend.

Picks up at: patch config -> start -> poll run-status -> interview -> aggregator.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import stock_sentiment_aggregator as agg  # noqa: E402
from app.services.simulation_runner import SimulationRunner  # noqa: E402
from scripts.run_stock_scenario import (  # noqa: E402
    Client, DEFAULT_CATALYST, log, patch_simulation_config, resolve_sim_dir,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-id", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--scenario", required=True,
                    choices=["bear_miss", "bull_beat", "neutral_inline", "no_catalyst"])
    ap.add_argument("--catalyst-file", default=None)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--platform", default="twitter",
                    choices=["twitter", "reddit", "parallel"])
    ap.add_argument("--backend-url", default="http://localhost:5001")
    ap.add_argument("--skip-interview", action="store_true")
    args = ap.parse_args()

    sim_dir = resolve_sim_dir(args.simulation_id)
    if not os.path.exists(os.path.join(sim_dir, "simulation_config.json")):
        log(f"ERROR: simulation_config.json missing in {sim_dir} — prepare not finished?")
        return 2

    if args.catalyst_file:
        with open(args.catalyst_file, "r", encoding="utf-8") as f:
            catalyst = f.read().strip().format(ticker=args.ticker)
    else:
        catalyst = DEFAULT_CATALYST[args.scenario].format(ticker=args.ticker)

    log(f"patching config in {sim_dir}")
    patch_simulation_config(sim_dir, args.ticker, args.scenario, catalyst, args.agents)

    c = Client(args.backend_url)

    log("starting simulation")
    c.start(args.simulation_id, args.platform, args.rounds)
    run = c.wait_run(args.simulation_id)
    log(f"final: round={run.get('current_round')}/{run.get('total_rounds')} actions={run.get('total_actions_count')}")

    if not args.skip_interview:
        log("interviewing agents (scenario-blind)")
        try:
            interview_platform = None if args.platform == "parallel" else args.platform
            c.interview_all(
                args.simulation_id,
                prompt=f"Summarize your current view of {args.ticker}.",
                platform=interview_platform,
                timeout=300,
            )
        except SystemExit:
            raise
        except Exception as exc:
            log(f"interview skipped: {exc}")

    log("running aggregator")
    result, out_path = agg.aggregate_to_file(
        sim_dir=sim_dir,
        scenario=args.scenario,
        ticker=args.ticker,
        platform=args.platform if args.platform != "parallel" else "",
    )
    log(f"sentiment_report.json -> {out_path}")
    log(f"distribution: {result['sentiment_distribution']}")
    log(f"stance_shift_keys: {list(result['stance_shift'].keys())}")

    template_path = agg.sample_gold_template(sim_dir, platform=args.platform if args.platform != "parallel" else "")
    log(f"gold template: {template_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
