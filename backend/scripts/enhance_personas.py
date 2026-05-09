"""Apply stock-mode persona enhancements to a prepared simulation.

Usage:
    uv run python backend/scripts/enhance_personas.py \
        --simulation-id sim_585b8bc7d6e1 \
        --target-agents 30 \
        --synth-ratio 0.5 \
        --locale en

Reads <RUN_STATE_DIR>/<sim_id>/{simulation_config.json, twitter_profiles.csv,
reddit_profiles.json}, injects sentiment-bias stance directives into existing
personas (A2), and appends synthetic stock-archetype personas (A1).

Always make backups first time (.pre_enhance.bak); subsequent runs are idempotent
on the bias-directive (guarded by '[Stock stance]' substring presence) but will
keep adding more synthetic agents if rerun. Use --restore-backup to roll back.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.simulation_runner import SimulationRunner  # noqa: E402
from app.services.stock_persona_enhancer import apply_stock_enhancements  # noqa: E402


def restore_backups(sim_dir: str) -> int:
    n = 0
    for fname in ("twitter_profiles.csv", "reddit_profiles.json", "simulation_config.json"):
        bak = os.path.join(sim_dir, fname + ".pre_enhance.bak")
        target = os.path.join(sim_dir, fname)
        if os.path.exists(bak):
            shutil.copy2(bak, target)
            n += 1
            print(f"  restored {target} from backup")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation-id", required=True)
    ap.add_argument("--target-agents", type=int, default=30, help="final total agent count after enhancement")
    ap.add_argument("--synth-ratio", type=float, default=0.5, help="floor for synthetic ratio (0..1)")
    ap.add_argument("--locale", default="en", choices=["en", "zh"])
    ap.add_argument("--restore-backup", action="store_true",
                    help="Roll back to .pre_enhance.bak files instead of enhancing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only; do not modify files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sim_dir = os.path.join(SimulationRunner.RUN_STATE_DIR, args.simulation_id)
    if not os.path.isdir(sim_dir):
        print(f"sim_dir not found: {sim_dir}", file=sys.stderr)
        sys.exit(2)

    if args.restore_backup:
        n = restore_backups(sim_dir)
        print(f"restored {n} file(s)")
        return

    if args.dry_run:
        from app.services.stock_persona_enhancer import plan_enhancements
        plan = plan_enhancements(
            sim_dir=sim_dir,
            target_total_agents=args.target_agents,
            synth_ratio=args.synth_ratio,
            locale=args.locale,
        )
        print(json.dumps({
            "sim_dir": plan.sim_dir,
            "ticker": plan.ticker,
            "existing_agent_count": plan.existing_agent_count,
            "target_total_agents": plan.target_total_agents,
            "synth_per_archetype": plan.synth_per_archetype,
            "bias_injection_count": plan.bias_injection_count,
        }, indent=2))
        return

    summary = apply_stock_enhancements(
        sim_dir=sim_dir,
        target_total_agents=args.target_agents,
        synth_ratio=args.synth_ratio,
        locale=args.locale,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
