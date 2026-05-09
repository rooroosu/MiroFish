"""Simulation-level tools exposed to ReportAgent (non-Zep).

Thin wrappers that read MiroFish simulation artifacts directly (SQLite trace,
simulation_config.json) and return dict payloads. Kept separate from
zep_tools.py because these are not graph-memory operations.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from . import stock_sentiment_aggregator as agg
from .simulation_runner import SimulationRunner

logger = logging.getLogger(__name__)


def _resolve_sim_dir(simulation_id: str) -> str:
    sim_dir = os.path.join(SimulationRunner.RUN_STATE_DIR, simulation_id)
    if not os.path.exists(sim_dir):
        raise FileNotFoundError(f"simulation dir not found: {sim_dir}")
    return sim_dir


def _read_sim_config(sim_dir: str) -> Dict:
    path = os.path.join(sim_dir, "simulation_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sentiment_distribution(
    simulation_id: str,
    ticker: str,
    scenario: str = "",
    platform: str = "",
    cache: bool = True,
) -> Dict:
    """Classify create_post + interview content in the simulation trace and
    return a sentiment-distribution summary. Result is persisted to
    <sim_dir>/sentiment_report.json and reused on subsequent calls when
    cache=True."""

    sim_dir = _resolve_sim_dir(simulation_id)
    cached_path = os.path.join(sim_dir, "sentiment_report.json")
    if cache and os.path.exists(cached_path):
        with open(cached_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if not scenario:
        sim_config = _read_sim_config(sim_dir)
        scenario = sim_config.get("scenario_label", sim_config.get("simulation_requirement", "")[:40])

    result, _ = agg.aggregate_to_file(
        sim_dir=sim_dir,
        scenario=scenario,
        ticker=ticker,
        platform=platform,
    )
    return result


def format_sentiment_for_report(payload: Dict) -> str:
    """Compact human-readable rendering for injection into ReportAgent context."""
    lines = [
        f"Ticker: {payload.get('ticker')}",
        f"Scenario: {payload.get('scenario')}",
        f"Agents: {payload.get('agent_count')}   Posts: {payload.get('post_count')}   "
        f"Interviews: {payload.get('interview_count')}",
        f"Post-content sentiment distribution: {payload.get('sentiment_distribution')}",
        f"Interview-stance distribution: {payload.get('interview_distribution')}",
        f"Pre-bucket distribution: {payload.get('pre_bucket_distribution')}",
        f"Post-bucket distribution: {payload.get('post_bucket_distribution')}",
        f"Stance shift matrix: {payload.get('stance_shift')}",
        f"Classifier validation: {payload.get('classifier_validation')}",
        "",
        "Top bullish quotes:",
    ]
    for q in payload.get("top_bullish_quotes", [])[:3]:
        lines.append(f"  > {q}")
    lines.append("Top bearish quotes:")
    for q in payload.get("top_bearish_quotes", [])[:3]:
        lines.append(f"  > {q}")
    lines.append("Interview excerpts:")
    for iv in payload.get("interview_excerpts", [])[:5]:
        lines.append(f"  - [{iv.get('label')}] {iv.get('agent')}: {iv.get('response')}")
    return "\n".join(lines)
