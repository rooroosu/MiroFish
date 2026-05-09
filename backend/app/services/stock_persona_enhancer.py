"""Stock-mode persona enhancement (A1 + A2).

Post-processes a prepared simulation's profile files to:

A1 — Pool overhaul: append synthetic stock-trader archetype personas
     (momentum_trader, retail_enthusiast, short_seller, value_investor,
     swing_trader) so the agent pool is not 100% professional/compliance voices.

A2 — Inject sentiment_bias as a stance directive into the persona text so
     OASIS agents actually post bullishly/bearishly in-character.

Designed as a post-prepare step, not a deep modification of
oasis_profile_generator. Reads simulation_config.json + twitter_profiles.csv /
reddit_profiles.json from <sim_dir> and rewrites in place (with a .bak backup
the first time).

Always routes LLM calls through utils.llm_client.LLMClient; never bypasses
_require_qwen.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


STOCK_ARCHETYPES = {
    "momentum_trader": (
        "Aggressive momentum trader. Buys breakouts, follows volume spikes and "
        "social-media chatter, posts fast and confidently, often using trader "
        "slang and emojis (\U0001F680 \U0001F4C8). Talks about charts, RSI, MACD, "
        "short interest, gamma squeezes. Highly directional — when bullish, "
        "explicit price-target calls; when bearish, calls puts."
    ),
    "retail_enthusiast": (
        "Enthusiastic retail investor with strong brand loyalty. Discovered the "
        "stock through a personal experience or hype cycle. Defends it on "
        "social media against critics, frames every dip as 'buying opportunity', "
        "posts memes and conviction-building threads. Emotional, not "
        "compliance-trained, willing to make explicit bullish predictions."
    ),
    "short_seller": (
        "Activist short-seller. Publishes detailed bearish theses pointing at "
        "accounting irregularities, valuation premiums, regulatory risk, "
        "competitive moats eroding. Adversarial tone, names risks explicitly, "
        "challenges management's narrative. Posts thread-style with citations. "
        "Comfortable being structurally bearish."
    ),
    "value_investor": (
        "Value-oriented long-term investor in the Buffett/Munger tradition. "
        "Focuses on intrinsic value, free cash flow, capital allocation, "
        "moat durability. Patient, contrarian, often takes the unpopular side "
        "(buys when others sell, sells when others buy). Direct opinions on "
        "fair value, willing to call mispricings."
    ),
    "swing_trader": (
        "Multi-day swing trader. Combines technicals with catalyst-driven "
        "thesis (earnings, FDA approvals, macro prints). Posts daily watchlists, "
        "calls out setups, takes a side and sizes position. Less compliance-toned "
        "than institutional analysts; willing to be wrong publicly."
    ),
}


@dataclass
class EnhancementPlan:
    sim_dir: str
    locale: str
    ticker: str
    target_total_agents: int  # final agent count after enhancement
    synth_per_archetype: Dict[str, int]  # how many of each archetype to add
    bias_injection_count: int  # how many existing agents will get a stance directive
    existing_agent_count: int


BIAS_DIRECTIVE_THRESHOLD = 0.2  # match stock_sentiment_aggregator.STANCE_THRESHOLD


def _bias_directive(bias: float, ticker: str) -> str:
    """Convert per-agent sentiment_bias into a persona stance paragraph.

    Returns empty string for |bias| ≤ BIAS_DIRECTIVE_THRESHOLD. Aligns with
    stock_sentiment_aggregator.STANCE_THRESHOLD so the 3×3 stance matrix is
    measurable — agents whose pre-bucket is bullish/bearish get a matching
    in-character directive.
    """
    t = ticker or "the stock"
    if bias > BIAS_DIRECTIVE_THRESHOLD:
        return (
            f"\n\n[Stock stance] You are structurally bullish on {t}. "
            f"Track-record of contrarian buy calls, conviction-driven framing, "
            f"comfortable making explicit price-target predictions. You frame "
            f"downside catalysts as buying opportunities and defend the long "
            f"thesis even under pressure. When you post about {t}, your "
            f"directional view is unambiguous."
        )
    if bias < -BIAS_DIRECTIVE_THRESHOLD:
        return (
            f"\n\n[Stock stance] You are structurally bearish on {t}. "
            f"Short bias, focus on regulatory headwinds, valuation premiums, "
            f"competitive risks, and accounting concerns. Fundamentals-first "
            f"skepticism — willing to publicly challenge bullish narratives. "
            f"When you post about {t}, you name the bear case explicitly."
        )
    return ""


def _archetype_prompt(archetype: str, ticker: str, idx: int, locale: str) -> str:
    blurb = STOCK_ARCHETYPES[archetype]
    lang_instruction = (
        "Please respond in English." if locale != "zh"
        else "请用中文回答。"
    )
    return (
        f"You are an expert at writing realistic social-media trader personas. "
        f"Generate a fictional persona (#{idx}) of archetype \"{archetype}\". "
        f"Archetype description: {blurb}\n\n"
        f"This persona should be active on stock-trading social media (Twitter/X, "
        f"Reddit r/wallstreetbets-style) and have a coherent track record of "
        f"posting about {ticker} specifically. Keep them realistic — they are "
        f"NOT a professional analyst at a bank.\n\n"
        f"Output JSON with fields:\n"
        f"  bio (≤200 chars, social-media-style),\n"
        f"  persona (1500-2000 char prose covering: background, trading style, "
        f"voice/tone, what catalysts move them, language idioms they use, prior "
        f"posts about {ticker}, conviction level on directional bets),\n"
        f"  age (integer 22-55),\n"
        f"  gender (\"male\"|\"female\"|\"other\"),\n"
        f"  mbti (4-letter code),\n"
        f"  country (English),\n"
        f"  profession (2-4 words),\n"
        f"  interested_topics (array of strings).\n\n"
        f"{lang_instruction}\n"
        f"Return ONLY valid JSON. No prose outside the JSON. No code fence."
    )


def _generate_one_archetype(
    client: LLMClient, archetype: str, ticker: str, idx: int, locale: str
) -> Optional[Dict]:
    prompt = _archetype_prompt(archetype, ticker, idx, locale)
    system = (
        "You generate realistic social-media trader personas for a stock-market "
        "swarm simulation. Output strictly valid JSON with no extra text."
    )
    try:
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85,
            max_tokens=2400,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
        # Minimal validation
        if not data.get("bio") or not data.get("persona"):
            logger.warning("archetype %s idx=%d returned without bio/persona", archetype, idx)
            return None
        data["_archetype"] = archetype
        return data
    except Exception as exc:
        logger.warning("archetype %s idx=%d generation failed: %s", archetype, idx, exc)
        return None


def _username(name: str, archetype: str, idx: int) -> str:
    base = "".join(c for c in (name or archetype).lower() if c.isalnum() or c == "_")
    return f"{base or archetype}_{random.randint(100, 9999)}"


def _profile_to_twitter_csv_row(
    user_id: int, profile: Dict, ticker: str, sentiment_bias: float
) -> Dict[str, str]:
    """Match the OASIS Twitter CSV schema (user_id, name, username, user_char, ...)."""
    name = profile.get("name") or profile.get("bio", "trader").split(",")[0][:40]
    user_char = profile["persona"] + _bias_directive(sentiment_bias, ticker)
    return {
        "user_id": str(user_id),
        "name": name,
        "username": profile.get("user_name") or _username(name, profile.get("_archetype", ""), user_id),
        "user_char": user_char,
        "description": profile["bio"],
        "created_at": "2024-01-01",
        "followers_count": str(random.randint(500, 50000)),
        "following_count": str(random.randint(50, 1500)),
    }


def _profile_to_reddit_json(
    user_id: int, profile: Dict, ticker: str, sentiment_bias: float
) -> Dict:
    persona = profile["persona"] + _bias_directive(sentiment_bias, ticker)
    return {
        "user_id": user_id,
        "username": profile.get("user_name") or _username(
            profile.get("name", ""), profile.get("_archetype", ""), user_id
        ),
        "name": profile.get("name", profile.get("bio", "trader")[:40]),
        "bio": profile["bio"],
        "persona": persona,
        "age": profile.get("age", 35),
        "gender": profile.get("gender", "other"),
        "mbti": profile.get("mbti", "INTJ"),
        "country": profile.get("country", "United States"),
        "profession": profile.get("profession", profile.get("_archetype", "trader")),
        "interested_topics": profile.get("interested_topics", [ticker, "trading", "markets"]),
    }


def _read_twitter_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def _write_twitter_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _read_reddit_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_reddit_json(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _backup_once(path: str) -> None:
    if not os.path.exists(path):
        return
    bak = path + ".pre_enhance.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def plan_enhancements(
    sim_dir: str,
    target_total_agents: int = 30,
    synth_ratio: float = 0.5,
    locale: str = "en",
) -> EnhancementPlan:
    cfg_path = os.path.join(sim_dir, "simulation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ticker = cfg.get("stock_ticker", "")
    agent_configs = cfg.get("agent_configs", [])
    existing = len(agent_configs)
    bias_inject = sum(
        1 for ac in agent_configs
        if abs(float(ac.get("sentiment_bias", 0.0))) > BIAS_DIRECTIVE_THRESHOLD
    )
    # How many synthetic agents to add to reach target_total_agents.
    synth_total = max(target_total_agents - existing, int(round(synth_ratio * target_total_agents)))
    archetypes = list(STOCK_ARCHETYPES.keys())
    base = synth_total // len(archetypes)
    rem = synth_total - base * len(archetypes)
    synth_per_archetype = {a: base + (1 if i < rem else 0) for i, a in enumerate(archetypes)}
    return EnhancementPlan(
        sim_dir=sim_dir,
        locale=locale,
        ticker=ticker,
        target_total_agents=existing + synth_total,
        synth_per_archetype=synth_per_archetype,
        bias_injection_count=bias_inject,
        existing_agent_count=existing,
    )


def inject_bias_directives_inplace(sim_dir: str) -> int:
    """A2: rewrite existing personas to include a stance directive when |bias| > 0.4.

    Returns count of profiles modified.
    """
    cfg_path = os.path.join(sim_dir, "simulation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ticker = cfg.get("stock_ticker", "")
    bias_by_id = {
        int(ac.get("agent_id", -1)): float(ac.get("sentiment_bias", 0.0))
        for ac in cfg.get("agent_configs", [])
        if ac.get("agent_id") is not None
    }
    modified = 0

    twitter_csv = os.path.join(sim_dir, "twitter_profiles.csv")
    if os.path.exists(twitter_csv):
        _backup_once(twitter_csv)
        rows, fieldnames = _read_twitter_csv(twitter_csv)
        if "user_char" not in fieldnames:
            logger.warning("twitter_profiles.csv missing user_char column — skipping bias injection")
        else:
            for r in rows:
                try:
                    uid = int(r.get("user_id", "-1"))
                except ValueError:
                    continue
                directive = _bias_directive(bias_by_id.get(uid, 0.0), ticker)
                if directive and "[Stock stance]" not in r.get("user_char", ""):
                    r["user_char"] = (r.get("user_char", "") or "") + directive
                    modified += 1
            _write_twitter_csv(twitter_csv, rows, fieldnames)

    reddit_json = os.path.join(sim_dir, "reddit_profiles.json")
    if os.path.exists(reddit_json):
        _backup_once(reddit_json)
        rows = _read_reddit_json(reddit_json)
        for r in rows:
            try:
                uid = int(r.get("user_id", -1))
            except (ValueError, TypeError):
                continue
            directive = _bias_directive(bias_by_id.get(uid, 0.0), ticker)
            if directive and "[Stock stance]" not in r.get("persona", ""):
                r["persona"] = (r.get("persona", "") or "") + directive
                modified += 1
        _write_reddit_json(reddit_json, rows)

    logger.info("injected stance directive into %d profile(s) in %s", modified, sim_dir)
    return modified


def append_synthetic_archetypes_inplace(
    sim_dir: str,
    synth_per_archetype: Dict[str, int],
    locale: str = "en",
    max_workers: int = 4,
) -> int:
    """A1: generate N synthetic archetype personas via LLM and append to profile files
    + simulation_config.agent_configs.
    Returns count appended."""
    cfg_path = os.path.join(sim_dir, "simulation_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ticker = cfg.get("stock_ticker", "")
    existing_ids = [int(ac.get("agent_id", 0)) for ac in cfg.get("agent_configs", [])]
    next_id = (max(existing_ids) + 1) if existing_ids else 0

    requested = []
    for archetype, count in synth_per_archetype.items():
        for i in range(count):
            requested.append((archetype, i))
    if not requested:
        return 0

    client = LLMClient(model=Config.LLM_MODEL_NAME)
    generated: List[Tuple[str, int, Dict]] = []  # (archetype, idx, profile)

    def task(archetype_idx):
        a, i = archetype_idx
        prof = _generate_one_archetype(client, a, ticker, i, locale)
        return a, i, prof

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, ai) for ai in requested]
        for fut in as_completed(futures):
            a, i, prof = fut.result()
            if prof is not None:
                generated.append((a, i, prof))

    if not generated:
        logger.warning("synthetic archetype generation produced 0 profiles")
        return 0

    # Distribute sentiment biases across archetypes:
    # short_seller → -0.6 to -0.9
    # momentum/retail/value/swing → -0.5 to +0.8 (mix; favor positive for momentum/retail)
    bias_map = {
        "short_seller": lambda: random.uniform(-0.9, -0.6),
        "momentum_trader": lambda: random.choice([random.uniform(0.5, 0.85), random.uniform(-0.7, -0.5)]),
        "retail_enthusiast": lambda: random.uniform(0.4, 0.85),
        "value_investor": lambda: random.choice([random.uniform(-0.6, -0.4), random.uniform(0.4, 0.6)]),
        "swing_trader": lambda: random.uniform(-0.7, 0.7),
    }

    new_agent_configs = []
    new_twitter_rows = []
    new_reddit_rows = []
    for archetype, _idx, prof in generated:
        bias = bias_map.get(archetype, lambda: 0.0)()
        uid = next_id
        next_id += 1
        prof["name"] = prof.get("name") or f"{archetype.replace('_', ' ').title()} #{uid}"
        new_agent_configs.append({
            "agent_id": uid,
            "entity_name": prof["name"],
            "sentiment_bias": round(bias, 3),
            "synthetic_archetype": archetype,
        })
        new_twitter_rows.append(_profile_to_twitter_csv_row(uid, prof, ticker, bias))
        new_reddit_rows.append(_profile_to_reddit_json(uid, prof, ticker, bias))

    # Append to twitter CSV
    twitter_csv = os.path.join(sim_dir, "twitter_profiles.csv")
    if os.path.exists(twitter_csv):
        _backup_once(twitter_csv)
        rows, fieldnames = _read_twitter_csv(twitter_csv)
        # Make sure all our new columns present in fieldnames; add unknown ones at end.
        if not fieldnames:
            fieldnames = list(new_twitter_rows[0].keys())
        for new_row in new_twitter_rows:
            for k in new_row:
                if k not in fieldnames:
                    fieldnames.append(k)
            rows.append(new_row)
        _write_twitter_csv(twitter_csv, rows, fieldnames)
        logger.info("appended %d synthetic personas to %s", len(new_twitter_rows), twitter_csv)

    # Append to reddit JSON
    reddit_json = os.path.join(sim_dir, "reddit_profiles.json")
    if os.path.exists(reddit_json):
        _backup_once(reddit_json)
        rows = _read_reddit_json(reddit_json)
        rows.extend(new_reddit_rows)
        _write_reddit_json(reddit_json, rows)
        logger.info("appended %d synthetic personas to %s", len(new_reddit_rows), reddit_json)

    # Append to simulation_config.json
    _backup_once(cfg_path)
    cfg.setdefault("agent_configs", []).extend(new_agent_configs)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger.info("appended %d agent_configs to simulation_config.json", len(new_agent_configs))

    return len(generated)


def apply_stock_enhancements(
    sim_dir: str,
    target_total_agents: int = 30,
    synth_ratio: float = 0.5,
    locale: str = "en",
) -> Dict:
    """Orchestrator: A2 (bias injection) then A1 (synthetic archetypes).

    Returns a summary dict.
    """
    plan = plan_enhancements(
        sim_dir=sim_dir,
        target_total_agents=target_total_agents,
        synth_ratio=synth_ratio,
        locale=locale,
    )
    logger.info(
        "stock-enhance plan: ticker=%s existing=%d → target=%d (synth=%s) bias_inject=%d",
        plan.ticker, plan.existing_agent_count, plan.target_total_agents,
        plan.synth_per_archetype, plan.bias_injection_count,
    )
    a2_modified = inject_bias_directives_inplace(sim_dir)
    a1_added = append_synthetic_archetypes_inplace(
        sim_dir=sim_dir,
        synth_per_archetype=plan.synth_per_archetype,
        locale=locale,
    )
    # A2 again to bias-inject the new synthetic agents that have |bias| > 0.4
    a2_modified_after = inject_bias_directives_inplace(sim_dir)
    return {
        "sim_dir": sim_dir,
        "ticker": plan.ticker,
        "plan": {
            "target_total_agents": plan.target_total_agents,
            "existing_agent_count": plan.existing_agent_count,
            "synth_per_archetype": plan.synth_per_archetype,
        },
        "a2_modified_first_pass": a2_modified,
        "a1_synthetic_added": a1_added,
        "a2_modified_total": a2_modified_after,
    }
