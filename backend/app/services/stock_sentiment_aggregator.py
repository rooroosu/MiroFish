"""Stock-scenario sentiment aggregator.

Reads a MiroFish simulation's SQLite trace + simulation_config.json, classifies
CREATE_POST content and INTERVIEW responses into {bullish, bearish, neutral}
via Qwen, then computes the sentiment distribution and a 3x3 stance shift
vs. each agent's pre-declared sentiment_bias.

Trace schema contract (verified against run_parallel_simulation.py):
    trace(rowid, user_id, action, info)   -- info is JSON text
      action='create_post'  -> info.content  (post text)
      action='interview'    -> info.response (post-sim stance quote)

Engagement-ranked top quotes are optional; when the simulation DB contains
a `post` table (Twitter/Reddit harness), we use like counts from trace rows
with action='like_post' referencing the post_id.

Classifier validation gate: if a gold-label JSONL is present at
    <sim_dir>/gold_labels.jsonl   (one line per {post_rowid, label})
we compute agreement vs. the classifier and include it in the output.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


SENTIMENT_LABELS = ("bullish", "bearish", "neutral")
STANCE_THRESHOLD = 0.2  # |sentiment_bias| >= this maps to bullish/bearish; else neutral


@dataclass
class _Post:
    rowid: int
    user_id: int
    content: str


@dataclass
class _Interview:
    user_id: int
    response: str


def bucket_from_bias(bias: float) -> str:
    if bias > STANCE_THRESHOLD:
        return "bullish"
    if bias < -STANCE_THRESHOLD:
        return "bearish"
    return "neutral"


def bucket_majority(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts = {k: labels.count(k) for k in SENTIMENT_LABELS}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top, runner = ranked[0], ranked[1]
    if top[1] == 0:
        return None
    if top[1] == runner[1]:
        return "neutral"
    return top[0]


def _resolve_db_path(sim_dir: str, platform: str) -> str:
    candidates = []
    if platform:
        candidates.append(os.path.join(sim_dir, f"{platform}_simulation.db"))
    candidates += [
        os.path.join(sim_dir, "twitter_simulation.db"),
        os.path.join(sim_dir, "reddit_simulation.db"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No simulation DB found under {sim_dir}")


def _load_catalyst_markers(sim_dir: str) -> List[str]:
    """Catalyst posts injected by three_way_gate / run_stock_scenario must not
    be counted as agent sentiment. Markers are written to <sim_dir>/catalyst_markers.json
    at injection time. Legacy fallback: any content starting with '🚨 Newswire' is
    treated as catalyst.
    """
    path = os.path.join(sim_dir, "catalyst_markers.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(c).strip() for c in data.get("contents", []) if c]
    except Exception as exc:
        logger.warning("catalyst_markers.json parse failed: %s", exc)
        return []


def _is_catalyst(content: str, markers: List[str]) -> bool:
    if not content:
        return False
    if content.lstrip().startswith("🚨 Newswire"):
        return True
    norm = content.strip()
    for m in markers:
        if not m:
            continue
        # exact match OR catalyst content is a prefix of the post (defensive against whitespace)
        if norm == m.strip() or norm.startswith(m.strip()[:80]):
            return True
    return False


def _load_posts(db_path: str, sim_dir: str = "") -> List[_Post]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT rowid, user_id, info FROM trace WHERE action = 'create_post' ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    markers = _load_catalyst_markers(sim_dir) if sim_dir else []
    out: List[_Post] = []
    skipped_catalyst = 0
    for rowid, user_id, info in rows:
        try:
            data = json.loads(info) if info else {}
        except json.JSONDecodeError:
            continue
        content = (data.get("content") or "").strip()
        if not content:
            continue
        if _is_catalyst(content, markers):
            skipped_catalyst += 1
            continue
        out.append(_Post(rowid=rowid, user_id=int(user_id), content=content))
    if skipped_catalyst:
        logger.info("skipped %d catalyst row(s) from sentiment count", skipped_catalyst)
    return out


def _load_interviews(db_path: str) -> List[_Interview]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT user_id, info, created_at FROM trace WHERE action = 'interview' "
            "ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()
    logger.info("interview rows in trace: %d (db=%s)", len(rows), db_path)
    latest: Dict[int, str] = {}
    parse_fail = 0
    empty_resp = 0
    for user_id, info, _created in rows:
        try:
            data = json.loads(info) if info else {}
        except json.JSONDecodeError:
            parse_fail += 1
            continue
        resp = data.get("response")
        if isinstance(resp, (dict, list)):
            resp = json.dumps(resp, ensure_ascii=False)
        resp = (resp or "").strip()
        if resp:
            latest[int(user_id)] = resp  # last one wins due to ORDER BY created_at
        else:
            empty_resp += 1
    if parse_fail or empty_resp:
        logger.info("interview rows: parse_fail=%d empty_response=%d", parse_fail, empty_resp)
    return [_Interview(user_id=uid, response=resp) for uid, resp in latest.items()]


def _engagement_by_post(db_path: str) -> Dict[int, int]:
    """Approximate post engagement via count of like_post trace rows referencing
    each post_id. Returns {trace_rowid_of_create_post: like_count}. Falls back
    to empty dict if schema doesn't match (reddit edge cases)."""
    try:
        conn = sqlite3.connect(db_path)
        create_rows = conn.execute(
            "SELECT rowid, info FROM trace WHERE action = 'create_post'"
        ).fetchall()
        rowid_for_post_id: Dict[int, int] = {}
        for rowid, info in create_rows:
            try:
                data = json.loads(info) if info else {}
            except json.JSONDecodeError:
                continue
            pid = data.get("post_id") or data.get("new_post_id")
            if pid is not None:
                rowid_for_post_id[int(pid)] = rowid
        like_rows = conn.execute(
            "SELECT info FROM trace WHERE action = 'like_post'"
        ).fetchall()
        counts: Dict[int, int] = {}
        for (info,) in like_rows:
            try:
                data = json.loads(info) if info else {}
            except json.JSONDecodeError:
                continue
            pid = data.get("post_id") or data.get("like_id")
            if pid is None:
                continue
            rowid = rowid_for_post_id.get(int(pid))
            if rowid is None:
                continue
            counts[rowid] = counts.get(rowid, 0) + 1
        return counts
    except Exception as exc:
        logger.info("engagement count skipped: %s", exc)
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _classify_batch(client: LLMClient, ticker: str, texts: List[str]) -> List[str]:
    """Return one label per input text. Falls back to 'neutral' on parse errors."""
    if not texts:
        return []
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    system = (
        "You classify stock-market social-media posts for their stance on a "
        "specific ticker. Labels: bullish (expects price to rise / positive on "
        "the company), bearish (expects price to fall / negative on the "
        "company), neutral (no directional stance, question, off-topic, or "
        "balanced). Output a JSON object of the form "
        '{"labels": ["bullish"|"bearish"|"neutral", ...]} with exactly one '
        "label per input, preserving order. No prose, no code fence."
    )
    user = f"Ticker: {ticker}\n\nPosts:\n{numbered}"
    try:
        data = client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=256 + 8 * len(texts),
        )
        labels = data.get("labels") or []
        out: List[str] = []
        for i in range(len(texts)):
            lbl = labels[i] if i < len(labels) else "neutral"
            lbl = (lbl or "").strip().lower()
            if lbl not in SENTIMENT_LABELS:
                lbl = "neutral"
            out.append(lbl)
        return out
    except Exception as exc:
        logger.warning("classifier batch failed: %s", exc)
        return ["neutral"] * len(texts)


def _classify_all(
    client: LLMClient, ticker: str, texts: List[str], batch_size: int = 10
) -> List[str]:
    out: List[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        out.extend(_classify_batch(client, ticker, chunk))
    return out


def _load_gold(sim_dir: str) -> Dict[int, str]:
    path = os.path.join(sim_dir, "gold_labels.jsonl")
    if not os.path.exists(path):
        return {}
    gold: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rowid = rec.get("post_rowid")
            label = (rec.get("label") or "").strip().lower()
            if rowid is not None and label in SENTIMENT_LABELS:
                gold[int(rowid)] = label
    return gold


def _distribution(labels: List[str]) -> Dict[str, int]:
    return {k: labels.count(k) for k in SENTIMENT_LABELS}


def _shift_matrix(pre: List[str], post: List[str]) -> Dict[str, int]:
    assert len(pre) == len(post)
    m: Dict[str, int] = {}
    for a, b in zip(pre, post):
        key = f"{a}_to_{b}"
        m[key] = m.get(key, 0) + 1
    return m


def aggregate(sim_dir: str, scenario: str, ticker: str, platform: str = "") -> Dict:
    config_path = os.path.join(sim_dir, "simulation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        sim_config = json.load(f)
    agent_configs = sim_config.get("agent_configs", [])
    pre_bias_by_agent = {
        int(ac.get("agent_id")): float(ac.get("sentiment_bias", 0.0))
        for ac in agent_configs
        if ac.get("agent_id") is not None
    }
    agent_name_by_id = {
        int(ac.get("agent_id")): ac.get("entity_name", f"Agent_{ac.get('agent_id')}")
        for ac in agent_configs
        if ac.get("agent_id") is not None
    }

    db_path = _resolve_db_path(sim_dir, platform)
    posts = _load_posts(db_path, sim_dir=sim_dir)
    interviews = _load_interviews(db_path)
    engagement = _engagement_by_post(db_path)

    client = LLMClient(model=Config.LLM_MODEL_NAME)

    post_labels = _classify_all(client, ticker, [p.content for p in posts])
    interview_labels = _classify_all(client, ticker, [iv.response for iv in interviews])

    # Per-agent post-stance: majority over the agent's own posts; fall back to interview label.
    labels_by_agent: Dict[int, List[str]] = {}
    for post, label in zip(posts, post_labels):
        labels_by_agent.setdefault(post.user_id, []).append(label)
    interview_by_agent = {iv.user_id: lbl for iv, lbl in zip(interviews, interview_labels)}

    pre_buckets: List[str] = []
    post_buckets: List[str] = []
    for agent_id, bias in pre_bias_by_agent.items():
        pre_buckets.append(bucket_from_bias(bias))
        post = bucket_majority(labels_by_agent.get(agent_id, [])) or interview_by_agent.get(agent_id)
        post_buckets.append(post or bucket_from_bias(bias))

    # Top quotes by engagement.
    ranked = sorted(
        zip(posts, post_labels),
        key=lambda pair: (engagement.get(pair[0].rowid, 0), len(pair[0].content)),
        reverse=True,
    )
    top_bullish = [p.content for p, lbl in ranked if lbl == "bullish"][:5]
    top_bearish = [p.content for p, lbl in ranked if lbl == "bearish"][:5]

    interview_excerpts = [
        {"agent": agent_name_by_id.get(iv.user_id, f"Agent_{iv.user_id}"),
         "response": iv.response,
         "label": lbl}
        for iv, lbl in zip(interviews, interview_labels)
    ]

    classifier_validation: Dict = {"gold_n": 0}
    gold = _load_gold(sim_dir)
    if gold:
        matched = [(lbl, gold[p.rowid]) for p, lbl in zip(posts, post_labels) if p.rowid in gold]
        if matched:
            agree = sum(1 for pred, gt in matched if pred == gt)
            classifier_validation = {
                "gold_n": len(matched),
                "agreement": round(agree / len(matched), 4),
            }

    return {
        "ticker": ticker,
        "scenario": scenario,
        "sim_dir": sim_dir,
        "db_path": db_path,
        "platform_inferred": os.path.basename(db_path).replace("_simulation.db", ""),
        "agent_count": len(pre_bias_by_agent),
        "post_count": len(posts),
        "interview_count": len(interviews),
        "sentiment_distribution": _distribution(post_labels),
        "interview_distribution": _distribution(interview_labels),
        "stance_shift": _shift_matrix(pre_buckets, post_buckets),
        "pre_bucket_distribution": _distribution(pre_buckets),
        "post_bucket_distribution": _distribution(post_buckets),
        "thresholds": {
            "bullish_if_bias_gt": STANCE_THRESHOLD,
            "bearish_if_bias_lt": -STANCE_THRESHOLD,
        },
        "top_bullish_quotes": top_bullish,
        "top_bearish_quotes": top_bearish,
        "interview_excerpts": interview_excerpts,
        "classifier_validation": classifier_validation,
    }


def aggregate_to_file(
    sim_dir: str, scenario: str, ticker: str, platform: str = ""
) -> Tuple[Dict, str]:
    result = aggregate(sim_dir=sim_dir, scenario=scenario, ticker=ticker, platform=platform)
    out_path = os.path.join(sim_dir, "sentiment_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result, out_path


def sample_gold_template(sim_dir: str, platform: str = "", n: int = 20, seed: int = 42) -> str:
    """Write gold_labels_template.jsonl with n random post rowids for manual labeling."""
    db_path = _resolve_db_path(sim_dir, platform)
    posts = _load_posts(db_path, sim_dir=sim_dir)
    if not posts:
        raise ValueError("no posts found to sample")
    rng = random.Random(seed)
    sample = rng.sample(posts, k=min(n, len(posts)))
    out_path = os.path.join(sim_dir, "gold_labels_template.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps({
                "post_rowid": p.rowid,
                "user_id": p.user_id,
                "content": p.content,
                "label": "",  # fill: bullish | bearish | neutral
            }, ensure_ascii=False) + "\n")
    return out_path
