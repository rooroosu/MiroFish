"""Classifier κ + bootstrap CI for stock-sentiment classifier validation.

Usage:
    uv run python backend/scripts/classifier_agreement.py \
        backend/uploads/stock_scenarios/NU/results/bear_miss \
        backend/uploads/stock_scenarios/NU/results/bull_beat \
        backend/uploads/stock_scenarios/NU/results/no_catalyst

Per scenario dir, reads gold_labels_template.jsonl (each row has
post_rowid, content, label — user fills label ∈ {bullish, bearish, neutral}),
re-runs the Qwen classifier on each text, computes Cohen's κ + per-class
P/R + bootstrap 95% CI for κ.

Threshold:
    κ ≥ 0.6                → PASS
    0.4 ≤ κ < 0.6          → DIRECTIONAL_ONLY (use directional shifts only)
    κ < 0.4                → FAIL
Outputs <scenario_dir>/classifier_agreement.json per scenario + a combined report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config  # noqa: E402
from app.utils.llm_client import LLMClient  # noqa: E402
from app.services.stock_sentiment_aggregator import _classify_all  # noqa: E402

LABELS = ("bullish", "bearish", "neutral")


def cohen_kappa(y_true: List[str], y_pred: List[str]) -> float:
    """Cohen's kappa for categorical labels."""
    n = len(y_true)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    po = agree / n
    counts_t = Counter(y_true)
    counts_p = Counter(y_pred)
    pe = sum((counts_t[lbl] / n) * (counts_p[lbl] / n) for lbl in LABELS)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_kappa_ci(
    y_true: List[str], y_pred: List[str], n_iter: int = 1000, seed: int = 42
) -> Tuple[float, float, float]:
    """Returns (kappa_point_estimate, ci_lo_2.5, ci_hi_97.5)."""
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = cohen_kappa(y_true, y_pred)
    samples = []
    for _ in range(n_iter):
        idxs = [rng.randrange(n) for _ in range(n)]
        bt = [y_true[i] for i in idxs]
        bp = [y_pred[i] for i in idxs]
        samples.append(cohen_kappa(bt, bp))
    samples.sort()
    lo = samples[int(0.025 * n_iter)]
    hi = samples[int(0.975 * n_iter) - 1]
    return point, lo, hi


def per_class_pr(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for lbl in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "support_true": tp + fn, "support_pred": tp + fp}
    return out


def confusion(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    m = {a: {b: 0 for b in LABELS} for a in LABELS}
    for t, p in zip(y_true, y_pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def threshold_label(kappa: float) -> str:
    if kappa >= 0.6:
        return "PASS"
    if kappa >= 0.4:
        return "DIRECTIONAL_ONLY"
    return "FAIL"


def load_gold(scenario_dir: str) -> List[Dict]:
    """Returns list of {post_rowid, content, label} where label is non-empty
    and in LABELS. Skips unlabeled rows."""
    path = os.path.join(scenario_dir, "gold_labels_template.jsonl")
    if not os.path.exists(path):
        path2 = os.path.join(scenario_dir, "gold_labels.jsonl")
        if os.path.exists(path2):
            path = path2
        else:
            return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = (rec.get("label") or "").strip().lower()
            if label in LABELS and rec.get("content"):
                out.append({
                    "post_rowid": rec.get("post_rowid"),
                    "content": rec["content"],
                    "label": label,
                })
    return out


def evaluate_scenario(scenario_dir: str, ticker: str = "NU") -> Optional[Dict]:
    print(f"\n=== {scenario_dir} ===", flush=True)
    gold = load_gold(scenario_dir)
    if not gold:
        print("  no labeled gold rows found (file empty or all labels blank)", flush=True)
        return None
    print(f"  loaded {len(gold)} labeled gold rows", flush=True)

    client = LLMClient(model=Config.LLM_MODEL_NAME)
    texts = [g["content"] for g in gold]
    y_true = [g["label"] for g in gold]
    y_pred = _classify_all(client, ticker, texts, batch_size=10)

    kappa, lo, hi = bootstrap_kappa_ci(y_true, y_pred, n_iter=1000)
    pr = per_class_pr(y_true, y_pred)
    cm = confusion(y_true, y_pred)
    verdict = threshold_label(kappa)

    print(f"  κ = {kappa:.3f}  [95% CI: {lo:.3f}, {hi:.3f}]  → {verdict}", flush=True)
    print(f"  per-class P/R: {pr}", flush=True)

    out = {
        "scenario_dir": scenario_dir,
        "n_labeled": len(gold),
        "kappa_point": round(kappa, 4),
        "kappa_ci_95_low": round(lo, 4),
        "kappa_ci_95_high": round(hi, 4),
        "verdict": verdict,
        "per_class_pr": pr,
        "confusion": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    out_path = os.path.join(scenario_dir, "classifier_agreement.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out_path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dirs", nargs="+")
    ap.add_argument("--ticker", default="NU")
    ap.add_argument("--combined-out", default=None,
                    help="optional path to write a combined cross-scenario report JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    results = []
    pooled_y_true = []
    pooled_y_pred = []
    for d in args.scenario_dirs:
        r = evaluate_scenario(d, ticker=args.ticker)
        if r:
            results.append(r)
            pooled_y_true.extend(r["y_true"])
            pooled_y_pred.extend(r["y_pred"])

    if not results:
        print("\n[no labeled gold sets found across scenarios — fill in 'label' field of gold_labels_template.jsonl first]")
        return

    if pooled_y_true:
        kappa, lo, hi = bootstrap_kappa_ci(pooled_y_true, pooled_y_pred, n_iter=1000)
        verdict = threshold_label(kappa)
        print(f"\n=== POOLED across {len(results)} scenario(s), n={len(pooled_y_true)} ===", flush=True)
        print(f"  κ = {kappa:.3f}  [95% CI: {lo:.3f}, {hi:.3f}]  → {verdict}", flush=True)
        if args.combined_out:
            with open(args.combined_out, "w", encoding="utf-8") as f:
                json.dump({
                    "scenarios": [
                        {"scenario_dir": r["scenario_dir"], "kappa": r["kappa_point"],
                         "ci_low": r["kappa_ci_95_low"], "ci_high": r["kappa_ci_95_high"],
                         "verdict": r["verdict"], "n": r["n_labeled"]}
                        for r in results
                    ],
                    "pooled": {"kappa": round(kappa, 4),
                               "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                               "verdict": verdict, "n": len(pooled_y_true)},
                }, f, ensure_ascii=False, indent=2)
            print(f"  wrote combined report -> {args.combined_out}", flush=True)


if __name__ == "__main__":
    main()
