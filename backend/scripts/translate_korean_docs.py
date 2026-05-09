"""Batch-translate Korean Markdown/Text docs into cached English siblings.

Walks one or more paths, finds .md/.markdown/.txt files whose Hangul ratio
exceeds the configured threshold, and writes `<name>.en.<ext>` next to each.
Re-runs are no-ops thanks to the SHA-256 source hash baked into the cache.

Usage:
    uv run python backend/scripts/translate_korean_docs.py \
        backend/uploads/stock_scenarios/NU \
        backend/uploads/stock_scenarios/SOFI

    uv run python backend/scripts/translate_korean_docs.py \
        --threshold 0.05 --model qwen/qwen-turbo \
        backend/uploads/stock_scenarios
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config  # noqa: E402
from app.services.doc_translator import (  # noqa: E402
    _cache_path,
    _content_sha256,
    _read_cache_if_fresh,
    _write_cache,
    korean_ratio,
    translate_text,
)
from app.utils.file_parser import _read_text_with_fallback  # noqa: E402


SUPPORTED_EXT = {".md", ".markdown", ".txt"}


def iter_candidates(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        rp = Path(root)
        if rp.is_file():
            out.append(rp)
            continue
        if not rp.is_dir():
            print(f"[translate] skip (not found): {root}", file=sys.stderr)
            continue
        for p in rp.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if p.stem.endswith(".en"):
                continue
            out.append(p)
    return sorted(set(out))


def process(path: Path, threshold: float, model: str | None, force: bool) -> str:
    text = _read_text_with_fallback(str(path))
    ratio = korean_ratio(text)
    if ratio < threshold:
        return f"skip (ko_ratio={ratio:.2f} < {threshold})"

    cache_path = _cache_path(str(path))
    source_sha = _content_sha256(text)

    if not force:
        cached = _read_cache_if_fresh(cache_path, source_sha)
        if cached is not None:
            return f"cached (ko_ratio={ratio:.2f})"

    translated = translate_text(text, model=model)
    _write_cache(cache_path, source_sha, translated)
    return f"translated (ko_ratio={ratio:.2f}, {len(text)} -> {len(translated)} chars)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Files or directories to scan recursively")
    ap.add_argument("--threshold", type=float,
                    default=Config.MIROFISH_TRANSLATION_KO_THRESHOLD,
                    help="Minimum Hangul-character ratio to translate (default from config)")
    ap.add_argument("--model", default=None,
                    help=f"Override translation model (default: {Config.MIROFISH_TRANSLATION_MODEL})")
    ap.add_argument("--force", action="store_true", help="Re-translate even if cache is fresh")
    args = ap.parse_args()

    candidates = iter_candidates(args.paths)
    if not candidates:
        print("[translate] no candidate files found")
        return 0

    print(f"[translate] {len(candidates)} candidate file(s); threshold={args.threshold}")
    for p in candidates:
        try:
            status = process(p, args.threshold, args.model, args.force)
        except Exception as exc:  # pragma: no cover
            status = f"error: {exc}"
        print(f"  {p}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
