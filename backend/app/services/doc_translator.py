"""
Document translator — Korean → English on ingest.

Sub-agents read stock-analysis docs many times; Korean Hangul fragments hard
under BPE (≈3× tokens vs English equivalent). Translating once at ingest and
caching the English version saves tokens on every downstream read.

Cache convention: `<name>.en.md` written next to the original `<name>.md`.
The cache is keyed by SHA-256 of the source content; if the source changes,
the cache is regenerated.

Translation is performed by a cheap Qwen model (default `qwen/qwen-turbo`)
to satisfy the project's qwen-only routing rule (see `model_routing.py`).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Optional

from ..config import Config
from ..model_routing import is_placeholder_secret, is_qwen_model
from ..utils.logger import get_logger

logger = get_logger('mirofish.translator')

_HANGUL_RE = re.compile(r'[가-힣]')
_CACHE_HEADER_RE = re.compile(r'^<!--\s*mirofish-translation\s+source_sha256=([0-9a-f]{64})\s*-->\s*\n', re.IGNORECASE)
_TRANSLATION_LOCK = threading.Lock()


def korean_ratio(text: str) -> float:
    """Fraction of characters that are Hangul syllables. 0..1."""
    if not text:
        return 0.0
    hangul = len(_HANGUL_RE.findall(text))
    return hangul / len(text)


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _cache_path(source_path: str) -> str:
    """`/x/foo.md` -> `/x/foo.en.md`. `/x/foo.txt` -> `/x/foo.en.txt`."""
    p = Path(source_path)
    suffix = p.suffix or '.md'
    return str(p.with_suffix('') ) + '.en' + suffix


def _read_cache_if_fresh(cache_path: str, source_sha: str) -> Optional[str]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            head = f.read(200)
            m = _CACHE_HEADER_RE.match(head)
            if not m or m.group(1) != source_sha:
                return None
            f.seek(0)
            content = f.read()
            return _CACHE_HEADER_RE.sub('', content, count=1)
    except Exception as exc:
        logger.warning(f"cache read failed for {cache_path}: {exc}")
        return None


def _write_cache(cache_path: str, source_sha: str, translated: str) -> None:
    try:
        header = f"<!-- mirofish-translation source_sha256={source_sha} -->\n"
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(translated)
    except Exception as exc:
        logger.warning(f"cache write failed for {cache_path}: {exc}")


_TRANSLATE_SYSTEM_PROMPT = (
    "You translate Korean financial and stock-analysis documents into clear, professional English. "
    "Preserve every number, ticker symbol, currency amount, percentage, date, and proper noun verbatim. "
    "Preserve markdown structure (headings, tables, lists, code blocks, blockquotes) exactly. "
    "Keep already-English passages unchanged. Do NOT summarize, omit, or add commentary. "
    "Output only the translated document with no preface or trailing notes."
)


def _translate_via_llm(text: str, model: Optional[str] = None) -> str:
    """Single-shot translation. Caller chunks if needed."""
    from openai import OpenAI

    api_key = Config.LLM_API_KEY
    base_url = Config.LLM_BASE_URL
    model_name = model or Config.MIROFISH_TRANSLATION_MODEL

    if is_placeholder_secret(api_key):
        raise ValueError("LLM_API_KEY is not configured; cannot translate")
    if not is_qwen_model(model_name):
        raise ValueError(f"MIROFISH_TRANSLATION_MODEL must be a qwen/... id, got: {model_name}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=8192,
    )
    out = resp.choices[0].message.content or ""
    out = re.sub(r'<think>[\s\S]*?</think>', '', out).strip()
    if not out:
        raise ValueError(f"translator {model_name} returned empty content")
    return out


def _split_for_translation(text: str, max_chars: int = 12000) -> list[str]:
    """Split on blank-line boundaries so markdown structure survives chunking."""
    if len(text) <= max_chars:
        return [text]
    paras = text.split('\n\n')
    chunks: list[str] = []
    buf = ''
    for p in paras:
        candidate = (buf + '\n\n' + p) if buf else p
        if len(candidate) > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def translate_text(text: str, model: Optional[str] = None) -> str:
    """Translate a Korean-bearing string to English. Chunks long inputs."""
    pieces = _split_for_translation(text)
    out = []
    for i, piece in enumerate(pieces, 1):
        if len(pieces) > 1:
            logger.info(f"translating chunk {i}/{len(pieces)} ({len(piece)} chars)")
        out.append(_translate_via_llm(piece, model=model))
    return '\n\n'.join(out)


def should_translate(text: str) -> bool:
    """Decide whether a doc warrants translation."""
    if not Config.MIROFISH_AUTO_TRANSLATE:
        return False
    return korean_ratio(text) >= Config.MIROFISH_TRANSLATION_KO_THRESHOLD


def translate_file_if_needed(source_path: str, source_text: str) -> str:
    """
    If `source_text` has enough Korean to warrant translation, return the
    cached/freshly-translated English version. Otherwise return `source_text`
    unchanged.

    The English version is persisted next to the source as `<name>.en.<ext>`
    and reused on subsequent ingests if the source content hash matches.
    """
    if not should_translate(source_text):
        return source_text

    cache_path = _cache_path(source_path)
    source_sha = _content_sha256(source_text)

    cached = _read_cache_if_fresh(cache_path, source_sha)
    if cached is not None:
        logger.debug(f"translation cache hit: {cache_path}")
        return cached

    with _TRANSLATION_LOCK:
        cached = _read_cache_if_fresh(cache_path, source_sha)
        if cached is not None:
            return cached
        ratio = korean_ratio(source_text)
        logger.info(f"translating {source_path} ({len(source_text)} chars, ko_ratio={ratio:.2f})")
        try:
            translated = translate_text(source_text)
        except Exception as exc:
            logger.error(f"translation failed for {source_path}: {exc}; falling back to source text")
            return source_text
        _write_cache(cache_path, source_sha, translated)
        return translated
