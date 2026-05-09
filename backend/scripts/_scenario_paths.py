"""Path resolver for the stock-scenario directory layout.

New canonical layout (preferred):
    <root>/<TICKER>/inputs/<scenario>/        -- source docs + catalyst.txt
    <root>/<TICKER>/results/<scenario>/       -- per-run snapshots

Legacy flat layout (still resolved for backward compatibility):
    <root>/<TICKER>_<scenario>/               -- inputs
    <root>/<TICKER>_<scenario>_results/       -- outputs
"""

from __future__ import annotations

import os
from typing import Optional


def inputs_dir(root: str, ticker: str, scenario: str) -> Optional[str]:
    """Return the inputs directory for a (ticker, scenario), or None if absent.

    Tries the new layout first, then the legacy flat layout.
    """
    candidates = [
        os.path.join(root, ticker, "inputs", scenario),
        os.path.join(root, f"{ticker}_{scenario}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def results_dir(root: str, ticker: str, scenario: str, *, create: bool = False) -> str:
    """Return the canonical results directory.

    On a fresh run, prefers the new layout. If a legacy `<TICKER>_<scenario>_results`
    dir already exists, that one is returned to avoid silent path drift.
    """
    new = os.path.join(root, ticker, "results", scenario)
    legacy = os.path.join(root, f"{ticker}_{scenario}_results")
    if os.path.isdir(legacy) and not os.path.isdir(new):
        return legacy
    if create:
        os.makedirs(new, exist_ok=True)
    return new


def catalyst_file(root: str, ticker: str, scenario: str) -> Optional[str]:
    """Return path to `catalyst.txt` for a scenario, or None if absent."""
    inputs = inputs_dir(root, ticker, scenario)
    if not inputs:
        return None
    path = os.path.join(inputs, "catalyst.txt")
    return path if os.path.isfile(path) else None
