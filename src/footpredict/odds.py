"""Betting-market baseline: turn outright odds into voter-model logits.

Market prices already carry a lot of public information, so any real model
has to beat this baseline before we trust it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .simulate import simulate
from .spec import JuryRules

FIELD = "*"  # marks that unpriced candidates are allowed (the field)
DEFAULT_FIELD_SHARE = 0.05


@dataclass(frozen=True)
class MarketSnapshot:
    candidates: list[str]
    probs: np.ndarray        # implied win probabilities, overround removed
    priced: list[bool]       # False where the field price was used
    source: str


def load_odds(path: str | Path, candidates: list[str], field_share: float = DEFAULT_FIELD_SHARE) -> MarketSnapshot:
    """Read ``player,decimal_odds,source,date`` rows.

    Bookmakers usually price only the contenders and quote the rest as
    "25/1 or bigger". A ``*`` row allows unpriced candidates: together they get
    ``field_share`` of the probability, split equally, and the priced players
    share the rest in proportion to their implied odds (which also removes the
    bookmaker's margin). Its odds value is kept only for the record.
    """
    prices: dict[str, float] = {}
    sources: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prices[row["player"]] = float(row["decimal_odds"])
            if row.get("source"):
                sources.add(f"{row['source']} ({row.get('date', '')})".strip())

    unknown = set(prices) - set(candidates) - {FIELD}
    if unknown:
        raise ValueError(f"odds given for players not on the shortlist: {sorted(unknown)}")

    priced = [c in prices for c in candidates]
    n_field = priced.count(False)
    if n_field and FIELD not in prices:
        missing = [c for c, ok in zip(candidates, priced) if not ok]
        raise ValueError(f"no price for {missing} and no field ('*') row")
    share = field_share if n_field else 0.0

    implied = np.array([1.0 / prices[c] if ok else 0.0 for c, ok in zip(candidates, priced)])
    p = np.where(priced, (1.0 - share) * implied / implied.sum(), share / max(n_field, 1))
    return MarketSnapshot(list(candidates), p, priced, "; ".join(sorted(sources)))


def calibrate_shock(
    market: MarketSnapshot,
    rules: JuryRules,
    grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    n_sims: int = 1500,
    seed: int = 0,
) -> tuple[float, float]:
    """Pick the jury shock size whose simulated win odds best match the market.

    Returns (shock_sd, KL divergence from market to simulation).
    """
    logits = np.log(market.probs)
    best = (grid[0], np.inf)
    for sd in grid:
        res = simulate(market.candidates, logits, rules, shock_sd=sd, n_sims=n_sims, seed=seed)
        q = np.clip(res.win, 1e-4, None)
        kl = float(np.sum(market.probs * np.log(market.probs / q)))
        if kl < best[1]:
            best = (sd, kl)
    return best
