"""Turn ballots into a final standing, following an award's jury rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import JuryRules


@dataclass(frozen=True)
class Standing:
    candidate: str
    points: int
    rank_counts: tuple[int, ...]  # how many voters put this candidate 1st, 2nd, ...


def order_by_rules(totals: np.ndarray, rank_counts: np.ndarray) -> np.ndarray:
    """Indices of candidates from best to worst.

    Most points first; ties broken by most 1st places, then most 2nd places,
    and so on. Any tie left after that is broken by input order (on the real
    award the organiser decides, which we cannot model).
    """
    n, k = rank_counts.shape
    # np.lexsort uses the last key as primary, so list the weakest tie-break first.
    keys = [np.arange(n)] + [-rank_counts[:, r] for r in reversed(range(k))] + [-totals]
    return np.lexsort(keys)


def tally(ballots: list[list[str]], rules: JuryRules) -> list[Standing]:
    """Score a list of ballots. Each ballot lists candidates from 1st place down."""
    names: list[str] = []
    index: dict[str, int] = {}
    for ballot in ballots:
        if len(ballot) > rules.ballot_size:
            raise ValueError(f"ballot has {len(ballot)} names but rules allow {rules.ballot_size}")
        if len(set(ballot)) != len(ballot):
            raise ValueError(f"ballot names a candidate twice: {ballot}")
        for name in ballot:
            if name not in index:
                index[name] = len(names)
                names.append(name)

    k = rules.ballot_size
    totals = np.zeros(len(names), dtype=np.int64)
    counts = np.zeros((len(names), k), dtype=np.int64)
    for ballot in ballots:
        for r, name in enumerate(ballot):
            i = index[name]
            totals[i] += rules.points[r]
            counts[i, r] += 1

    order = order_by_rules(totals, counts)
    return [Standing(names[i], int(totals[i]), tuple(int(c) for c in counts[i])) for i in order]
