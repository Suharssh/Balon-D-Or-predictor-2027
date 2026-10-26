"""Past results, and integrity checks against each edition's spec."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .spec import AwardSpec


@dataclass(frozen=True)
class ResultRow:
    year: int
    rank: int
    player: str
    nationality: str
    club: str
    points: int


def load_results(path: str | Path) -> list[ResultRow]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            ResultRow(int(r["year"]), int(r["rank"]), r["player"], r["nationality"], r["club"], int(r["points"]))
            for r in csv.DictReader(f)
        ]


def check_edition(rows: list[ResultRow], spec: AwardSpec) -> list[str]:
    """Problems found in one edition's results. An empty list means it is consistent.

    Every juror hands out exactly ``sum(points)`` points, so the edition's
    total must equal voters x points per ballot. This catches typos and a
    wrong points scale.
    """
    problems = []
    if spec.jury is None:
        return problems
    total = sum(r.points for r in rows)
    expected = spec.jury.voters * spec.jury.points_per_ballot
    if total != expected:
        problems.append(
            f"{spec.edition}: points sum to {total}, expected {expected} "
            f"({spec.jury.voters} voters x {spec.jury.points_per_ballot})"
        )
    if spec.shortlist_size and len(rows) != spec.shortlist_size:
        problems.append(f"{spec.edition}: {len(rows)} players listed, shortlist is {spec.shortlist_size}")
    pts = [r.points for r in sorted(rows, key=lambda r: r.rank)]
    if pts != sorted(pts, reverse=True):
        problems.append(f"{spec.edition}: points do not decrease with rank")
    return problems
