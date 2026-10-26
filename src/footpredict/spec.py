"""Award specifications.

An AwardSpec describes one edition of one award: the eligibility window, who
decides it, and how ballots turn into a result. Everything award-specific lives
in a TOML file under ``specs/`` so the engine itself stays generic.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DECIDER_TYPES = {"jury", "stat", "committee"}
TIEBREAKS = {"rank_counts"}


@dataclass(frozen=True)
class JuryRules:
    voters: int
    points: tuple[int, ...]
    tiebreak: str = "rank_counts"

    @property
    def ballot_size(self) -> int:
        return len(self.points)

    @property
    def points_per_ballot(self) -> int:
        return sum(self.points)


@dataclass(frozen=True)
class AwardSpec:
    award: str
    edition: int
    window_start: date
    window_end: date
    decider: str
    jury: JuryRules | None = None
    shortlist_size: int | None = None
    notes: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.award}_{self.edition}"


def _parse(raw: dict) -> AwardSpec:
    decider = raw["decider"]["type"]
    if decider not in DECIDER_TYPES:
        raise ValueError(f"unknown decider type {decider!r}; expected one of {sorted(DECIDER_TYPES)}")

    jury = None
    if decider == "jury":
        d = raw["decider"]
        points = tuple(int(p) for p in d["points"])
        if list(points) != sorted(points, reverse=True) or points[-1] < 0:
            raise ValueError("jury points must be non-increasing and non-negative")
        tiebreak = d.get("tiebreak", "rank_counts")
        if tiebreak not in TIEBREAKS:
            raise ValueError(f"unknown tiebreak {tiebreak!r}")
        jury = JuryRules(voters=int(d["voters"]), points=points, tiebreak=tiebreak)

    window = raw["window"]
    start, end = window["start"], window["end"]
    if end < start:
        raise ValueError("window end is before window start")

    return AwardSpec(
        award=raw["award"],
        edition=int(raw["edition"]),
        window_start=start,
        window_end=end,
        decider=decider,
        jury=jury,
        shortlist_size=raw.get("shortlist_size"),
        notes=raw.get("notes", ""),
        sources=tuple(raw.get("sources", ())),
    )


def load_spec(path: str | Path) -> AwardSpec:
    with open(path, "rb") as f:
        return _parse(tomllib.load(f))


def find_spec(award: str, edition: int, specs_dir: str | Path = "specs") -> AwardSpec:
    return load_spec(Path(specs_dir) / f"{award}_{edition}.toml")
