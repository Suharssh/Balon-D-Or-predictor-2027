"""Monte Carlo simulation of a jury vote.

Each simulated edition draws one shared shock per candidate (our uncertainty
about how the jury as a whole rates them), then lets every voter fill in a
ballot independently with the Plackett-Luce voter model. The ballots are
scored with the award's points and tie-break rules.

Without the shared shock, 100 independent voters would almost always agree on
the favourite and the model would be wildly overconfident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plackett_luce import sample_top_k
from .spec import JuryRules
from .tally import order_by_rules


@dataclass(frozen=True)
class SimulationResult:
    candidates: list[str]
    win: np.ndarray          # P(finish 1st)
    podium: np.ndarray       # P(finish top 3)
    top10: np.ndarray        # P(finish top 10)
    mean_points: np.ndarray
    mean_rank: np.ndarray
    n_sims: int

    def table(self) -> list[dict]:
        rows = [
            {
                "candidate": c,
                "win": float(self.win[i]),
                "podium": float(self.podium[i]),
                "top10": float(self.top10[i]),
                "mean_points": float(self.mean_points[i]),
                "mean_rank": float(self.mean_rank[i]),
            }
            for i, c in enumerate(self.candidates)
        ]
        return sorted(rows, key=lambda r: (-r["win"], -r["mean_points"]))


def simulate(
    candidates: list[str],
    logits: np.ndarray,
    rules: JuryRules,
    shock_sd: float = 0.5,
    n_sims: int = 5000,
    seed: int = 0,
    chunk: int = 500,
) -> SimulationResult:
    n = len(candidates)
    if logits.shape != (n,):
        raise ValueError("need one logit per candidate")
    k = min(rules.ballot_size, n)
    points = np.asarray(rules.points[:k], dtype=np.int64)
    rng = np.random.default_rng(seed)

    wins = np.zeros(n)
    podium = np.zeros(n)
    top10 = np.zeros(n)
    pts_sum = np.zeros(n)
    rank_sum = np.zeros(n)

    done = 0
    while done < n_sims:
        s = min(chunk, n_sims - done)
        jury_logits = logits[None, :] + shock_sd * rng.standard_normal((s, n))
        voter_logits = np.broadcast_to(jury_logits[:, None, :], (s, rules.voters, n))
        ballots = sample_top_k(voter_logits, k, rng)  # (s, voters, k)

        totals = np.zeros((s, n), dtype=np.int64)
        counts = np.zeros((s, n, k), dtype=np.int64)
        sim_idx = np.arange(s)[:, None, None]
        rank_idx = np.arange(k)[None, None, :]
        np.add.at(totals, (np.broadcast_to(sim_idx, ballots.shape), ballots), np.broadcast_to(points, ballots.shape))
        np.add.at(counts, (np.broadcast_to(sim_idx, ballots.shape), ballots, np.broadcast_to(rank_idx, ballots.shape)), 1)

        for j in range(s):
            order = order_by_rules(totals[j], counts[j])
            position = np.empty(n, dtype=np.int64)
            position[order] = np.arange(n)
            wins[order[0]] += 1
            podium[order[:3]] += 1
            top10[order[:10]] += 1
            rank_sum += position + 1
        pts_sum += totals.sum(axis=0)
        done += s

    return SimulationResult(
        candidates=list(candidates),
        win=wins / n_sims,
        podium=podium / n_sims,
        top10=top10 / n_sims,
        mean_points=pts_sum / n_sims,
        mean_rank=rank_sum / n_sims,
        n_sims=n_sims,
    )
