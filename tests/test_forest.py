import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from footpredict import forest


def synthetic(editions=(2022, 2023, 2024, 2025), n=30, seed=0):
    """Fake editions where points share is driven by goals and a Champions League win."""
    rng = np.random.default_rng(seed)
    rows = []
    for ed in editions:
        goals = rng.integers(0, 40, n)
        won_cl = (rng.random(n) < 0.15).astype(int)
        score = goals + 25 * won_cl + rng.normal(0, 2, n)
        share = np.exp(score / 8)
        share /= share.sum()
        for i in range(n):
            rows.append({
                "edition": ed, "player": f"p{ed}_{i}", "points_share": share[i],
                "goals": goals[i], "won_champions_league": won_cl[i], "position": "Attack",
                **{c: 0.0 for c in forest.NUMERIC if c not in ("goals", "won_champions_league")},
            })
    return forest.prepare(pd.DataFrame(rows))


def test_backtest_beats_chance_on_learnable_signal():
    res = forest.backtest(synthetic())
    rf = res[res["model"] == "random forest"]
    assert len(rf) == 4
    assert rf["spearman"].mean() > 0.6


def test_win_and_podium_probabilities_sum_per_edition():
    df = synthetic()
    pred = forest.fit_predict(df[df["edition"] < 2025], df[df["edition"] == 2025]).table
    assert pred["win_prob"].sum() == pytest.approx(1.0)
    assert pred["podium_prob"].sum() == pytest.approx(3.0)


def test_past_only_skips_editions_without_two_training_years():
    res = forest.backtest(synthetic(), past_only=True)
    assert sorted(res["edition"].unique()) == [2024, 2025]
