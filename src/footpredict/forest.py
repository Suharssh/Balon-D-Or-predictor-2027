"""Random forest model of Ballon d'Or points share.

Each row is one shortlisted player in one edition; the target is the share of
all points handed out that edition (comparable across the top-5 and top-10
ballot eras). Features come from data/features/ballon_dor_men.csv, built by
scripts/build_features.py.

International tournament minutes and goals are in the features file but not
used: the dataset only has player appearances for the newest tournaments, so
they would be zero for every earlier edition. Tournament wins are used instead.

Win probability is the share of trees that rank a player first in their
edition, so it reflects how much the forest disagrees with itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

NUMERIC = [
    "minutes", "goals", "assists", "ga_per90", "league_ga",
    "cl_minutes", "cl_ga", "cl_knockout_ga",
    "won_champions_league", "won_international_tournament", "won_league",
    "market_value_m", "age", "prev_points_share",
]
POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]
FEATURES = NUMERIC + [f"pos_{p.lower()}" for p in POSITIONS]
BASELINES = {"market value": "market_value_m", "goals + assists": "_ga", "last year's points": "prev_points_share"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in POSITIONS:
        df[f"pos_{p.lower()}"] = (df["position"] == p).astype(int)
    df["market_value_m"] = df["market_value_m"].fillna(df.groupby("edition")["market_value_m"].transform("median"))
    df[NUMERIC] = df[NUMERIC].fillna(0.0)
    df["_ga"] = df["goals"] + df["assists"]
    return df


def make_model(seed: int = 0) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=-1)


@dataclass
class Prediction:
    table: pd.DataFrame        # edition, player, pred_share, win_prob, podium_prob
    model: RandomForestRegressor


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> Prediction:
    model = make_model(seed).fit(train[FEATURES], train["points_share"])
    out = test[["edition", "player"]].copy()
    out["pred_share"] = model.predict(test[FEATURES])

    per_tree = np.stack([t.predict(test[FEATURES].to_numpy()) for t in model.estimators_])  # (trees, rows)
    win = np.zeros(len(test))
    podium = np.zeros(len(test))
    for ed in out["edition"].unique():
        idx = np.flatnonzero(out["edition"].to_numpy() == ed)
        order = np.argsort(-per_tree[:, idx], axis=1)
        np.add.at(win, idx[order[:, 0]], 1)
        for k in range(min(3, len(idx))):
            np.add.at(podium, idx[order[:, k]], 1)
    out["win_prob"] = win / len(model.estimators_)
    out["podium_prob"] = podium / len(model.estimators_)
    return Prediction(out.sort_values(["edition", "pred_share"], ascending=[True, False]), model)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def score_edition(actual: pd.Series, predicted: pd.Series, players: pd.Series) -> dict:
    """Compare a predicted score per player with the real points."""
    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    names = players.to_numpy()
    # Break ties in the prediction alphabetically, never by the input order
    # (the features file is sorted by real points, which would leak the answer).
    top_a = np.argsort(-a)
    top_p = np.lexsort((names, -p))
    return {
        "actual_winner": names[top_a[0]],
        "predicted_winner": names[top_p[0]],
        "winner_hit": bool(names[top_a[0]] == names[top_p[0]]),
        "top3_overlap": len(set(names[top_a[:3]]) & set(names[top_p[:3]])),
        "spearman": _spearman(a, p),
    }


def backtest(df: pd.DataFrame, past_only: bool = False, seed: int = 0) -> pd.DataFrame:
    """Hold out each labelled edition in turn.

    By default trains on every other labelled edition (leave-one-year-out).
    With ``past_only`` only earlier editions are used, which is the honest
    setup but leaves just two test years (2024, 2025).
    """
    labelled = df[df["points_share"].notna()]
    rows = []
    for ed in sorted(labelled["edition"].unique()):
        train = labelled[(labelled["edition"] < ed) if past_only else (labelled["edition"] != ed)]
        if train["edition"].nunique() < 2:
            continue
        test = labelled[labelled["edition"] == ed]
        pred = fit_predict(train, test, seed).table.set_index("player").loc[test["player"]]
        rows.append({"edition": ed, "model": "random forest",
                     **score_edition(test["points_share"], pred["pred_share"], test["player"]),
                     "winner_prob": float(pred.loc[test.loc[test["points_share"].idxmax(), "player"], "win_prob"])})
        for label, col in BASELINES.items():
            if test[col].nunique() < 2:
                continue  # e.g. no previous-year points for the first edition
            rows.append({"edition": ed, "model": label, **score_edition(test["points_share"], test[col], test["player"])})
    return pd.DataFrame(rows)


def importances(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    labelled = df[df["points_share"].notna()]
    model = make_model(seed).fit(labelled[FEATURES], labelled["points_share"])
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
