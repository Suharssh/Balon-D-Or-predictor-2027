"""Command-line interface: ``footpredict <command>``."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .history import check_edition, load_results
from .odds import calibrate_shock, load_odds
from .simulate import simulate
from .spec import find_spec


def _read_shortlist(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["player"] for r in csv.DictReader(f)]


def cmd_check(args: argparse.Namespace) -> int:
    rows = load_results(args.results)
    years = sorted({r.year for r in rows})
    failed = False
    for y in years:
        spec = find_spec(args.award, y, args.specs)
        problems = check_edition([r for r in rows if r.year == y], spec)
        status = "ok" if not problems else "FAIL"
        print(f"{y}: {status}  ({spec.jury.voters} voters, points {list(spec.jury.points)})")
        for p in problems:
            print(f"  - {p}")
        failed |= bool(problems)
    return 1 if failed else 0


def cmd_history(args: argparse.Namespace) -> int:
    rows = [r for r in load_results(args.results) if r.year == args.year]
    if not rows:
        print(f"no results for {args.year}", file=sys.stderr)
        return 1
    for r in sorted(rows, key=lambda r: r.rank)[: args.top]:
        print(f"{r.rank:>3}  {r.player:<28} {r.points:>5}  {r.club}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    spec = find_spec(args.award, args.edition, args.specs)
    shortlist = _read_shortlist(args.shortlist)
    market = load_odds(args.odds, shortlist, field_share=args.field_share)

    if args.shock is None:
        shock, kl = calibrate_shock(market, spec.jury, seed=args.seed)
        print(f"calibrated jury shock sd = {shock} (KL to market {kl:.3f})")
    else:
        shock = args.shock

    res = simulate(shortlist, np.log(market.probs), spec.jury, shock_sd=shock, n_sims=args.sims, seed=args.seed)
    priced = dict(zip(market.candidates, market.priced))
    market_p = dict(zip(market.candidates, market.probs))

    print(f"{spec.award} {spec.edition}: {res.n_sims} simulated juries of {spec.jury.voters}")
    print(f"odds: {market.source}")
    print(f"{'player':<26} {'market':>7} {'win':>7} {'podium':>7} {'top10':>7} {'pts':>7}")
    for row in res.table()[: args.top]:
        c = row["candidate"]
        flag = "" if priced[c] else " *"
        print(
            f"{c:<26} {market_p[c]:>7.1%} {row['win']:>7.1%} {row['podium']:>7.1%} "
            f"{row['top10']:>7.1%} {row['mean_points']:>7.0f}{flag}"
        )
    if not all(market.priced):
        print(f"* no individual price; shares {args.field_share:.0%} with the rest of the field")
    return 0


def _load_rf_features(path: Path):
    try:
        import pandas as pd
        from . import forest
    except ImportError:
        print("the random forest needs the ml extra: pip install -e '.[ml]'", file=sys.stderr)
        raise SystemExit(2)
    return forest, forest.prepare(pd.read_csv(path))


def cmd_rf_backtest(args: argparse.Namespace) -> int:
    forest, df = _load_rf_features(args.features)
    res = forest.backtest(df, past_only=args.past_only, seed=args.seed)
    setup = "trained on earlier editions only" if args.past_only else "leave-one-edition-out"
    print(f"backtest ({setup})")
    print(f"{'edition':<8} {'model':<20} {'predicted winner':<22} {'actual winner':<22} {'hit':>4} {'top3':>5} {'rank corr':>9}")
    for r in res.itertuples():
        print(f"{r.edition:<8} {r.model:<20} {r.predicted_winner:<22} {r.actual_winner:<22} "
              f"{'yes' if r.winner_hit else 'no':>4} {r.top3_overlap:>4}/3 {r.spearman:>9.2f}")
    print("\naverages")
    summary = res.groupby("model")[["winner_hit", "top3_overlap", "spearman"]].mean()
    for model, r in summary.sort_values("spearman", ascending=False).iterrows():
        print(f"  {model:<20} winners {r.winner_hit:.0%}  top-3 overlap {r.top3_overlap:.2f}/3  rank corr {r.spearman:.2f}")
    return 0


def cmd_rf_predict(args: argparse.Namespace) -> int:
    forest, df = _load_rf_features(args.features)
    train = df[df["points_share"].notna() & (df["edition"] < args.edition)]
    test = df[df["edition"] == args.edition]
    if test.empty:
        print(f"no features for {args.edition}", file=sys.stderr)
        return 1
    spec = find_spec(args.award, args.edition, args.specs)
    pred = forest.fit_predict(train, test, seed=args.seed)
    total = spec.jury.voters * spec.jury.points_per_ballot
    eds = sorted(train["edition"].unique())
    print(f"random forest trained on {eds[0]}-{eds[-1]} ({len(train)} player-seasons)")
    print(f"{'player':<26} {'pred pts':>8} {'win':>6} {'podium':>7}")
    for r in pred.table.head(args.top).itertuples():
        print(f"{r.player:<26} {r.pred_share * total:>8.0f} {r.win_prob:>6.1%} {r.podium_prob:>7.1%}")
    print("\nmost important features")
    for name, v in forest.importances(df[df["edition"] < args.edition], seed=args.seed).head(6).items():
        print(f"  {name:<30} {v:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="footpredict", description=__doc__)
    p.add_argument("--specs", type=Path, default=Path("specs"))
    p.add_argument("--award", default="ballon_dor_men")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="validate historical results against each edition's spec")
    c.add_argument("--results", type=Path, default=Path("data/history/ballon_dor_men.csv"))
    c.set_defaults(func=cmd_check)

    h = sub.add_parser("history", help="print a past edition's final ranking")
    h.add_argument("year", type=int)
    h.add_argument("--results", type=Path, default=Path("data/history/ballon_dor_men.csv"))
    h.add_argument("--top", type=int, default=10)
    h.set_defaults(func=cmd_history)

    s = sub.add_parser("simulate", help="simulate an edition from betting odds (market baseline)")
    s.add_argument("edition", type=int)
    s.add_argument("--shortlist", type=Path, help="defaults to data/shortlists/<award>_<edition>.csv")
    s.add_argument("--odds", type=Path, help="defaults to data/odds/<award>_<edition>.csv")
    s.add_argument("--field-share", type=float, default=0.05,
                   help="total win probability given to players without their own price (default 0.05)")
    s.add_argument("--sims", type=int, default=5000)
    s.add_argument("--shock", type=float, help="jury shock sd; calibrated to the market if omitted")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_simulate)

    features = Path("data/features/ballon_dor_men.csv")
    b = sub.add_parser("rf-backtest", help="backtest the random forest against simple baselines")
    b.add_argument("--features", type=Path, default=features)
    b.add_argument("--past-only", action="store_true", help="train only on editions before the one being tested")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_rf_backtest)

    r = sub.add_parser("rf-predict", help="predict an edition with the random forest")
    r.add_argument("edition", type=int)
    r.add_argument("--features", type=Path, default=features)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(func=cmd_rf_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        stem = f"{args.award}_{args.edition}.csv"
        args.shortlist = args.shortlist or Path("data/shortlists") / stem
        args.odds = args.odds or Path("data/odds") / stem
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
