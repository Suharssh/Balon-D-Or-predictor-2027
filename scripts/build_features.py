"""Build per-candidate features for each Ballon d'Or edition from transfermarkt-datasets.

Source: https://github.com/dcaribou/transfermarkt-datasets (free, built from
Transfermarkt, one row per player per game). Snapshot ends 6 Jul 2026.

Runs in GitHub Actions (see .github/workflows/build-features.yml) because the
dataset host is not reachable from every environment. Writes:

  data/features/ballon_dor_men.csv   one row per (edition, candidate)
  data/players/tm_matches.csv        how each candidate name was matched, for review

Every stat is computed over the edition's eligibility window from specs/.
"""

from __future__ import annotations

import csv
import re
import sys
import tomllib
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["players", "appearances", "games", "competitions", "player_valuations", "clubs"]

# Curated competition names -> transfermarkt competition_id.
COMPETITION_IDS = {
    "UEFA Champions League": "CL",
    "FIFA World Cup": "FIWC",
    "UEFA European Championship": "EURO",
    "Copa América": "COPA",
    "Africa Cup of Nations": "AFCN",
    "AFC Asian Cup": "AFAC",
}
COUNTRY_ALIASES = {"Ivory Coast": "Cote d'Ivoire"}  # curated name -> transfermarkt citizenship
INTERNATIONAL = {"FIFA World Cup", "UEFA European Championship", "Copa América", "Africa Cup of Nations", "AFC Asian Cup"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def load_tables(cache: Path) -> dict[str, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in TABLES:
        path = cache / f"{t}.csv.gz"
        if not path.exists():
            print(f"downloading {t}", flush=True)
            # The host rejects Python's default user agent.
            req = urllib.request.Request(f"{BASE_URL}/{t}.csv.gz", headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
        out[t] = pd.read_csv(path, low_memory=False)
        print(f"{t}: {len(out[t]):,} rows; columns {list(out[t].columns)}", flush=True)
    for t in ("appearances", "games", "player_valuations"):
        out[t]["date"] = pd.to_datetime(out[t]["date"])
    return out


def load_editions() -> dict[int, tuple[pd.Timestamp, pd.Timestamp, int]]:
    eds = {}
    for p in sorted((ROOT / "specs").glob("ballon_dor_men_*.toml")):
        with open(p, "rb") as f:
            s = tomllib.load(f)
        d = s["decider"]
        eds[s["edition"]] = (pd.Timestamp(s["window"]["start"]), pd.Timestamp(s["window"]["end"]), d["voters"] * sum(d["points"]))
    return eds


def load_candidates() -> pd.DataFrame:
    hist = pd.read_csv(ROOT / "data/history/ballon_dor_men.csv")
    rows = [hist[["year", "player", "points"]].rename(columns={"year": "edition"})]
    for p in sorted((ROOT / "data/shortlists").glob("ballon_dor_men_*.csv")):
        ed = int(p.stem.rsplit("_", 1)[1])
        if ed not in set(hist["year"]):
            sl = pd.read_csv(p)
            rows.append(pd.DataFrame({"edition": ed, "player": sl["player"], "points": pd.NA}))
    return pd.concat(rows, ignore_index=True)


def match_players(cands: pd.DataFrame, players: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Map candidate names to transfermarkt player_id. Overrides (name as spelled on
    Transfermarkt) win; then exact
    accent-insensitive name; then token containment. Ties go to the player with
    the most minutes in 2021-2026 (filters out namesakes in lower leagues)."""
    overrides_path = ROOT / "data/players/tm_overrides.csv"
    overrides = {}
    if overrides_path.exists():
        with open(overrides_path, encoding="utf-8") as f:
            overrides = {r["player"]: r["tm_name"] for r in csv.DictReader(f)}

    recent = apps[apps["date"] >= "2021-08-01"].groupby("player_id")["minutes_played"].sum()
    p = players[["player_id", "name", "date_of_birth", "position", "country_of_citizenship"]].copy()
    p["minutes_recent"] = p["player_id"].map(recent).fillna(0)
    p = p[p["minutes_recent"] > 0]
    p["key"] = p["name"].map(norm)
    p["tokens"] = p["key"].str.split().map(set)

    rows = []
    for name in sorted(set(cands["player"])):
        if name in overrides:
            hit = p[p["key"] == norm(overrides[name])]
            how = "override"
        else:
            k = norm(name)
            toks = set(k.split())
            hit = p[p["key"] == k]
            how = "exact"
            if hit.empty:
                hit = p[p["tokens"].map(lambda t: toks <= t or (t <= toks and len(t) > 0))]
                how = "tokens"
        if hit.empty:
            rows.append({"player": name, "player_id": None, "tm_name": None, "how": "UNMATCHED", "n_candidates": 0})
            continue
        best = hit.sort_values("minutes_recent", ascending=False).iloc[0]
        rows.append({
            "player": name, "player_id": int(best["player_id"]), "tm_name": best["name"],
            "date_of_birth": best["date_of_birth"], "position": best["position"],
            "country": best["country_of_citizenship"], "how": how, "n_candidates": len(hit),
        })
    return pd.DataFrame(rows)


def find_competitions(comps: pd.DataFrame) -> dict[str, list[str]]:
    known = dict(zip(comps["competition_id"], comps["name"]))
    found = {}
    for label, cid in COMPETITION_IDS.items():
        if cid not in known:
            raise SystemExit(f"competition {cid} ({label}) is not in the dataset")
        print(f"competition {label!r} -> {cid} ({known[cid]})", flush=True)
        found[label] = [cid]
    return found


_warned: set = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"WARNING: {msg}", flush=True)


def winner_team_id(label, final_date, winner, comp_ids, games, start, end):
    """Team id of the curated winner, found among teams that played that competition in the window."""
    g = games[games["competition_id"].isin(comp_ids) & games["date"].between(start, end)]
    names = pd.concat([
        g[["home_club_id", "home_club_name"]].set_axis(["id", "name"], axis=1),
        g[["away_club_id", "away_club_name"]].set_axis(["id", "name"], axis=1),
    ]).dropna().drop_duplicates()
    if names.empty:
        _warn(f"{label} {final_date}: no games in the window")
        return None
    w = norm(winner)
    hit = names[names["name"].map(norm).map(lambda n: n == w or n.startswith(w + " ") or w in n.split(" fc")[0])]
    ids = hit["id"].unique()
    if len(ids) != 1:
        _warn(f"{label} {final_date}: winner {winner!r} matched {hit.values.tolist()}")
        return None
    return int(ids[0])


def main() -> int:
    t = load_tables(ROOT / ".cache/transfermarkt")
    apps, games, players, comps, vals = t["appearances"], t["games"], t["players"], t["competitions"], t["player_valuations"]
    editions = load_editions()
    cands = load_candidates()

    matches = match_players(cands, players, apps)
    (ROOT / "data/players").mkdir(parents=True, exist_ok=True)
    matches.to_csv(ROOT / "data/players/tm_matches.csv", index=False)
    unmatched = matches[matches["player_id"].isna()]["player"].tolist()
    print(f"matched {len(matches) - len(unmatched)}/{len(matches)} names; unmatched: {unmatched}", flush=True)
    ids = dict(zip(matches["player"], matches["player_id"]))

    comp_ids = find_competitions(comps)
    finals = pd.read_csv(ROOT / "data/trophies/major_finals.csv")
    comp_types = comps.set_index("competition_id")["type"].to_dict() if "type" in comps else {}

    # Knockout rounds: anything that is not a group stage, league phase or numbered matchday.
    games["knockout"] = ~games["round"].astype(str).str.lower().str.contains(r"group|league phase|matchday|^\d", regex=True)
    apps = apps.merge(games[["game_id", "round", "knockout", "home_club_id", "away_club_id",
                             "home_club_goals", "away_club_goals"]], on="game_id", how="left")
    cl_ids = set(comp_ids["UEFA Champions League"])
    intl_ids = {c for k in INTERNATIONAL for c in comp_ids[k]}

    prof_country = players.set_index("player_id")["country_of_citizenship"].to_dict()
    rows = []
    credited: dict[tuple, list[str]] = {}
    for _, c in cands.iterrows():
        ed, name = int(c["edition"]), c["player"]
        start, end, total_pts = editions[ed]
        pid = ids.get(name)
        row = {"edition": ed, "player": name, "points": c["points"],
               "points_share": (c["points"] / total_pts) if pd.notna(c["points"]) else None,
               "player_id": pid}
        if pid is None or pd.isna(pid):
            rows.append(row)
            continue
        a = apps[(apps["player_id"] == pid) & apps["date"].between(start, end)]
        is_cl = a["competition_id"].isin(cl_ids)
        is_intl = a["competition_id"].isin(intl_ids)
        is_league = a["competition_id"].map(comp_types).eq("domestic_league") if comp_types else False
        ga = a["goals"] + a["assists"]
        row.update({
            "minutes": a["minutes_played"].sum(),
            "apps": len(a),
            "goals": a["goals"].sum(),
            "assists": a["assists"].sum(),
            "ga_per90": 90 * ga.sum() / max(a["minutes_played"].sum(), 1),
            "league_ga": ga[is_league].sum(),
            "cl_minutes": a.loc[is_cl, "minutes_played"].sum(),
            "cl_ga": ga[is_cl].sum(),
            "cl_knockout_ga": ga[is_cl & a["knockout"]].sum(),
            "intl_tournament_minutes": a.loc[is_intl, "minutes_played"].sum(),
            "intl_tournament_ga": ga[is_intl].sum(),
        })

        # Trophies from the curated finals table.
        won_cl = won_intl = 0
        for _, f in finals[finals["edition"] == ed].iterrows():
            ids_f = comp_ids.get(f["competition"], [])
            team = winner_team_id(f["competition"], f["final_date"], f["winner"], ids_f, games, start, end)
            has_apps = bool(apps["competition_id"].isin(ids_f).loc[apps["date"].between(start, end)].any())
            if team is None or not has_apps:
                # Older tournaments are missing from the dataset (no games, or games
                # without player appearances). For national teams,
                # fall back to citizenship: every shortlisted player from a winning
                # nation in 2022-2025 was in that squad (checked by hand).
                if f["competition"] in INTERNATIONAL and norm(COUNTRY_ALIASES.get(f["winner"], f["winner"])) == norm(prof_country.get(pid, "")):
                    credited.setdefault((ed, f["competition"], f["winner"], "by citizenship"), []).append(name)
                    won_intl = 1
                elif f["competition"] not in INTERNATIONAL:
                    _warn(f"{f['competition']} {f['final_date']}: no appearance data; nobody credited")
                continue
            played = a[a["competition_id"].isin(ids_f) & (a["player_club_id"] == team)]
            if len(played):
                credited.setdefault((ed, f["competition"], f["winner"], team), []).append(name)
                if f["competition"] in INTERNATIONAL:
                    won_intl = 1
                else:
                    won_cl = 1
        row["won_champions_league"] = won_cl
        row["won_international_tournament"] = won_intl

        # Domestic league title: main club's position in its last league game of the window.
        lg = a[is_league] if comp_types else a.iloc[0:0]
        won_league = 0
        if len(lg):
            club = lg.groupby("player_club_id")["minutes_played"].sum().idxmax()
            comp = lg[lg["player_club_id"] == club]["competition_id"].mode().iloc[0]
            cg = games[(games["competition_id"] == comp) & games["date"].between(start, end)
                       & ((games["home_club_id"] == club) | (games["away_club_id"] == club))].sort_values("date")
            if len(cg):
                last = cg.iloc[-1]
                pos = last["home_club_position"] if last["home_club_id"] == club else last["away_club_position"]
                won_league = int(pos == 1)
        row["won_league"] = won_league

        v = vals[(vals["player_id"] == pid) & (vals["date"] <= end)].sort_values("date")
        row["market_value_m"] = v["market_value_in_eur"].iloc[-1] / 1e6 if len(v) else None
        prof = players[players["player_id"] == pid].iloc[0]
        dob = pd.to_datetime(prof["date_of_birth"], errors="coerce")
        row["age"] = (end - dob).days / 365.25 if pd.notna(dob) else None
        row["position"] = prof["position"]
        rows.append(row)

    for (ed, comp, winner, team), names in sorted(credited.items()):
        print(f"{ed} {comp} won by {winner} (team {team}): {names}", flush=True)
    out = pd.DataFrame(rows).sort_values(["edition", "points"], ascending=[True, False])
    # Reputation: share of points in the previous edition (0 if not in the top 30).
    prev = out[["edition", "player", "points_share"]].assign(edition=lambda d: d["edition"] + 1)
    out = out.merge(prev.rename(columns={"points_share": "prev_points_share"}), on=["edition", "player"], how="left")
    out["prev_points_share"] = out["prev_points_share"].fillna(0.0)
    (ROOT / "data/features").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data/features/ballon_dor_men.csv", index=False)
    print(out.groupby("edition")[["minutes", "won_champions_league", "won_international_tournament", "won_league"]].sum())
    print(f"wrote {len(out)} rows", flush=True)
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
