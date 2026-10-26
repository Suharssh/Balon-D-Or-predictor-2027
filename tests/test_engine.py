from pathlib import Path

import numpy as np
import pytest

from footpredict.history import check_edition, load_results
from footpredict.odds import load_odds
from footpredict.plackett_luce import fit_linear, log_likelihood, sample_top_k
from footpredict.simulate import simulate
from footpredict.spec import JuryRules, find_spec, load_spec
from footpredict.tally import tally

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
TOP10 = JuryRules(voters=3, points=(15, 12, 10, 8, 7, 5, 4, 3, 2, 1))


def test_all_specs_load():
    for path in SPECS.glob("*.toml"):
        spec = load_spec(path)
        assert spec.window_start < spec.window_end
        assert spec.jury is not None


def test_tally_ties_on_points_go_to_most_first_places():
    rules = JuryRules(voters=3, points=(3, 1))
    # Everyone ends on 3 points; B got there with 2nd places only, so B is last.
    standings = tally([["A", "B"], ["C", "B"], ["D", "B"]], rules)
    assert {s.points for s in standings} == {3}
    assert standings[-1].candidate == "B"


def test_tally_tiebreak_falls_through_to_second_places():
    rules = JuryRules(voters=3, points=(4, 2, 1))
    # X = 4 + 2 and Y = 1 + 4 + 1: both 6 points and one 1st place,
    # but X has a 2nd place, so X finishes ahead even though Y is listed first.
    standings = tally([["Y", "X", "B"], ["X", "A", "Y"], ["C", "D", "Y"]], rules)
    assert [(s.candidate, s.points) for s in standings[:2]] == [("X", 6), ("Y", 6)]


def test_tally_rejects_bad_ballots():
    with pytest.raises(ValueError):
        tally([["A", "A"]], TOP10)
    with pytest.raises(ValueError):
        tally([[str(i) for i in range(11)]], TOP10)


def test_history_matches_published_points_scales():
    rows = load_results(ROOT / "data/history/ballon_dor_men.csv")
    for year in sorted({r.year for r in rows}):
        spec = find_spec("ballon_dor_men", year, SPECS)
        assert check_edition([r for r in rows if r.year == year], spec) == []


def test_sample_top_k_is_a_valid_partial_ranking():
    rng = np.random.default_rng(1)
    out = sample_top_k(np.zeros((50, 8)), 5, rng)
    assert out.shape == (50, 5)
    assert all(len(set(row)) == 5 for row in out)


def test_sample_top_k_first_place_frequencies_match_softmax():
    rng = np.random.default_rng(2)
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    first = sample_top_k(np.broadcast_to(logits, (20000, 3)), 1, rng)[:, 0]
    freq = np.bincount(first, minlength=3) / len(first)
    assert np.allclose(freq, [0.6, 0.3, 0.1], atol=0.015)


def test_log_likelihood_of_full_ranking_with_equal_strengths():
    ll = log_likelihood(np.zeros(3), np.array([0, 1, 2]))
    assert ll == pytest.approx(np.log(1 / 6))


def test_fit_linear_recovers_sign_and_scale_of_weights():
    rng = np.random.default_rng(3)
    true_w = np.array([1.5, -0.8])
    feats, ranks = [], []
    for _ in range(40):
        X = rng.standard_normal((12, 2))
        ranks.append(sample_top_k(X @ true_w, 6, rng))
        feats.append(X)
    w = fit_linear(feats, ranks, l2=0.01)
    assert np.sign(w).tolist() == [1.0, -1.0]
    assert np.allclose(w, true_w, atol=0.4)


def test_simulate_strong_favourite_wins_and_points_are_conserved():
    rules = JuryRules(voters=50, points=(15, 12, 10, 8, 7, 5, 4, 3, 2, 1))
    names = [f"p{i}" for i in range(15)]
    logits = np.zeros(15)
    logits[0] = 4.0
    res = simulate(names, logits, rules, shock_sd=0.1, n_sims=200, seed=0)
    assert res.win[0] > 0.95
    assert res.mean_points.sum() == pytest.approx(50 * 67)
    assert res.win.sum() == pytest.approx(1.0)
    assert res.podium.sum() == pytest.approx(3.0)


def test_odds_field_row_and_normalisation(tmp_path):
    f = tmp_path / "odds.csv"
    f.write_text("player,decimal_odds,source,date\nA,2.0,x,2026-01-01\n*,10.0,x,2026-01-01\n")
    m = load_odds(f, ["A", "B", "C"], field_share=0.1)
    assert m.probs.sum() == pytest.approx(1.0)
    assert m.priced == [True, False, False]
    assert m.probs.tolist() == pytest.approx([0.9, 0.05, 0.05])


def test_odds_without_field_row_need_every_price(tmp_path):
    f = tmp_path / "odds.csv"
    f.write_text("player,decimal_odds\nA,2.0\n")
    with pytest.raises(ValueError):
        load_odds(f, ["A", "B"])


def test_odds_reject_unknown_player(tmp_path):
    f = tmp_path / "odds.csv"
    f.write_text("player,decimal_odds\nNobody,2.0\n")
    with pytest.raises(ValueError):
        load_odds(f, ["A"])
