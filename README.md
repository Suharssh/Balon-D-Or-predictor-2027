# footpredict

Predict jury-voted football awards by simulating the jury. v1 covers the men's Ballon d'Or.

The Ballon d'Or isn't decided by stats. 100 journalists each rank 10 players and the points are added up. So the model predicts how each voter fills in a ballot, scores the ballots with the award's real rules, and repeats that thousands of times to get win, podium and top-10 probabilities.

## Quick start

```bash
pip install -e '.[dev]'
pytest

footpredict check            # validate past results against each edition's rules
footpredict history 2025     # print a past final ranking
footpredict simulate 2026    # market baseline for the 2026 vote
```

Example output (odds snapshot from 22 Oct 2026):

```
player                      market     win  podium   top10     pts
Harry Kane                   62.4%   52.3%   87.5%   98.7%    1152
Lamine Yamal                 27.7%   30.3%   76.7%   97.1%    1000
Kylian Mbappé                 4.9%    9.6%   43.4%   85.3%     660
```

## How it works

- **Award specs** (`specs/<award>_<edition>.toml`): eligibility window, decider type, number of voters, points per ballot place, tie-break rule. Nothing award-specific is hard-coded, so adding The Best FIFA or the women's Ballon d'Or means writing a spec, not code.
- **Voter model** (`plackett_luce.py`): each voter ranks candidates with a Plackett-Luce model. `fit_linear` learns feature weights from observed rankings.
- **Simulator** (`simulate.py`): each simulated edition draws a shared shock per candidate (our uncertainty about how the whole jury sees a player), then samples every voter's ballot and scores it with `tally.py`, which applies the points and the "most 1st places, then 2nd places..." tie-break.
- **Market baseline** (`odds.py`): converts bookmaker odds into voter strengths and calibrates the shock size so simulated win odds match the market. Any feature-based model has to beat this.

## Data

| File | Contents | Source |
|---|---|---|
| `data/history/ballon_dor_men.csv` | Full 30-player results with points, 2022 to 2025 | Wikipedia edition pages (linked in each spec) |
| `data/shortlists/ballon_dor_men_2026.csv` | 2026 shortlist | ESPN |
| `data/odds/ballon_dor_men_2026.csv` | William Hill prices, 22 Oct 2026 | Squawka |

`footpredict check` confirms every edition's points add up to voters × points per ballot. That check is also how the voter counts (93, 92, 99, 100) and the points scale change (top 5 with 6-4-3-2-1 up to 2023, top 10 with 15-12-10-8-7-5-4-3-2-1 from 2024) were confirmed. Ranks are shared when points are equal.

## Roadmap

1. **Feature ingest** for each eligibility window: goals, assists and minutes (FBref basic tables; its Opta advanced stats were withdrawn in 2026), xG and xA for the top five leagues (Understat), trophies and tournament results (openfootball, Wikipedia).
2. **Feature model**: fit `fit_linear` on 2022 to 2025 and backtest each edition using only data from before the vote, against the market baseline.
3. **Per-country ballots**: France Football publishes votes by country. Using them would turn 4 rankings into about 400.
4. **More awards**: women's Ballon d'Or, The Best FIFA, stat-based awards (`decider.type = "stat"`).
