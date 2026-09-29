# Reproducibility and provenance audit — 2026 World Cup prediction project

Audit date: 2026-09-28. This project is a **historical forecast reconstruction**. The tournament is over; the archive is not a current forecast. All displayed model percentages are conditional on an imperfect historical model and are **not** validated probabilities of real-world outcomes.

## Source inventory and decisions

| Input artifact | What it represents | Audit decision |
|---|---|---|
| `team_ratings.ipynb` / Library `team_ratings(4).ipynb` | Elo fitted using public international match dataset | Keep original in `legacy/`; frozen ratings CSV is the executable input. The exact historical training-data snapshot was not archived. |
| `bracket_simulation.ipynb` / Library `bracket_simulation(3).ipynb` | 1,000-run illustrative **generic-seeded bracket** | Preserve separately as historical prototype, not a published-format bracket. |
| `champion_probabilities.csv` | Results from 1,000-run generic-seeded notebook | Preserve in `legacy/` only, do not conflate with official-slot outputs. |
| `world_cup_visual_bracket_simulator.py` | Later 600-run slot-constrained simulation | Reconstructed as portable `world_cup_simulation.py`; its exported CSVs can be reproduced exactly. |
| `wc26_visual_outputs*.zip` | Duplicated 600-run graphics/CSVs/script | Checked against extracted source; retain a single set of eight original PNGs and newly reproduced CSVs. |
| `argentina_world_cup_2026_math_deck*.html` | Presentation iterations that mix different model numbers | Reference only; **not** authoritative data. Some versions report 21.5%/19.3%, others 21.7%/21.2%. They also repeatedly describe a 1,000-run simulation while illustrating 600-run outputs. |
| `wc26_story_figures.zip` | Presentation narrative assets | Do not treat images as independent numerical evidence. |

## Reproduced historical baseline

The archived visual output corresponds to **600 simulated tournaments with seed 2026**, using a separate fixed seed **14** for the illustrative single bracket. It yields Argentina **21.50%** and Spain **19.33%** title frequency. The separately archived generic-seeded **1,000-run** notebook output yields Spain **21.7%** and Argentina **21.2%**. These numbers are not interchangeable. The 600-run results and their complete accompanying stage/standings/knockout CSVs were regenerated and compared to the archived output, column by column. This establishes code-to-artifact reproducibility only, not forecast calibration or accuracy.

## Methodology and validity issues

1. **Training-data provenance:** `team_ratings.ipynb` downloads a moving `results.csv` from `martj42/international_results/master` without storing the exact revision, raw snapshot, or match cutoff. Thus its pre-tournament training set and the optimized 27.09 / 30.77 / 27.41 / 37.61 K-factors cannot currently be independently reproduced from archived raw data. The frozen 48-team Elo file is the starting point for this package.
2. **Potential temporal leakage:** the training notebook contains no enforced pre-kickoff cutoff. Do not re-download its `master` branch in 2026 and call the resulting ratings pre-tournament.
3. **Optimizer evaluation:** K-factors are fit to in-sample rolling historical prediction errors; the code does not show a separately reserved, prospective holdout validation period. Its so-called Brier score is squared error on Elo *expected match score* (a 0/0.5/1 target), not a fully calibrated three-class win/draw/loss Brier score.
4. **Group match generation:** `match_probs()` heuristically assigns draws according to Elo gap. Score margins are sampled from fixed heuristic distributions, not estimated from observed goals. Home advantage at tournament venues is not modeled in bracket simulations.
5. **Tournament fidelity:** 12 groups of four and 24 top-two plus eight best-third qualifiers are represented. However the group and third-place tiebreakers use Elo after points, goal difference and goals scored rather than all FIFA criteria. Third-place slot assignment obeys eligible-group constraints but uses deterministic backtracking, **not FIFA Annex C's official 495-case table**. The old generic-seeding notebook is even further from the official bracket. Consequently the visualization is properly called *approximate-slot*, never *official*.
6. **Monte Carlo uncertainty:** 600 runs imply coarse title-probability steps of 1/600 ≈ 0.167 percentage points and noticeable sampling variability. For the archived Argentina frequency p=0.215, the approximate binomial standard error is sqrt(p(1-p)/600) ≈ 1.68 percentage points; this excludes model error. The Argentina–Spain difference in this run is much smaller than standalone sampling uncertainty.
7. **Actual outcome:** FIFA records Spain beating Argentina 1–0 in the July 19, 2026 final. This one realized tournament is consistent with either team having a nonzero pre-tournament title chance; a single winner cannot establish the model's calibration. A serious accuracy evaluation needs match-by-match frozen probability predictions and observed results, ideally plus pre-tournament backtests on multiple tournaments. Those inputs were not preserved in this project.

## Reproduction checks

* `python world_cup_simulation.py --no-plots` (from the package root) reproduces all eight historical output CSVs in `outputs/`, including the 600-run title and stage tables.
* `python -m unittest discover -s tests -v` runs probability, bracket-size, and regression tests.
* Existing PNGs in `outputs/` come from the original June 2026 visual package and correspond to the same archived run; we visually inspected the champion ranking, heatmap, bracket and story graphics for contextual consistency. They are legacy illustrations, not freshly fitted predictions.

## External references

* Historical match dataset: https://github.com/martj42/international_results
* FIFA tournament format: https://gpcustomersupportfwc2026.tickets.fifa.com/hc/en-gb/articles/28784798873117-9-What-is-the-format-for-the-FIFA-World-Cup-2026-tournament
* FIFA Round-of-32 match schedule: https://www.fifa.com/en/articles/view-the-fifa-world-cup-26-match-schedule
* FIFA 2026 tournament results: https://www.fifa.com/en/tournaments/mens/worldcup/2026/articles/knockout-stage-match-schedule-bracket
