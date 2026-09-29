# 2026 World Cup Predictions — reproducible historical archive

**Project by Hammon Dutra.** This repository reconstructs the original June 2026 Elo-based World Cup experiment. It is an educational historical prediction project, **not a current forecast or a validated official-bracket model**. Read [AUDIT.md](AUDIT.md) before interpreting its charts.

## What was actually produced?

The later visual simulator ran **600 Monte Carlo tournaments (seed 2026)**. In that run Argentina won **129 / 600 = 21.50%**, Spain **116 / 600 = 19.33%**, and France **75 / 600 = 12.50%**. The older generic-seeded 1,000-run notebook gave *different* numbers (Spain 21.7%, Argentina 21.2%). We retain both but do not combine them. FIFA records Spain beating Argentina 1–0 in the real final.

The displayed percentages are **model-specific historical simulation frequencies**, not demonstrated real-world probabilities. The frozen ratings are archived, but the exact training dataset revision, cutoff, and independent validation were not preserved. The backtracking third-place assignment does not reproduce FIFA Annex C.

## Start here

- [Explanatory notebook](world_cup_explained.ipynb): data lineage, equations, model assumptions, one complete simulation, stage probabilities, reproducibility checks and limitations.
- [Standalone Python script](world_cup_simulation.py): identical importable tournament mechanics, command-line options, tables and eight chart generators.
- [Reproducibility and methodology audit](AUDIT.md): source reconciliation, discovered discrepancies, data limitations and real-results context.
- [Frozen team ratings](data/optimized_elo_2026.csv): exact 48-team input; do not regenerate it using the current upstream `master` and label the output pre-tournament.
- [Historical results tables](outputs/monte_carlo_champion_probabilities.csv): replicated 600-run output, with other outputs and original figures in the same folder.
- [Original research notebooks](legacy/): preserved source artifacts, not the preferred reproduction path.

## Local usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python world_cup_simulation.py --no-plots  # exact archived 600-run numerical replay
python -m unittest discover -s tests -v
jupyter notebook world_cup_explained.ipynb
```

To generate all charts again, omit `--no-plots`. To explore simulation noise without overwriting the original archives, **copy this directory first** and then use `--simulations 10000 --seed 42`.

## Historical workflow

1. Use historic international match results to update Elo ratings, with different match-type K-factors and a goal-difference multiplier. Freeze 48 team ratings in CSV (archived; raw training revision missing).
2. Convert Elo score expectations into heuristic win/draw/loss probabilities and separately sample plausible scorelines.
3. Simulate 12 groups, select 24 top-two qualifiers and eight best third-place teams.
4. Construct a published-slot-compatible **approximation** of the knockout bracket. FIFA Annex C is *not* implemented.
5. Repeat the tournament with seeded random number generation, count advancement, and divide by total simulation count. Generate CSVs and graphics.

![Historical title simulation frequency](outputs/06_champion_probability_ranking.png)

## Key cautions

The earlier `bracket_simulation.ipynb` built a generic seeded bracket; its probabilities are not directly comparable to the later approximate-slot simulator. Historical charts should be read as records of what the model generated, not evidence the model was calibrated. See [AUDIT.md](AUDIT.md) for the complete discrepancy log and FIFA references.
