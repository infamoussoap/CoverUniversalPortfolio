# Certified Approximation of Cover’s Universal Portfolio

Reproduction code for the November 2026 manuscript by James Chok and Domenic
Petzinna. Given daily asset price relatives, the code encloses the wealth of
Cover’s continuous Dirichlet mixture in interval arithmetic. One such
calculation per market can benchmark multiple allocation strategies or test
whether equal weighting is growth-optimal. The default pair-series evaluation
intersects the signed cubic-moment interval with the integrated bound
$|I(J)| \le I(C)$ at every checkpoint.

## Quick start

Use CPython 3.12. The pinned package versions are those used for the paper.
The synthetic check requires only Python and the listed packages.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py smoke
```

The first check compares the Arb evidence interval with a Bernstein recursion
and independent Gauss–Legendre quadrature on 40 trading days. The second
replays a five-stock alternating market. Its interval first becomes
inconclusive on day 56, and direct integration first finds a gap exceeding
0.01 on day 78.

## Reproduce the paper

Install Git, a LaTeX distribution with the fonts used by Matplotlib, and
licensed MATLAB for the original OLPS toolbox replay. Set `MATLAB_BIN` if
`matlab` is outside your shell’s path. The paper used MATLAB R2023a Update 4
and an Apple M1 Pro. The Python runners support macOS and Linux.

```sh
python scripts/prepare_data.py all
python reproduce.py all
```

The first command fetches the pinned OLPS and FinOL data and checks their
hashes. It obtains Yahoo adjusted prices using yfinance. Yahoo can revise
its history. A matching download or the original two snapshots are required
for an exact replay of the Yahoo figures. Data preparation supports importing
the frozen stock and sector files, and reports a differing fingerprint before
the paper comparison. [Data sources, fingerprints and import commands](docs/data.md)

The second command verifies the inputs, runs all numerical stages
sequentially with one computational thread, generates the six PDF figures,
and checks the summaries against the archived paper values. Allow several
hours for a full run. On the original machine, one evidence pass over the
403-stock panel took about 48 minutes, and the timing study takes three
measurements. Data preparation and the full run can be split by stage.

| Paper result | Command from the repository root | Generated files |
| --- | --- | --- |
| §3.5, Table 1, Figures 1–2 | `python output/cover_revision_20261121/run_benchmarks.py examples` | `table1_width.csv`, `aapl_msft_pair_series.pdf`, `fig2_50stock.pdf` |
| §4, Tables 2–5, Figures 3–4, regret and extreme-day checks | `python reproduce.py benchmark` | `tableB.csv`, `tableC.csv`, `tableD.csv`, `bench_cost.csv`, `bench_wealth.pdf`, `bench_scaling.pdf`, `exact_sets_all.csv`, `finol_nyse_exact_sets.csv`, `finol_nyse_empty_set_control.csv` |
| §5, Tables 6–7, Figures 5–6, calibration and planted advantage | `python reproduce.py inference` | `5a_summary.csv`, `5a_subset_summary.csv`, `figure_A.pdf`, `figure_C.pdf`, `5a_null_summary.csv`, `5a_power_summary.csv` |
| Appendix C, alternating market, 12,600 closes | `python reproduce.py appendix` | `output/appendix/alternating.json` |

The benchmark also writes `bench_half_prior_gaps.csv` and `bench_half_prior_summary.csv` for Soft-Bayes, VB-FTRL and the MATLAB toolbox UP on the same six tapes, using symmetric Dirichlet(½) Cover wealth as the benchmark.

Sections 3–4 write to `output/cover_experiments_final/`. Section 5 writes
to `output/experiment_5a/`. Individual stage names, order, seeds and output
definitions are documented in the [benchmark guide](docs/benchmarks.md)
and [inference guide](docs/inference.md).

## Comparing a run with the reported results

```sh
python reproduce.py verify
```

The verifier compares generated CSVs with independent records in
[`expected/`](expected/). It checks decision counts, signed wealth gaps,
regret bounds, interval widths and the existence of the paper figures.
Numerical estimates allow small floating-point differences. Elapsed times
and memory are measured again on the reproducing machine. References for an
individual stage can be checked with, for example,
`python scripts/check_results.py examples`.

## Using the certificate on your own data

```python
from flint import ctx
from particle_universal.tilted_interval_cover import TiltedCover

ctx.prec = 128
monitor = TiltedCover(dimension=2, alpha=1, exact_pairs=True)
for row in [(1.02, 0.99), (0.98, 1.01), (1.04, 1.00)]:
    monitor.update(row, [0.6, 0.4])
    lo, hi = monitor.evidence()
    gap = ((lo - monitor.log_played).lower(),
           (hi - monitor.log_played).upper())
    print(gap)
```

The inputs are gross returns and the weights held **before** each return.
`evidence()` encloses log Cover wealth in units of the equal-weight strategy.
Subtracting the played log wealth gives the certified gap
`log(Cover wealth) − log(strategy wealth)`. A positive lower endpoint
certifies that Cover is ahead. The streaming monitor above takes strictly
positive price relatives. The [product-tilt implementation](particle_universal/product_tilt.py)
handles zero entries on rows with a positive payoff somewhere.

For the exact-moment regression tests, install pytest and run
`python -m pytest -q` from this directory.
