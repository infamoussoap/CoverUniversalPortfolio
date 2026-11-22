# Testing equal weighting

Section 5 uses a uniform Dirichlet prior, a rejection threshold of 20 and
checks at every trading close. Each e-value measures Cover wealth in units
of daily equal-weight rebalancing on the same assets. Figure 5 and Table 6
use the nine sector ETFs and seven secondary markets. Figure 6 and Table 7
use overlapping subsets of the 50-stock series.

From the repository root, prepare the inputs and run the study with

```sh
python scripts/prepare_data.py olps
python scripts/prepare_data.py yahoo
python output/experiment_5a/run.py all
```

`python output/experiment_5a/run.py smoke` runs a data-free test of the
binary64 screen, Arb certificate, independent exact recursion and quadrature
on a deterministic 40-day example. The full study runs sequentially and sets
the supported numerical libraries to one thread. Use the same environment and
input vintage throughout a reproduction. The historical results condition on
the stored binary64 values. Revisions to vendor-adjusted prices can change
the outcome of redownloads.

## Stages and outputs

Run an individual stage with `python output/experiment_5a/run.py STAGE`.
Files below are relative to `output/experiment_5a/`. For a full reproduction
the stages are executed in the order shown.

| Stage | Calculation | Outputs |
| --- | --- | --- |
| `markets` | Inventory, eight daily Arb paths and terminal e-BH test | `5a_markets.csv`, `5a_extremes.csv`, `5a_paths.csv`, `5a_summary.csv` (Table 6) |
| `exact` | Observed and drift-planted XLK–XLU, checked by Bernstein recursion and 3,396-node Gauss–Legendre quadrature | `5a_exact_d2.csv` |
| `null` | Ten full Arb paths per market, then 1,000 sector and 200 DJIA permutations | Cached replay records and Arb validation paths |
| `power` | 200 sector replays at each annual gap, five gaps from 5% through 80%. Two smallest gaps with fewer than half crossing receive extended 25,000-day follow-up | Cached stopping decisions |
| `audit` | Independent bound on the true mixture over every prefix of the null and plant paths, including the extended replays from `power` | `5a_threshold_audit.csv` |
| `summaries` | Assemble null and power results, exact-binomial intervals, vertex-optimum checks, bootstrap detection quantiles and the terminal replay-0 check | `5a_null.csv`, `5a_null_summary.csv`, `5a_power.csv`, `5a_power_summary.csv`, `5a_winner.csv` |
| `subsets` | Freeze the samples, compute terminal enclosures, check every pair by independent integration and audit thresholds. The full-panel row uses the earlier `markets` output | `5a_stock_ranking.csv`, `5a_subset_design.csv`, `5a_subsets.csv`, `5a_subset_summary.csv` (Table 7) |
| `figures` | Typeset four plots from the generated CSVs, with Matplotlib's LaTeX text rendering | `figure_A.pdf` (Figure 5), `figure_C.pdf` (Figure 6), `figure_B.pdf` (supplementary power), `figure_D.pdf` (supplementary variance ratios) |

The sector dates cover 4 January 1999 through 31 December 2025. The
50-stock history contains 6,539 returns. Both Yahoo tapes use adjusted
closes. The six OLPS matrices have row indices and source-supplied periods.
Their published files omit individual trading dates. The inventory records
the input-file and return-matrix SHA-256 values before fitting.

## Design details

- PCG64 seed `[583211, d, replay, pass]` gives a fresh independent
  permutation for each date. The sector and DJIA nulls keep every day's
  cross-sectional return distribution. Their asset labels are exchangeable
  conditional on the row. The primary sector test monitors the realized path
  at every close. The eight-market e-BH comparison uses terminal e-values.
- The plant multiplies each relabelled relative by the exponential of its
  assigned daily log drift, then renormalizes by the new equal-weight
  return. The annual drifts are equally spaced from −g/2 to +g/2.
  Seed 583210 gives increasing sector drift order XLB, XLV, XLE, XLI, XLY,
  XLK, XLP, XLF and XLU. All gaps share the first 200 permutation streams.
  Extensions preserve the initial pass and draw fresh permutations on each
  subsequent pass. The two-asset plant uses XLK and XLU in that order.
- Null mixture maxima are numerical estimates using two independently
  scrambled 512-point Sobol nets with a degree-four control polynomial.
  The seeds are `[583213, d, j]`, for j = 0, 1. The reported rejection
  decisions require an Arb lower endpoint at or above the threshold.
  Binary64 screens nominate candidate days within 0.05 log units of the
  threshold. The first ten null replicas per market are also checked
  exhaustively in Arb. All maxima exclude the initial endowment.
- At each of 3, 5, 10 and 20 stocks, PCG64 seed `[583214, d]` draws 200
  distinct subsets. The experiment also enumerates all 1,225 pairs and the
  full panel. The retrospective top-five label comes from terminal
  buy-and-hold wealth. The data overlap, so these summaries are descriptive.
  All pair endpoints are checked by polynomial-degree-exact quadrature on
  3,300 nodes and an independent Arb polynomial product with bounded tail.
  Figure C uses seed 583215 for the horizontal jitter.
- The replay-0 vertex projection is reported only when interval arithmetic
  verifies strictly negative derivatives away from the candidate single
  asset. It uses a common directional scale and takes the increasing branch
  of the approximation. Censored median times in the power summary have
  explicit flags. The bootstrap uses 4,000 resamples and seed 583212.
- The independent threshold audit uses `log(1+y) <= y` to bound the true
  mixture by the Dirichlet moment generating function of the cumulative
  score. It bounds a block of trading dates with coordinatewise score
  maxima and subdivides if needed. The additive score pad is 0.00002. With
  u = 2^−53 and γ(j) = ju/(1−ju), a bound on the coordinatewise summation
  and normalization error is T d γ(d+4) + γ(T) T (d + d γ(d+4)). This is
  below the pad at T ≤ 25,000, d ≤ 30 for the supported raw relatives in
  [0.1, 5]. The full 50-stock path uses its own every-close Arb upper
  endpoints. Inspect `unresolved_closes` and `unresolved_by_upper` to see
  which prefixes the independent audit settled.

Completed permutation and subset chunks are reused on a rerun. Their cache
directory is keyed by the hashes of the eight input files. The running times
and PDF metadata can vary across machines.
