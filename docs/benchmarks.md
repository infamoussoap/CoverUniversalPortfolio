# Stock examples and portfolio-selection benchmarks

Run the following in the repository root after installing the dependencies
and preparing the inputs with `python scripts/prepare_data.py all`. The
full run needs MATLAB on the path (or `MATLAB_BIN` set to its executable)
for the native OLPS audit. Matplotlib uses LaTeX to render the figures.
The six stages run sequentially with a single computational thread.

```sh
python output/cover_revision_20261121/run_benchmarks.py all
```

Individual stages use the same entry point. The tables below identify
dependencies and where each result appears in the paper.

| Stage | Required inputs and earlier stages | Manuscript outputs |
| --- | --- | --- |
| `examples` | 50-stock Yahoo panel, including AAPL and MSFT | Table 1, Figures 1 and 2, all daily reference and evidence values |
| `markets` | Six OLPS matrices | Table 2, strategy tapes for Tables 3 and 4, and the six hindsight regret intervals in §4.6 |
| `matlab` | Six OLPS matrices, MATLAB, pinned OLPS checkout at `vendor/OLPS` | Three native replays per dataset for Table 4. Set `PAPER_OLPS_ROOT` to use another copy of the pinned checkout. |
| `tables` | `markets` and `matlab` | Table 3, Table 4, Figure 3, MATLAB payoff verification and the published-source audit |
| `scaling` | FinOL and OLPS matrices | Table 5 and Figure 4. Running after `markets` also records the isolated timing measurements in Table 2. |
| `sensitivity` | FinOL and OLPS matrices | Terminal intervals with zero through five exact rows on the classic markets and FinOL NYSE(N), plus the FinOL control with the cubic intersection disabled |

### Numerical conventions

The certificate uses 128-bit Arb. Every recorded binary64 return is treated
as an exact input. The evidence is in units of equal-weight wealth. A
strategy gap is log Cover wealth minus log strategy wealth. The sign therefore
applies to the entire time series, including terminal values and seed ranges.
Table 3 uses units of 0.0001 log wealth with endpoints rounded outward. A star
in the generated CSV identifies a strategy with seeds certified on both sides
of Cover. The manuscript uses a dagger for this condition.

The two-asset reference uses positive Bernstein recursion at every close and
independent Gauss–Legendre quadrature of orders 3,270 and 3,300. The 50-stock
and six-market references pool eight independently scrambled Sobol nets of
65,536 points with a degree-four control polynomial, seeds 841120–841127.
Width decomposition uses orders 256 and 512 for two assets and eight
8,192-point nets (seeds 91491–91498) for the 50 stocks. The 100 through
100,000-portfolio fixed-sampling runs use NumPy seeds 74081–74085. The
toolbox audit uses MATLAB's twister seed 724191 and zero transaction costs.
All reported strategy tapes start at equal weights.

### Output and runtime

The final reports appear in `output/cover_experiments_final/`. With the
exception of hardware-dependent timings and floating-point reference estimates,
the results are deterministic for the fixed tapes and software versions.

| Files | Contents |
| --- | --- |
| `aapl_msft_exact.csv`, `aapl_msft_pair_series.pdf`, `fig2_50stock.csv`, `fig2_50stock.pdf`, `table1_width.csv` | Exact or numerical daily wealth, certified endpoints, decomposed widths and first two figures |
| `bench_datasets.csv`, `bench_extremes.csv`, `bench_evidence.csv`, `tableB.csv`, `certified_regret.csv` | Benchmarks, preflight checks, reference coverage, certified peak errors and regret |
| `bench_gaps.csv`, `bench_summary.csv`, `tableC.csv`, `tableC_seed_ranges.csv`, `bench_wealth.pdf` | Per-close gaps, terminal comparisons, seed ranges, verdicts and Figure 3 |
| `matlab_replay.csv`, `up_audit.csv`, `tableD.csv`, `payoff_checks.csv` | Native toolbox and independent wealth replays, published wealths and verification |
| `bench_cost.csv`, `bench_alloc_time.csv`, `bench_cost_trials.csv`, `bench_cost_slopes.csv`, `bench_scaling.pdf` | Median times, individual trials, memory, log–log slopes and Figure 4 |
| `exact_row_width.csv`, `exact_sets_all.csv`, `finol_nyse_exact_sets.csv`, `finol_nyse_empty_set_control.csv` | One-row diagnostic, all 42 nested exact-row cases, the five FinOL exact sets, and the cubic-bound ablation with all rows in the expansion |

Timing trials for scaling use fresh single-threaded Python processes and
retain peak resident memory in MiB. The paper's full 403-stock, 6,430-day
path took about 48 minutes per timing trial on an M1 Pro. A complete run
takes hours and repeats timed calculations three times. Performance varies
with hardware, BLAS and MATLAB. Intermediate files are saved by market,
seed and repetition in `output/cover_revision_20261121/cache/`. Restart
the same stage to resume with the same inputs. Each stage verifies the
input signatures associated with its checkpoints. Use a fresh checkout or
move the cache aside before changing the market vintage, algorithms or
seeds. Filenames identify the published configuration.
