# Market inputs

Run these commands from the repository root after installing the Python
dependencies. Git is required for the two pinned upstream repositories.

```sh
python scripts/prepare_data.py all
python scripts/prepare_data.py verify
```

Each source also has its own command, for example
`python scripts/prepare_data.py olps` and
`python scripts/prepare_data.py verify --only olps`. Preparation can be
resumed. The published identities are in [the input manifest](../provenance/inputs.json).
A return matrix SHA-256 covers the contiguous sequence of row-major
little-endian binary64 values. Calendar hashes cover the concatenated ISO
dates as 10-byte ASCII strings. The Yahoo price calendar includes the base
close. The classic MAT inputs have no individual calendar labels.

| Input | Dimensions (days × assets) | Prepared location |
| --- | --- | --- |
| NYSE(O), NYSE(N) | 5,651 × 36 and 6,431 × 23 | Study data, `nyse-o.mat` and `nyse-n.mat` |
| DJIA, SP500, MSCI, TSE | 507 × 30, 1,276 × 25, 1,043 × 24, 1,259 × 88 | Study data, corresponding lowercase `.mat` files |
| FinOL NYSE(N), FinOL SP500 | 6,430 × 403 and 1,277 × 339 | Study data, `finol_nyse.npz` and `finol_sp.npz` |
| Stock panel | 6,539 × 50 | `particle_universal/results/yahoo_fixed50.npz` |
| Sector panel | 6,791 × 9 | Study data, `sectors.csv` (includes one base close) |

The study data directory is `output/cover_revision_20261121/data/`.
The vendor checkouts, price histories and prepared return matrices are local
inputs excluded from version control.

## OLPS and FinOL

The OLPS setup checks out the pinned Li–Sahoo–Hoi toolbox under
`vendor/OLPS/`, the path used by the native MATLAB replay. It copies the
six original MAT files and checks each file digest, matrix shape and return
digest. [Pinned toolbox and data](https://github.com/OLPS/OLPS/tree/9120783cd59a7966b0f78e2b5668030a4378b8af)

The FinOL setup requests the NYSE(N) and SP500 workbooks and the two released
test matrices at the recorded Git revision. It verifies each of the 742
workbooks, reads the CLOSE column in filename order and labels each price
relative with the earlier close. The last row consists of ones, following
FinOL's convention. Every entry of the resulting test suffix is checked
against the released MAT file. The SP500 experiment uses all 1,277 rows.
[Upstream FinOL release](https://github.com/ai4finol/finol_data/tree/e90f4614cde7742f9362177dbb50f8667454b4bb)
· [Workbook checksums](../provenance/finol_sources.json)

Existing checkouts at the same commits can supply the inputs offline.
The OLPS command materializes the toolbox under `vendor/OLPS/`, while
the FinOL command reads the supplied checkout in place.

```sh
python scripts/prepare_data.py olps --olps-source /path/to/OLPS
python scripts/prepare_data.py finol --finol-source /path/to/finol_data
```

## Yahoo vintages

The paper used yfinance 1.4.1 and Yahoo's adjusted close, with automatic
price adjustment and repair disabled. The 50-stock prices were retrieved
on 21 November 2026 at 04:10 UTC and the nine sector ETF prices at 15:16 UTC
on the same day. The original stock selection took the first five complete
positive histories in each of ten industry lists on SPY's calendar. The
manifest records the candidate order and the exact resulting 50 tickers.
The downloader requests these same constituents and checks the entire
calendar. The stock archive begins at the base close of 31 December 1999.
The nine sector ETF series begin on 31 December 1998. Both panels end
on 31 December 2025. The sector returns use the original CSV serialization
and pandas' default floating-point parser. [yfinance data-use notice](https://github.com/ranaroussi/yfinance#readme)

For a fresh Yahoo download, run the command below. It compares the resulting
return matrix with the paper fingerprint before installing it. The file digest
and matrix digest are both recorded in the local `yahoo_provenance.json`.
Different encodings of the same inputs can have different file hashes.

```sh
python scripts/prepare_data.py yahoo
```

Yahoo can revise historical adjusted prices. For an exact replay, a holder
of the original snapshots can import the two files and verify the return
hashes. The stock archive has the arrays `tickers`, `price_dates`, `dates`,
`adjusted_close` and `returns`. The sector CSV contains a `Date` column followed
by XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY, in that order.

```sh
python scripts/prepare_data.py yahoo \
    --stock-snapshot /path/to/yahoo_fixed50.npz \
    --sector-snapshot /path/to/sectors.csv
python scripts/prepare_data.py verify --only yahoo
```

To rerun the same experiment on a revised Yahoo vintage, use a **fresh
checkout**, then run the following. This retains the study universe and dates
and records any changed return digest. A separate checkout keeps existing
experiment checkpoints associated with their original inputs.

```sh
python scripts/prepare_data.py yahoo --allow-revised-yahoo
python scripts/prepare_data.py verify --only yahoo --allow-revised-yahoo
```
