"""Check daily containment, played intervals and payoffs."""

import numpy as np
import pandas as pd
from flint import arb
from scipy.special import roots_legendre


from assemble_market import OUT, read
from extra_checks import ends
from particle_universal.linear_cover import linear_cover
from study import CACHE, evidence, tape


def verify_pair():
    x = tape('pair')
    exact = read('aapl_msft_exact.csv', OUT)
    box, _, _ = evidence(x, 1, pairs=False)
    exact['taylor_lo'] = box[:, 0]
    exact['taylor_hi'] = box[:, 1]
    assert (exact.taylor_lo <= exact.exact_lo).all()
    assert (exact.exact_hi <= exact.taylor_hi).all()

    orders = (3270, 3300)
    for order in orders:
        nodes, weights = roots_legendre(order)
        nodes, weights = (1 + nodes) / 2, weights / 2
        log_p = np.zeros(order)
        values = np.empty(len(x))

        for i, (a, b) in enumerate(x):
            centered = (a - b) / (a + b)
            log_p += np.log1p((2 * nodes - 1) * centered)
            values[i] = np.log(weights @ np.exp(log_p))

        exact[f'gl_{order}'] = values

    assert np.max(abs(exact.gl_3270 - exact.gl_3300)) < 1e-10
    assert np.max(abs(exact.gl_3300 - exact.exact_lo)) < 1e-10
    exact.to_csv(OUT / 'aapl_msft_exact.csv', index=False)

    table = read('table1_width.csv', OUT)
    last = exact.iloc[-1]
    for tag, lo, hi in [('pair_terminal', last.z_lo, last.z_hi),
                        ('taylor', last.taylor_lo, last.taylor_hi)]:
        low = ((arb(float(lo)) - arb(float(last.exact_hi))).exp() - 1) * 100
        high = ((arb(float(hi)) - arb(float(last.exact_lo))).exp() - 1) * 100
        table.loc[table.path == 'pair', [tag + '_lo_pct',
                                         tag + '_hi_pct']] = ends(low, high)

    table.to_csv(OUT / 'table1_width.csv', index=False)


def subset_tapes():
    records = []
    for name, dims, horizons in [
        ('finol_nyse', (25, 50, 100, 200, 403), (1000, 2000, 4000, 6430)),
        ('finol_sp', (339,), (1268, 1277))]:
        raw = tape(name)
        for d in dims:
            x = raw[:, :d]
            w = linear_cover(x)
            pay = np.einsum('ij,ij->i', x, w)
            assert np.isfinite(w).all() and w.min() >= 0
            assert np.max(abs(w.sum(1) - 1)) < 2e-12 and pay.min() > 0

            a = x / x.mean(1, keepdims=True) - 1
            cumulative = a.cumsum(0)
            log_wealth = np.log(pay).cumsum()
            for T in horizons:
                failures = np.sum(cumulative[:T-1].min(1) <= -(d + 1))
                records.append(dict(dataset=name, d=d, T=T, fallback=failures,
                                    min_payoff=pay[:T].min(),
                                    measured_log_wealth=log_wealth[T-1]))

    pd.DataFrame(records).to_csv(OUT / 'scaling_first_order.csv', index=False)


def verify_gaps():
    gaps = read('bench_gaps.csv', OUT)
    evidence_df = read('bench_evidence.csv', OUT)
    keys = ['dataset', 'alpha', 't']
    assert not evidence_df.duplicated(keys).any()
    assert evidence_df.ref_logZ.between(evidence_df.z_lo,
                                        evidence_df.z_hi).all()

    base = gaps.merge(evidence_df[keys + ['z_lo', 'z_hi']], on=keys,
                      validate='many_to_one', how='left')
    assert np.isfinite(base[['z_lo', 'z_hi']]).all().all()
    # delta_hi rounds upward from z_hi - v, so z_hi - delta_hi <= v.
    gaps['v_lo'] = np.nextafter(base.z_hi - base.delta_hi, -np.inf)
    gaps['v_hi'] = np.nextafter(base.z_lo - base.delta_lo, np.inf)
    assert (gaps.v_lo <= gaps.v_hi).all()

    stats = read('bench_summary.csv', OUT)
    path_key = ['dataset', 'alpha', 'strategy', 'seed']
    assert not stats.duplicated(path_key).any()
    assert not gaps.duplicated(path_key + ['t']).any()
    n = (stats.n_ahead + stats.n_behind + stats.n_undecided).sum()
    assert len(stats) == 306 and len(gaps) == n

    by_key = stats.set_index(path_key, verify_integrity=True)
    payoff_audit = []
    for key, group in gaps.groupby(path_key, sort=False):
        name, alpha, strategy, seed = key
        t = group.t.to_numpy()
        x = tape(name)
        assert np.array_equal(t, np.arange(1, len(x) + 1))
        f = CACHE / f'w_{name}_{alpha:g}_{strategy}_{seed}.npz'
        w = np.load(f)['w']
        assert w.shape == x.shape and w.min() >= 0
        assert np.max(abs(w.sum(1) - 1)) < 2e-12
        pay = np.einsum('ij,ij->i', w, x)
        assert np.isfinite(pay).all() and min(pay) > 0

        norm = np.load(CACHE / ('g_' + f.name))['v']
        assert ((group.v_lo <= norm) & (norm <= group.v_hi)).all()
        s = by_key.loc[key]
        assert s.n_ahead == (group.delta_hi < 0).sum()
        assert s.n_behind == (group.delta_lo > 0).sum()
        assert s.B == max(group.delta_hi.max(), -group.delta_lo.min())
        assert s.term_lo == group.delta_lo.iloc[-1]
        assert s.term_hi == group.delta_hi.iloc[-1]
        payout = dict(zip(path_key, key))
        payout.update(min_daily_payoff=min(pay), max_daily_payoff=max(pay))
        payoff_audit.append(payout)

    pd.DataFrame(payoff_audit).to_csv(OUT / 'payoff_checks.csv', index=False)
    gaps.to_csv(OUT / 'bench_gaps.csv', index=False)
    assert (stats.loc[stats.E_resolved, 'B_over_E'] >= 1).all()
