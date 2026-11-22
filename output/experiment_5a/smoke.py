"""Check daily decisions on a small deterministic two-asset market."""

import numpy as np
from flint import arb, ctx

from arb_checks import gauss
from evalues import positive_pair
from market import first_rejection, random_market
from screen import screen
from study import evidence
from subsets import pair_checkpoints
from threshold_audit import certify


def smoke():
    ctx.prec, ctx.threads = 128, 1
    rng = np.random.default_rng(1128)
    x = np.exp(rng.normal(0, .025, (40, 2)))
    base = random_market(x, 0, len(x))
    exact = positive_pair(base)
    bounds, _, _ = evidence(base, 1)
    gl = gauss(base)

    np.testing.assert_allclose(gl, exact.mean(1), rtol=0, atol=5e-14)
    assert np.all(bounds[:, 0] <= exact[:, 0])
    assert np.all(exact[:, 1] <= bounds[:, 1])
    np.testing.assert_allclose(screen(base, True), bounds[:, 0],
                               rtol=0, atol=1e-12)

    boxes, oracle = pair_checkpoints(base, {40})
    lo, hi = boxes[40]
    assert lo <= oracle.lower() and oracle.upper() <= hi
    assert oracle.overlaps(arb(exact[-1, 0]).union(arb(exact[-1, 1])))
    upper, unresolved = certify(base)
    assert unresolved == 0 and upper < np.log(20)
    assert first_rejection(bounds[:, 0]) is None
    print('40/40 Bernstein intervals enclosed; both decisions agree.')


if __name__ == '__main__':
    smoke()
