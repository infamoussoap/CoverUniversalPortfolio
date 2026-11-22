"""Check the cubic intersection against an exactly integrable market."""

from flint import arb, ctx

from particle_universal.tilted_interval_cover import TiltedCover


def test_cubic_intersection():
    # With a uniform two-asset prior the one-day payoff is uniform on [a,b].
    with ctx.workprec(128):
        a, b = arb(1) / 4, arb(7) / 4
        for pairs in (True, False):
            monitor = TiltedCover(2, exact_pairs=pairs)
            tightened = False
            old = []

            def previous(lower, upper):
                old[:] = lower, upper

            for t in range(1, 41):
                monitor.update([a, b], [1, 1])
                lo, hi = monitor.evidence(legacy_bounds=previous)
                mass = (b**(t + 1) - a**(t + 1)) / ((t + 1) * (b - a))
                exact = mass.log()
                assert old[0] <= lo <= exact.lower()
                assert exact.upper() <= hi <= old[1]
                tightened |= lo > old[0] or hi < old[1]

            if pairs:
                assert tightened
