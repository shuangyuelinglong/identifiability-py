# -*- coding: utf-8 -*-
"""
End-to-end demo: run all four checks on a real curve.

The data are the CutMix penalty curve from the paper's main ladder
(CIFAR-10, 3 seeds per budget point). The curve looks like a clean decay
and R² is high. The checks tell a more careful story.

Run:  python examples/demo.py
"""
import math

import numpy as np

import identifiability as I

# CIFAR-10, CutMix penalty (none minus cutmix), mean of 3 seeds per point.
X = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)
Y = np.array([11.97, 9.18, 7.14, 9.50, 5.59, 3.17, 2.98, 1.69, 1.25, -1.15])


def rule(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    rule("The curve")
    print(f"  {len(X)} points, penalty falls from {Y[0]:+.2f} to {Y[-1]:+.2f}")
    print("  It looks like a clean exponential decay. It is not quite one:")
    print("  the penalty rebounds at 5 epochs (7.14 -> 9.50) before falling again.")

    # ---------------------------------------------------------------- 1
    rule("1. Null distribution: is R² better than noise?")
    r = I.null_r2(X, Y, k=3, iters=2000, seed=0)
    print(f"  R² = {r.r2:.3f}")
    print(f"  null median = {r.null_median:.3f}, 90th pct = {r.null_q90:.3f}")
    print(f"  p = {r.p_value:.4f}")
    print(f"  -> {'PASS' if r.p_value < 0.05 else 'FAIL'}: the curve does contain structure.")
    print("     The fit is not noise. Keep going.")

    # ---------------------------------------------------------------- 2
    rule("2. Leave-one-out: do the parameters survive?")
    lv = I.loo_spread(X, Y)
    print("  " + str(lv))
    print(f"  A = {lv.full[0]:.2f}, tau = {lv.full[1]:.2f}, C = {lv.full[2]:.2f}")
    print(f"  -> {'PASS' if lv.passed else 'FAIL'}: C is not constrained.")
    print("     C is the asymptote, and the curve has not flattened out, so")
    print("     dropping a single point moves C by a thousand percent.")

    # ---------------------------------------------------------------- 3
    rule("3. Conditioning: can the curve be inverted?")
    for x0, note in ((3.0, "inside the rebound"), (13.0, "mid-decay"), (50.0, "near saturation")):
        c = I.conditioning(X, Y, x0=x0)
        verdict = "ok" if c.rel_err <= c.thresh else "ABOVE THRESHOLD"
        print(f"  x0={x0:>5.0f} ({note:<18}) slope={c.slope:+.4f}  "
              f"rel.err={c.rel_err*100:>6.1f}%  {verdict}")
    print()
    print("  At x=3 the local slope is positive: the curve rebounds there, so a")
    print("  local linear slope is meaningless. At x=13 the relative error is")
    print("  23.4%, above the 15% threshold. Anything obtained by inverting the")
    print("  curve near x=13 is not a quantitative argument.")

    # ---------------------------------------------------------------- 4
    rule("4. Power: how many seeds does an effect of this size need?")
    for eff, sd in ((2.0, 0.6), (0.9, 1.5), (5.6, 2.8)):
        n = I.required_n(eff, sd, sd)
        print(f"  effect {eff:+.1f} pt, sd {sd:.1f} -> n = {math.ceil(n)} per group")
    print()
    print("  A 0.9 pt effect against 1.5 pt of noise needs 44 seeds per group.")
    print("  The variance you feed in is itself estimated from a handful of runs,")
    print("  so read these as order of magnitude.")

    # ---------------------------------------------------------------- summary
    rule("Summary")
    print("  Check 1 passes: the curve is real.")
    print("  Check 2 fails:  one parameter is unconstrained.")
    print("  Check 3 fails:  the curve is not invertible where it matters.")
    print("  Check 4 says:   small effects need far more seeds than is typical.")
    print()
    print("  High R², unreliable fit. That is the whole point of the package.")


if __name__ == "__main__":
    main()
