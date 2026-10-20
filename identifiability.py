# -*- coding: utf-8 -*-
"""
identifiability -- four checks for whether a fitted curve and the quantities
derived from it are identifiable from data.

Implements the reporting protocol of section 5.6 of the accompanying paper:

    1. null distribution (null_r2)   is the reported R^2 above what noise gives?
    2. leave-one-out     (loo_spread) do the parameters survive dropping a point?
    3. conditioning      (conditioning) is the local slope steep enough to invert?
    4. power             (required_n) how many seeds does the effect actually need?

See the __main__ block or the README for usage.

All console output is ASCII so the library runs on any terminal, including
Windows consoles using a legacy code page.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

__all__ = ["exp_decay", "NullResult", "null_r2", "LOOResult", "loo_spread",
           "CondResult", "conditioning", "required_n", "power_table", "report"]

EPS = 1e-12


# ---------------------------------------------------------------- model

def exp_decay(t, A, tau, C):
    """Three-parameter exponential decay P(t) = A*exp(-t/tau) + C (paper eq. 1)."""
    t = np.asarray(t, dtype=float)
    return A * np.exp(-t / np.maximum(tau, EPS)) + C


def _fit(model, x, y, p0=None, bounds=None, bounded=False):
    """Least-squares fit.

    Defaults to bounded=False, i.e. *unbounded* least squares, matching the
    procedure used to fit the decay law in the paper. This matters: the null
    distribution must use the same fitting procedure as the actual curve, or
    the test is invalid. A bounded fit (A >= 0) pushes the R^2 of pure noise
    lower, which shrinks the null and makes the test too permissive.

    Set bounded=True to enable the bounded variant (multi-start, A >= 0) for
    comparison.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = _auto_p0(x, y)
    p0 = np.asarray(p0, dtype=float)
    if not bounded:
        try:
            popt, _ = optimize.curve_fit(model, x, y, p0=p0, maxfev=20000)
            if np.all(np.isfinite(popt)):
                return popt
        except Exception:
            pass
        return p0
    if bounds is None:
        span_y = max(y.max() - y.min(), 1.0)
        bounds = ([0.0, 1e-3, y.min() - span_y],
                  [max(abs(y.min()), abs(y.max())) * 10 + span_y, 1e4, y.max() + span_y])
    lo, hi = bounds
    p0 = np.clip(p0, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
    try:
        popt, _ = optimize.curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception:
        popt = p0
    return np.asarray(popt, dtype=float)


def _auto_p0(x, y):
    """Starting values: the y range as amplitude, the x span as the time scale."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = float(y.max() - y.min()) or 1.0
    tau = float(x.max() - x.min()) / 2 or 1.0
    C = float(y.min())
    return [A, tau, C]


def _r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0


# ---------------------------------------------------------------- check 1: null distribution

@dataclass
class NullResult:
    r2: float                 # R^2 of the actual fit
    null_median: float        # median R^2 under the null
    null_q90: float           # 90th percentile of the null
    p_value: float            # P(noise R^2 >= observed R^2)
    n: int
    k: int
    iters: int

    @property
    def passed(self) -> bool:
        return self.r2 > self.null_q90

    def __str__(self):
        return (f"null    R^2={self.r2:.3f}  null_median={self.null_median:.3f} "
                f"null_q90={self.null_q90:.3f}  p={self.p_value:.4f}  "
                f"n={self.n} k={self.k}  -> {'PASS' if self.passed else 'FAIL'}")


def null_r2(x, y, model=exp_decay, k=3, iters=2000, seed=0, design_x=None, bounded=False):
    """
    Empirical null distribution of R^2 at the same (n, k).

    Draws iters independent N(0, 1) series, fits them with the same fitter, and
    records R^2. This is as important as the fit itself: the reported R^2 only
    means something if it sits clearly above what noise would produce.

    design_x: if given, use this x design for the null instead of the observed x.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xd = np.asarray(design_x, float) if design_x is not None else x
    n = len(x)
    yhat = model(x, *_fit(model, x, y, bounded=bounded))
    r2 = _r2(y, yhat)
    rng = np.random.default_rng(seed)
    nulls = np.empty(iters)
    for i in range(iters):
        yn = rng.standard_normal(len(xd))
        nulls[i] = _r2(yn, model(xd, *_fit(model, xd, yn, bounded=bounded)))
    p = float((nulls >= r2).mean())
    return NullResult(r2, float(np.median(nulls)), float(np.quantile(nulls, 0.9)),
                      p, n, k, iters)


# ---------------------------------------------------------------- check 2: leave-one-out

@dataclass
class LOOResult:
    full: np.ndarray          # parameters from all points
    loo: np.ndarray           # parameters from each leave-one-out fit, shape (m, k)
    names: list = field(default_factory=lambda: ["A", "tau", "C"])

    @property
    def spread(self) -> np.ndarray:
        """Relative spread per parameter: (max - min) / |full value|."""
        s = []
        for j in range(self.full.shape[0]):
            rng = self.loo[:, j].max() - self.loo[:, j].min()
            s.append(float(rng / max(abs(self.full[j]), EPS)))
        return np.array(s)

    @property
    def passed(self) -> bool:
        """A relative spread above 100% on any parameter means it is unreliable."""
        return bool(np.all(self.spread < 1.0))

    def __str__(self):
        parts = [f"{nm}: {sp*100:.1f}%" for nm, sp in zip(self.names, self.spread)]
        return ("loo     spread  " + "  ".join(parts)
                + f"  -> {'PASS' if self.passed else 'FAIL'}")


def loo_spread(x, y, model=exp_decay, names=None):
    """Refit once per dropped point. A stable fit barely moves its parameters."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    full = _fit(model, x, y)
    rows = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        rows.append(_fit(model, x[m], y[m]))
    r = LOOResult(full, np.vstack(rows))
    if names:
        r.names = list(names)
    return r


# ---------------------------------------------------------------- check 3: conditioning

@dataclass
class CondResult:
    x0: float                 # target point
    slope: float              # local slope dY/dX
    rel_err: float            # relative error of the inverted quantity
    thresh: float

    @property
    def passed(self) -> bool:
        return self.rel_err < self.thresh

    def __str__(self):
        return (f"cond    x={self.x0:g}  slope={self.slope:.4g}  "
                f"rel_err={self.rel_err*100:.1f}%  threshold<{self.thresh*100:.0f}%  "
                f"-> {'PASS' if self.passed else 'FAIL'}")


def conditioning(x, y, x0=None, delta_y=None, thresh=0.15, window=3):
    """
    Error propagation: relative error = delta_y / (slope * x).

    Any quantity that has to be obtained by inverting the curve should first be
    checked against the local slope. Inversion on a flat segment is unusable no
    matter how precisely the input was measured.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x); x, y = x[order], y[order]
    if x0 is None:
        x0 = float(x[len(x) // 2])
    i = int(np.argmin(np.abs(x - x0)))
    lo = max(0, i - window // 2); hi = min(len(x), lo + window)
    if hi - lo < 2:
        lo, hi = max(0, i - 1), min(len(x), i + 2)
    sl, *_ = np.polyfit(x[lo:hi], y[lo:hi], 1)
    if delta_y is None:
        # No external error given: estimate the measurement precision of y from
        # the local fit residual, but floor it.
        #
        # Why the floor is necessary: on a saturated segment the local points are
        # nearly collinear, so the residual approaches zero. Taking the residual
        # as the precision then reports a relative error of 0, marking the least
        # invertible region as reliable. As the slope approaches 0 any input error
        # is amplified, so the precision cannot be better than a fixed fraction of
        # the data's overall spread (5% here).
        coef = np.polyfit(x[lo:hi], y[lo:hi], 1)
        fit = np.polyval(coef, x[lo:hi])
        resid = float(np.std(y[lo:hi] - fit))
        floor = float(np.std(y)) * 0.05
        delta_y = max(resid, floor)
    denom = abs(sl) * max(abs(x0), EPS)
    rel = float(delta_y / denom) if denom > EPS else float("inf")
    return CondResult(float(x0), float(sl), rel, thresh)


# ---------------------------------------------------------------- check 4: power

def required_n(effect, sd_control, sd_treat, alpha=0.05, power=0.8, convention="exact"):
    """
    Per-group sample size for a two-sample t-test (Cohen's d, pooled sd).

        d = |effect| / sqrt((sd_control^2 + sd_treat^2) / 2)
        n = 2 * (z_{alpha/2} + z_{power})^2 / d^2

    convention="exact"  the formula above (coefficient 15.70 at the defaults)
    convention="paper"  the rounded n ~= 16 / d^2 from section 3.3 of the paper,
                        about 2% larger, for aligning with its tables

    Caveat: the variance you pass in is itself estimated from a handful of runs
    and can be off by a factor of two or more. Read the result as an order of
    magnitude, not a budget.
    """
    sp = math.sqrt((sd_control ** 2 + sd_treat ** 2) / 2.0)
    if sp <= EPS or effect == 0:
        return float("inf")
    d = abs(effect) / sp
    if convention == "paper":
        coef = 16.0
    else:
        coef = 2 * (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2
    return float(coef / d ** 2)


def power_table(groups: dict, control: str = "none", alpha=0.05, power=0.8):
    """groups: {condition: [accuracy of each run]}; required n for every non-control."""
    base = np.asarray(groups[control], float)
    out = {}
    for name, vals in groups.items():
        if name == control:
            continue
        a = np.asarray(vals, float)
        eff = base.mean() - a.mean()
        n = required_n(eff, base.std(ddof=1), a.std(ddof=1), alpha, power)
        out[name] = dict(effect=float(eff), sd_control=float(base.std(ddof=1)),
                         sd_treat=float(a.std(ddof=1)), required_n=(None if math.isinf(n) else int(math.ceil(n))))
    return out


# ---------------------------------------------------------------- summary

def report(x, y, model=exp_decay, k=3, iters=2000, x0=None, thresh=0.15, seed=0):
    """Run the three curve-level checks and print them. Call required_n separately."""
    print("=" * 68)
    print("identifiability checks (paper section 5.6 protocol)")
    print("=" * 68)
    r1 = null_r2(x, y, model=model, k=k, iters=iters, seed=seed)
    r2 = loo_spread(x, y, model=model)
    r3 = conditioning(x, y, x0=x0, thresh=thresh)
    for r in (r1, r2, r3):
        print("  " + str(r))
    ok = r1.passed and r2.passed and r3.passed
    print()
    print("  verdict: " + ("all three passed, the fit may be used quantitatively"
                           if ok else "at least one check failed, do not use it quantitatively"))
    return r1, r2, r3


if __name__ == "__main__":
    # The CutMix and MixUp penalty curves from table 1 of the paper.
    t = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)
    cutmix = np.array([11.97, 9.18, 7.14, 9.50, 5.59, 3.17, 2.98, 1.69, 1.25, -1.15])
    mixup = np.array([2.39, 0.50, 0.97, 1.97, 0.92, 1.46, -0.51, -1.01, 0.09, -0.13])

    print("### CutMix penalty curve (large, well-shaped effect)")
    report(t, cutmix, iters=500)
    print()
    print("### MixUp penalty curve (effect not measurable)")
    report(t, mixup, iters=500)

    print()
    print("=" * 68)
    print("power: how many seeds would it take to establish the MixUp penalty")
    print("=" * 68)
    print("  Standard deviations measured in the paper (pooled per section 3.3).")
    rows = [(1, 2.39, 0.27, 2.77), (2, 0.50, 2.04, 2.21), (3, 0.97, 0.54, 0.79),
            (5, 1.97, 1.06, 1.18), (8, 0.92, 0.21, 1.47)]
    print(f"  {'budget':>7}{'effect':>9}{'sd_none':>9}{'sd_mixup':>10}{'exact':>8}{'paper':>7}")
    for ep, eff, s1, s2 in rows:
        ne = required_n(eff, s1, s2, convention="exact")
        np_ = required_n(eff, s1, s2, convention="paper")
        print(f"  {ep:>7}{eff:>9.2f}{s1:>9.2f}{s2:>10.2f}"
              f"{int(math.ceil(ne)):>8}{int(math.ceil(np_)):>7}")
    print()
    print("  exact = 2 (z_a/2 + z_b)^2 / d^2, the default")
    print("  paper = n ~= 16 / d^2 from section 3.3, about 2% larger")
    print()
    print("  Validation in section 4.4: at 1/2/3 epochs the exact formula predicts")
    print("  11/290/8 seeds; with 23 seeds collected, the two significant cases")
    print("  stayed significant and the undetectable one stayed undetectable.")
