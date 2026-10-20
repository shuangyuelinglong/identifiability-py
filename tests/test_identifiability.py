# -*- coding: utf-8 -*-
"""Tests for the four identifiability checks.

Every test uses a fixed seed and fixed data, so results are deterministic.
"""
import math

import numpy as np
import pytest

import identifiability as I


# ---------------------------------------------------------------- fixtures

X10 = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)


def clean_curve(noise=0.0, seed=0):
    """A well-behaved exponential decay, optionally with Gaussian noise."""
    y = I.exp_decay(X10, 12.0, 8.0, 0.2)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, len(X10))
    return X10, y


# ---------------------------------------------------------------- model

def test_exp_decay_matches_formula():
    y = I.exp_decay(np.array([0.0, 1.0]), 3.0, 2.0, 5.0)
    assert y[0] == pytest.approx(3.0 * math.exp(0) + 5.0)
    assert y[1] == pytest.approx(3.0 * math.exp(-0.5) + 5.0)


def test_exp_decay_handles_zero_tau():
    """tau is floored at EPS, so this must not divide by zero."""
    y = I.exp_decay(np.array([1.0]), 1.0, 0.0, 0.0)
    assert np.isfinite(y).all()


# ---------------------------------------------------------------- check 1: null distribution

def test_null_r2_rejects_pure_noise():
    """Fitting noise must yield a high p-value (cannot reject the null)."""
    y = np.random.default_rng(1).normal(0, 1, len(X10))
    r = I.null_r2(X10, y, k=3, iters=200, seed=0)
    assert r.n == 10 and r.k == 3
    assert r.p_value > 0.05, f"pure noise should not be significant, got p={r.p_value}"


def test_null_r2_accepts_clean_curve():
    """A clean exponential must sit far above the null distribution."""
    x, y = clean_curve()
    r = I.null_r2(x, y, k=3, iters=200, seed=0)
    assert r.r2 > 0.99
    assert r.r2 > r.null_q90
    assert r.p_value < 0.01


def test_null_r2_is_reproducible():
    """Same seed must give identical output (the paper's reproducibility claim)."""
    x, y = clean_curve(noise=1.0, seed=3)
    a = I.null_r2(x, y, k=3, iters=100, seed=42)
    b = I.null_r2(x, y, k=3, iters=100, seed=42)
    assert a.null_q90 == b.null_q90
    assert a.p_value == b.p_value


def test_null_r2_design_x_changes_the_null():
    """The null depends on the x design, not only on (n, k)."""
    x, y = clean_curve(noise=1.0, seed=5)
    same = I.null_r2(x, y, k=3, iters=300, seed=0)
    other = I.null_r2(x, y, k=3, iters=300, seed=0,
                      design_x=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], float))
    assert same.null_q90 != other.null_q90


# ---------------------------------------------------------------- check 2: leave-one-out

def test_loo_returns_one_fit_per_point():
    x, y = clean_curve()
    r = I.loo_spread(x, y)
    assert r.full.shape == (3,)
    assert r.loo.shape == (len(x), 3)
    assert len(r.names) == 3


def test_loo_passes_on_a_well_determined_curve():
    x, y = clean_curve()
    r = I.loo_spread(x, y)
    assert r.passed, f"clean curve should pass LOO, spreads={r.spread}"


def test_loo_spread_is_relative():
    """spread is (max-min)/|full|, so a near-zero parameter blows it up."""
    x, y = clean_curve()
    r = I.loo_spread(x, y)
    assert np.all(r.spread >= 0)
    assert np.all(np.isfinite(r.spread))


# ---------------------------------------------------------------- check 3: conditioning

def test_conditioning_flat_region_is_unreliable():
    """At the saturated end the slope is ~0, so inversion error explodes."""
    x, y = clean_curve()
    r = I.conditioning(x, y, x0=80.0)
    assert r.x0 == 80.0
    assert r.rel_err > r.thresh, f"saturated end should fail, got {r.rel_err}"


def test_conditioning_early_region_is_reliable():
    """Near the steep start the slope is large, so inversion is stable."""
    x, y = clean_curve()
    r = I.conditioning(x, y, x0=3.0)
    assert r.rel_err < 1.0


# ---------------------------------------------------------------- check 4: required n

def test_required_n_shrinks_with_larger_effect():
    small = I.required_n(1.0, 2.0, 2.0)
    large = I.required_n(4.0, 2.0, 2.0)
    assert large < small


def test_required_n_grows_with_larger_sd():
    tight = I.required_n(2.0, 1.0, 1.0)
    loose = I.required_n(2.0, 3.0, 3.0)
    assert loose > tight


def test_required_n_matches_the_16_over_d2_rule():
    """For equal sds, the paper's approximation is n = 16/d^2."""
    n = I.required_n(2.0, 1.0, 1.0, convention="paper")
    assert n == pytest.approx(16.0 / 4.0, abs=1)


def test_required_n_is_infinite_for_zero_effect():
    """A literally zero effect can never be detected, whatever the sample size."""
    assert I.required_n(0.0, 1.0, 1.0) == float("inf")


# ---------------------------------------------------------------- aggregate

def test_power_table_structure():
    groups = {
        "none": [60.1, 61.2, 59.8],
        "cutmix": [48.3, 49.1, 47.9],
        "mixup": [58.9, 59.1, 60.2],
    }
    rows = I.power_table(groups, control="none")
    assert set(rows) == {"cutmix", "mixup"}
    for v in rows.values():
        assert "effect" in v and "required_n" in v
    # CutMix's effect is far larger, so it must need far fewer seeds
    assert rows["cutmix"]["required_n"] < rows["mixup"]["required_n"]


def test_report_returns_all_three_curve_checks():
    """report() returns the three curve-level checks as a tuple."""
    x, y = clean_curve(noise=0.3, seed=7)
    rep = I.report(x, y, k=3, iters=100)
    assert isinstance(rep, tuple) and len(rep) == 3
    assert isinstance(rep[0], I.NullResult)
    assert isinstance(rep[1], I.LOOResult)
    assert isinstance(rep[2], I.CondResult)
