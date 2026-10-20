# Changelog

## 1.0.0 — 2026-10-20

First release.

- `null_r2`: empirical null distribution for a fitted curve, with the
  option to use a canonical `design_x` instead of the observed x.
- `loo_spread`: leave-one-out parameter stability.
- `conditioning`: local-slope error propagation for inverting a curve.
- `required_n` / `power_table`: per-group sample size for a two-sample t-test.
- `report`: the three curve-level checks in one call.
- 17 tests, all deterministic.

### Fixed before first release

- **`conditioning` reported zero error on flat segments.** When `delta_y` was
  omitted, the local linear residual was used as the measurement precision.
  On a locally straight, near-flat segment that residual approaches zero, so the
  relative error came out as 0 — marking the *least* invertible region as
  reliable. Precision is now floored at 5% of the data's spread.
  Found by a unit test asserting that a perfectly straight flat segment must fail.
- `__all__` named `cond_result` instead of `CondResult`.
