# Changelog

## 1.0.1 -- 2026-10-21

Correction to the worked example. The demo previously stated that the CutMix
penalty curve rebounds between three and five epochs. That reading came from a
three-seed ladder; twenty-three seeds per point refute it.

### Changed

- `examples/demo.py` no longer asserts that the dip-and-rise is real. The curve is
  presented as a three-seed measurement, the question is left open, and an
  epilogue reports the re-measurement: the penalty at 3 epochs is +7.76
  (95% CI +6.01 to +9.50), at 5 epochs +7.04 (+5.41 to +8.66), and the
  difference is -0.72 (p = 0.54). The point estimate changes sign; the curve is
  monotone in that interval.
- The conditioning panel in the demo now says "at the local minimum" rather than
  "inside the rebound", and the explanation notes that a linear slope fitted
  across a turning point is not the slope of anything.
- The summary line reads "three seeds per point are not enough for this shape".

### Why this is a new version rather than a fix to the example data

The data were correct throughout: those are the real three-seed means. What was
wrong was the interpretation placed on them. The package's own fourth check is
what revealed the error, which is a better advertisement for the package than
the claim it replaced.

## 1.0.0 -- 2026-10-20

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
  relative error came out as 0 -- marking the *least* invertible region as
  reliable. Precision is now floored at 5% of the data's spread.
  Found by a unit test asserting that a perfectly straight flat segment must fail.
- `__all__` named `cond_result` instead of `CondResult`.
