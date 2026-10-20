# identifiability

[![tests](https://github.com/shuangyuelinglong/identifiability-py/actions/workflows/tests.yml/badge.svg)](https://github.com/shuangyuelinglong/identifiability-py/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-v1.0.0-brightgreen)](CHANGELOG.md)

**English** | [简体中文](README.zh-CN.md)

> **A high R² does not mean the fit is trustworthy.**

Four checks for one question: is this fitted curve — and the numbers derived from it — actually identifiable from the data you have?

![The four checks applied to one real curve](docs/four_checks.png)

*One real penalty curve (10 points). The fit looks like a clean decay and R² = 0.92. Check 1 passes: the curve is real. Check 2 fails: one parameter moves by 1017% when a single point is dropped. Check 3 fails at four of ten locations: the curve cannot be inverted there. Check 4 says a 0.9 pt effect against 1.5 pt of noise needs 44 seeds, not 3.*

---

## Install

```bash
pip install git+https://github.com/shuangyuelinglong/identifiability-py.git
```

Requires `git`. Not on PyPI yet — if both work for you, the archive route is more robust:

```bash
# download the source archive, then
pip install identifiability-py-main.tar.gz
```

Two runtime dependencies: `numpy` and `scipy`. No training, no GPU, no data download. Python 3.9+.

---

## Why

Suppose you fit a 3-parameter decay law to 4 data points and get R² = 0.99. Is that evidence?

- **Maybe not.** Pure noise fitted with 3 parameters reaches R² = 0.43 at the 90th percentile when n = 10, and exceeds 0.5 with 46% probability when n = 4.
- **Even if the curve is real**, individual parameters can be unconstrained — dropping one point may move a parameter by 1000%.
- **And any quantity you invert out of the curve** (a break-even point, a cost multiplier) inherits the local slope, which may be almost zero exactly where you need it.

Each of these has a cheap check. That is what this package is.

---

## The four checks

| # | Check | Function | Passes when |
|---|---|---|---|
| 1 | **Null distribution** | `null_r2` | Your R² sits far above what pure noise would give at the same (n, k) |
| 2 | **Leave-one-out** | `loo_spread` | Removing any single point barely moves the parameters |
| 3 | **Conditioning** | `conditioning` | The local slope is steep enough that inverting it is stable |
| 4 | **Power** | `required_n` | You actually ran as many seeds as the effect needs |

If any check fails, the number should not be used as a quantitative argument.

---

## Quick start

```python
import numpy as np
import identifiability as I

x = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)
y = I.exp_decay(x, 12.0, 8.0, 0.2)          # your fitted curve

# 1. Is R² higher than noise would give?
r = I.null_r2(x, y, k=3, iters=2000, seed=0)
print(r.r2, r.null_q90, r.p_value)           # -> high, low-ish, tiny

# 2. Do individual parameters survive dropping one point?
lv = I.loo_spread(x, y)
print(lv.spread, lv.passed)                  # per-parameter relative spread

# 3. Is the curve invertible where you need it?
c = I.conditioning(x, y, x0=13.0)
print(c.rel_err, c.thresh)                   # relative error vs 15% threshold

# 4. How many seeds does an effect of this size actually need?
n = I.required_n(effect=2.0, sd_control=1.0, sd_treat=1.0)
print(n)                                      # ~4
```

Or run all three curve-level checks at once:

```python
null, loo, cond = I.report(x, y, k=3, iters=2000)
```

A runnable end-to-end example is in [`examples/demo.py`](examples/demo.py) — it produces the narrative behind the figure above.

---

## API

### `exp_decay(t, A, tau, C)`

The default model, `A * exp(-t / tau) + C`. Pass your own function as `model=` anywhere the parameter appears; it must have the signature `model(x, *params)`.

### `null_r2(x, y, k=3, iters=2000, seed=0, design_x=None, bounded=False)`

Fits `iters` independent noise series with the same model and the same design, and returns:

| Field | Meaning |
|---|---|
| `r2` | R² of your actual fit |
| `null_median`, `null_q90` | Where noise lands |
| `p_value` | Fraction of null fits at least as good as yours |
| `n`, `k`, `iters` | For the record |

**The null depends on the x design, not only on (n, k).** Pass `design_x=` to use a canonical grid instead of your observed x; the two can differ substantially.

`bounded=False` (the default) uses unconstrained least squares. **The null must use the same fitting procedure as your actual curve** — switching to bounded fitting lowers the null and makes the test too permissive.

### `loo_spread(x, y, names=None)`

Refits once per dropped point.

| Field | Meaning |
|---|---|
| `full` | Parameters from all points |
| `loo` | Parameters from each leave-one-out fit, shape `(n, k)` |
| `names` | Parameter labels |
| `.spread` | `(max - min) / |full|` per parameter |
| `.passed` | True only if every spread is below 100% |

### `conditioning(x, y, x0=None, delta_y=None, thresh=0.15, window=3)`

Error propagation for inverting the curve at `x0`: relative error is `delta_y / (slope * x0)`.

When `delta_y` is omitted, the local residual is used **with a floor of 5% of the data's spread**. Without that floor a locally straight, near-flat segment reports zero error — exactly the case where inversion is worst. This bug was caught by a unit test in this repository; see [CHANGELOG.md](CHANGELOG.md).

### `required_n(effect, sd_control, sd_treat, alpha=0.05, power=0.8, convention="exact")`

Per-group sample size for a two-sample t-test.

- `convention="exact"` uses `2 (z_{a/2} + z_{power})^2 / d^2` (15.70 at the defaults).
- `convention="paper"` uses the rounded `16 / d^2` from the paper, about 2% larger.

Returns `float("inf")` for a zero effect — no sample size detects it.

**The variance you feed in is itself estimated.** With 3 seeds it can be off by a factor of two or more, so treat the result as an order of magnitude, not a budget.

### `power_table(groups, control="none", alpha=0.05, power=0.8)`

`groups` maps a condition name to a list of measured values. Returns per-condition effect size and required n against `control`.

---

## What this is not

Scope matters more than features. This package:

- **is not** a goodness-of-fit test. It tells you whether the curve could have arisen from noise, not whether your model is the right one.
- **is not** a cross-validation library. The leave-one-out check is about parameter stability, not prediction error.
- **does not** replace reporting your data. Every check needs the actual points and, for check 4, the actual measurements per condition.
- **cannot** rescue a number that a reviewer should not trust. It is a filter, not a fix.

---

## Reproducibility

Everything here is deterministic. All tests use fixed seeds and fixed data:

```bash
pip install -e ".[test]"
pytest -q          # 17 tests, ~2 s
python examples/demo.py
```

CI runs the suite on Ubuntu and Windows across Python 3.9, 3.11 and 3.12 on every push. The badge at the top reflects the latest run.

---

## Rules of thumb

1. **Report the number of points alongside R².** Four points fitting three parameters is not evidence, whatever R² says.
2. **Compare against the null at the same (n, k).** A linear-model degrees-of-freedom formula overestimates the null for nonlinear fits, so it cannot replace the empirical distribution.
3. **Check the slope before inverting.** A quantity derived from the curve does not inherit the curve's credibility.
4. **Match the null to your fitting procedure.** Constrained fits need a constrained null.
5. **Distinguish "cannot be detected" from "does not exist."** A wide confidence interval is not a null result.

---

## Development

```
pip install -e ".[test]"
pytest -q
```

## Citation

If you use this package, please cite the paper it accompanies:

```bibtex
@misc{gong2026identifiability,
  title  = {Identifiability of the Data-Augmentation Penalty: Per-Budget Tests, Ex-Ante Prediction, and Derived Quantities},
  author = {Gong, Miaoan},
  year   = {2026},
  note   = {ChinaXiv T202609.00792, under review}
}
```

A machine-readable record is in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
