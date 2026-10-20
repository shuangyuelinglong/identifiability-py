# identifiability

[![tests](https://github.com/shuangyuelinglong/identifiability-py/actions/workflows/tests.yml/badge.svg)](https://github.com/shuangyuelinglong/identifiability-py/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![status](https://img.shields.io/badge/status-v1.0.0-brightgreen)](CHANGELOG.md)

[English](README.md) | **简体中文**

> **R² 高，不代表拟合可信。**

四项检查，回答同一个问题：这条拟合出来的曲线——以及从它派生出的数字——在你手头的数据下，真的可辨识吗？

![四项检查作用于一条真实曲线](docs/four_checks.png)

*一条真实的惩罚曲线（10 个点）。它看起来像干净的衰减，R² = 0.92。检查 1 通过：曲线是真的。检查 2 不通过：去掉一个点，某个参数移动 1017%。检查 3 在十处中有四处不通过：那里无法反演。检查 4 说，0.9 pt 的效应配 1.5 pt 的噪声需要 44 个种子，不是 3 个。*

---

## 安装

```bash
pip install git+https://github.com/shuangyuelinglong/identifiability-py.git
```

需要本机有 `git`。**尚未发布到 PyPI**，所以 `pip install identifiability` 会失败。如果 git 不可用，可下载源码包再装：

```bash
# 下载源码包后
pip install identifiability-py-main.tar.gz
```

两个运行时依赖：`numpy` 与 `scipy`。不需要训练、不需要显卡、不需要下载数据。Python 3.9+。

---

## 为什么需要它

假设你用 3 参数衰减律拟合 4 个数据点，得到 R² = 0.99。这是证据吗？

- **未必。** 纯噪声用 3 参数拟合，n = 10 时 R² 的 90 分位就有 0.43；n = 4 时有 46% 的概率超过 0.5。
- **即便曲线是真的**，个别参数也可能完全不受约束——去掉一个点，某个参数可能移动 1000%。
- **而且任何反演出来的量**（盈亏平衡点、成本倍数）都继承局部斜率，而你需要的那个位置斜率可能接近 0。

每一项都有一个便宜的检查。这就是本包要做的事。

---

## 四项检查

| # | 检查 | 函数 | 通过条件 |
|---|---|---|---|
| 1 | **零分布** | `null_r2` | R² 明显高于同 (n, k) 下纯噪声能达到的水平 |
| 2 | **留一法** | `loo_spread` | 去掉任一点，参数几乎不动 |
| 3 | **条件数** | `conditioning` | 局部斜率够陡，反演是稳定的 |
| 4 | **功效** | `required_n` | 你实际跑的种子数达到了这个效应所需的量 |

**任何一项不通过，那个数字就不应被当作定量论据。**

---

## 快速上手

```python
import numpy as np
import identifiability as I

x = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)
y = I.exp_decay(x, 12.0, 8.0, 0.2)          # 你拟合出的曲线

# 1. R² 是否高于噪声？
r = I.null_r2(x, y, k=3, iters=2000, seed=0)
print(r.r2, r.null_q90, r.p_value)

# 2. 参数能否经受住去掉任意一点？
lv = I.loo_spread(x, y)
print(lv.spread, lv.passed)                  # 各参数的相对跨度

# 3. 在你需要的位置，曲线可逆吗？
c = I.conditioning(x, y, x0=13.0)
print(c.rel_err, c.thresh)                   # 相对误差 vs 15% 判据

# 4. 这个量级的效应实际需要多少种子？
n = I.required_n(effect=2.0, sd_control=1.0, sd_treat=1.0)
print(n)                                      # 约 4
```

或者一次性跑完三项曲线检查：

```python
null, loo, cond = I.report(x, y, k=3, iters=2000)
```

可运行的完整示例见 [`examples/demo.py`](examples/demo.py)——它就是上面那张图背后的过程。

---

## API

### `exp_decay(t, A, tau, C)`

默认模型，`A * exp(-t / tau) + C`。任何带 `model=` 参数的地方都可以换成你自己的函数，签名需为 `model(x, *params)`。

### `null_r2(x, y, k=3, iters=2000, seed=0, design_x=None, bounded=False)`

用同一模型、同一设计，拟合 `iters` 条独立噪声序列，返回：

| 字段 | 含义 |
|---|---|
| `r2` | 实测拟合的 R² |
| `null_median`、`null_q90` | 噪声能达到的位置 |
| `p_value` | 零分布中不逊于实测 R² 的比例 |
| `n`、`k`、`iters` | 留档 |

**零分布取决于 x 的设计，不只是 (n, k)。** 可用 `design_x=` 传入一个标准网格；两者可能差很多。

`bounded=False`（默认）使用无界最小二乘。**零分布必须与实测曲线使用同一种拟合方式**——改成有界拟合会压低零分布，使检验过于宽松。

### `loo_spread(x, y, names=None)`

每去掉一个点重拟合一次。

| 字段 | 含义 |
|---|---|
| `full` | 全部点拟合的参数 |
| `loo` | 各次留一拟合的参数，形状 `(n, k)` |
| `names` | 参数名 |
| `.spread` | 每个参数的 `(max - min) / |full|` |
| `.passed` | 所有参数的跨度都低于 100% 才为真 |

### `conditioning(x, y, x0=None, delta_y=None, thresh=0.15, window=3)`

在 `x0` 处反演曲线时的误差传播：相对误差 = `delta_y / (slope * x0)`。

不给 `delta_y` 时用局部残差估计，**但设有 5% 的离散度下限**。没有这个下限，局部平直的一段会报出零误差——而那恰恰是反演最差的区间。**这个 bug 是本仓库的单元测试抓到的**，见 [CHANGELOG.md](CHANGELOG.md)。

### `required_n(effect, sd_control, sd_treat, alpha=0.05, power=0.8, convention="exact")`

两独立样本 t 检验的每组样本量。

- `convention="exact"` 用 `2 (z_{a/2} + z_{power})^2 / d^2`（默认参数下系数为 15.70）
- `convention="paper"` 用论文中的四舍五入写法 `16 / d^2`，约大 2%

效应恰为 0 时返回 `float("inf")`——再多样本也测不出。

**你传入的方差本身也是估出来的。** 3 个种子估的方差可能差 2 倍以上，所以这个数字应读作量级参考，不是精确预算。

### `power_table(groups, control="none", alpha=0.05, power=0.8)`

`groups` 把条件名映射到该条件的实测值列表，返回各条件相对 `control` 的效应量与所需 n。

---

## 它不是什么

**边界比功能更重要。** 本包：

- **不是**拟合优度检验。它只回答"这条曲线会不会来自噪声"，不回答"你的模型对不对"。
- **不是**交叉验证库。留一法检查的是参数稳定性，不是预测误差。
- **不能**替代报数据。每一项检查都需要真实的点；第 4 项还需要每个条件下的真实重复测量。
- **救不了**一个本就不该被信任的数字。它是筛子，不是修理工。

---

## 可复现性

全部确定性：测试用固定种子与固定数据。

```bash
pip install -e ".[test]"
pytest -q          # 17 个测试，约 2 秒
python examples/demo.py
```

CI 在每次推送时于 Ubuntu 与 Windows 上、Python 3.9 / 3.11 / 3.12 六个组合下跑全套测试。顶部的徽章反映最近一次结果。

---

## 五条经验

1. **报 R² 时一并报点数。** 四个点拟合三个参数，无论 R² 多高都不构成证据。
2. **零分布要用相同的 (n, k)。** 线性模型的自由度公式会高估非线性拟合的零分布，不能替代经验分布。
3. **反演之前先看斜率。** 派生量不会继承曲线的可信度。
4. **零分布要与你的拟合方式匹配。** 有界拟合就需要有界的零分布。
5. **分清"测不出"与"不存在"。** 置信区间宽不是零结果。

---

## 开发

```
pip install -e ".[test]"
pytest -q
```

## 引用

如果你使用了本包，请引用它所配套的论文：

```bibtex
@misc{gong2026identifiability,
  title  = {Identifiability of the Data-Augmentation Penalty: Per-Budget Tests, Ex-Ante Prediction, and Derived Quantities},
  author = {Gong, Miaoan},
  year   = {2026},
  note   = {ChinaXiv T202609.00792, under review}
}
```

机器可读的记录见 [`CITATION.cff`](CITATION.cff)。

## 许可

MIT，见 [LICENSE](LICENSE)。
