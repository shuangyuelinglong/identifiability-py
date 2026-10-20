# -*- coding: utf-8 -*-
"""
identifiability —— 数据增强惩罚的可辨识性工具箱

把论文 5.6 节的报告协议做成可直接调用的代码。四项检查：

    1. 零分布 (null_r2)      报告的 R² 是否显著高于同 (n, k) 的纯噪声零分布
    2. 留一法 (loo_spread)   去掉任一点重拟合，参数跨度是否远小于参数本身
    3. 条件数 (conditioning) 目标区间的局部斜率是否足以支撑反演
    4. 功效   (required_n)   用实测方差算出的所需种子数，报告值是否达到

用法见文件末尾的 __main__ 示例，或 README。
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

__all__ = ["exp_decay", "NullResult", "null_r2", "LOOResult", "loo_spread",
           "CondResult", "conditioning", "required_n", "power_table", "report"]

EPS = 1e-12


# ---------------------------------------------------------------- 模型

def exp_decay(t, A, tau, C):
    """三参数指数衰减 P(t) = A*exp(-t/tau) + C，与论文式 (1) 一致。"""
    t = np.asarray(t, dtype=float)
    return A * np.exp(-t / np.maximum(tau, EPS)) + C


def _fit(model, x, y, p0=None, bounds=None, bounded=False):
    """拟合。

    默认 bounded=False，即**无界**最小二乘——这与论文 4.x 节拟合衰减律所用
    的方式一致。这一点很关键：**零分布必须与实测曲线使用同一种拟合**，
    否则检验无效。有界拟合（A>=0）会把纯噪声的 R² 压得更低，
    使零分布偏小、检验偏松。

    置 bounded=True 可启用有界拟合（多起点、A>=0），用于对照。
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
    """自动初值：把 y 的极差当幅度、x 的跨度当时间尺度。"""
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


# ---------------------------------------------------------------- 检查 1：零分布

@dataclass
class NullResult:
    r2: float                 # 实测 R²
    null_median: float        # 零分布中位数
    null_q90: float           # 零分布 90 分位
    p_value: float            # P(纯噪声的 R² >= 实测 R²)
    n: int
    k: int
    iters: int

    @property
    def passed(self) -> bool:
        return self.r2 > self.null_q90

    def __str__(self):
        return (f"零分布  R²={self.r2:.3f}  零分布中位={self.null_median:.3f} "
                f"90分位={self.null_q90:.3f}  p={self.p_value:.4f}  "
                f"n={self.n} k={self.k}  -> {'通过' if self.passed else '**不通过**'}")


def null_r2(x, y, model=exp_decay, k=3, iters=2000, seed=0, design_x=None, bounded=False):
    """
    计算同 (n, k) 的纯噪声零分布。

    因变量取 N(0, 1)，用同一拟合器拟合，记录 R²。这样得到的分布与 1 同等重要——
    报告的 R² 只有显著高于它，才说明数据中真的存在可拟合的结构。

    design_x: 若给出，用它作为零分布的自变量设计（默认沿用实测的 x）。
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


# ---------------------------------------------------------------- 检查 2：留一法

@dataclass
class LOOResult:
    full: np.ndarray          # 全点拟合的参数
    loo: np.ndarray           # 每次去掉一个点的参数 (m, k)
    names: list = field(default_factory=lambda: ["A", "tau", "C"])

    @property
    def spread(self) -> np.ndarray:
        """每个参数的相对跨度 = (max-min)/|全点值|。"""
        s = []
        for j in range(self.full.shape[0]):
            rng = self.loo[:, j].max() - self.loo[:, j].min()
            s.append(float(rng / max(abs(self.full[j]), EPS)))
        return np.array(s)

    @property
    def passed(self) -> bool:
        """任一参数跨度超过 100% 即视为不可靠。"""
        return bool(np.all(self.spread < 1.0))

    def __str__(self):
        parts = [f"{nm}: {sp*100:.1f}%" for nm, sp in zip(self.names, self.spread)]
        return ("留一法  相对跨度  " + "  ".join(parts)
                + f"  -> {'通过' if self.passed else '**不通过**'}")


def loo_spread(x, y, model=exp_decay, names=None):
    """去掉任一点重拟合，看参数跨度。跨度远小于参数本身才说明拟合稳定。"""
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


# ---------------------------------------------------------------- 检查 3：条件数

@dataclass
class CondResult:
    x0: float                 # 目标点
    slope: float              # 局部斜率 (dY/dX)
    rel_err: float            # 反演量的相对误差 = delta_y / (slope * x)
    thresh: float

    @property
    def passed(self) -> bool:
        return self.rel_err < self.thresh

    def __str__(self):
        return (f"条件数  x={self.x0:g}  局部斜率={self.slope:.4g}  "
                f"相对误差={self.rel_err*100:.1f}%  判据<{self.thresh*100:.0f}%  "
                f"-> {'通过' if self.passed else '**不通过**'}")


def conditioning(x, y, x0=None, delta_y=None, thresh=0.15, window=3):
    """
    误差传播：相对误差 = delta_y / (slope * x)。

    任何需要通过反演校准曲线才能得到的量，都应先检查目标区间的斜率。
    平坦区间的反演结果不可用，无论输入测得多准。
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
        # 无外部误差时用局部残差估计 y 的测量精度，但设一个下限。
        #
        # 为什么必须设下限：在曲线的饱和段，局部几个点几乎共线，拟合残差
        # 趋近于 0。若直接拿残差当精度，相对误差会被算成 0，把最不可靠的
        # 区间误判为"可靠"。斜率接近 0 时，任何微小的输入误差都会被放大，
        # 所以精度不可能优于数据整体离散度的一个固定比例（此处取 5%）。
        coef = np.polyfit(x[lo:hi], y[lo:hi], 1)
        fit = np.polyval(coef, x[lo:hi])
        resid = float(np.std(y[lo:hi] - fit))
        floor = float(np.std(y)) * 0.05
        delta_y = max(resid, floor)
    denom = abs(sl) * max(abs(x0), EPS)
    rel = float(delta_y / denom) if denom > EPS else float("inf")
    return CondResult(float(x0), float(sl), rel, thresh)


# ---------------------------------------------------------------- 检查 4：功效

def required_n(effect, sd_control, sd_treat, alpha=0.05, power=0.8, convention="exact"):
    """
    两独立样本 t 检验所需每组样本量（Cohen's d 的标准定义，用合并标准差）。

        d = |effect| / sqrt((sd_control^2 + sd_treat^2) / 2)
        n = 2 * (z_{alpha/2} + z_{power})^2 / d^2

    convention="exact"  用上式（alpha=0.05, power=0.8 时系数为 15.70）
    convention="paper"  沿用论文 3.3 节的四舍五入写法 n ~= 16 / d^2，
                        便于与论文表 3 的数值逐项对齐（约大 2%）

    注意：小样本估出的方差本身不可靠（论文 4.4 节实测可差 2 倍以上），
    该数字应读作量级参考，不是精确预算。
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
    """groups: {算子名: 该组各次运行的准确率列表}，对每个非对照算子算所需 n。"""
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


# ---------------------------------------------------------------- 汇总

def report(x, y, model=exp_decay, k=3, iters=2000, x0=None, thresh=0.15, seed=0):
    """跑完三项拟合检查并打印清单。效应量检查请单独调用 required_n / power_table。"""
    print("=" * 68)
    print("可辨识性检查（论文 5.6 节协议）")
    print("=" * 68)
    r1 = null_r2(x, y, model=model, k=k, iters=iters, seed=seed)
    r2 = loo_spread(x, y, model=model)
    r3 = conditioning(x, y, x0=x0, thresh=thresh)
    for r in (r1, r2, r3):
        print("  " + str(r))
    ok = r1.passed and r2.passed and r3.passed
    print()
    print("  总判定：" + ("三项通过，该拟合可作为定量论据"
                       if ok else "**有检查未通过，该数值不应作为定量论据**"))
    return r1, r2, r3


if __name__ == "__main__":
    # 用论文表 1 的真实惩罚曲线做演示
    t = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 80], float)
    cutmix = np.array([11.97, 9.18, 7.14, 9.50, 5.59, 3.17, 2.98, 1.69, 1.25, -1.15])
    mixup = np.array([2.39, 0.50, 0.97, 1.97, 0.92, 1.46, -0.51, -1.01, 0.09, -0.13])

    print("### CutMix 的惩罚曲线（效应显著、形状规整）")
    report(t, cutmix, iters=500)
    print()
    print("### MixUp 的惩罚曲线（效应测不出）")
    report(t, mixup, iters=500)

    print()
    print("=" * 68)
    print("功效检查：确立 MixUp 的惩罚需要多少种子")
    print("=" * 68)
    print("  数据取论文表 3 的实测标准差（已按 3.3 节的合并标准差公式重算）。")
    rows = [(1, 2.39, 0.27, 2.77), (2, 0.50, 2.04, 2.21), (3, 0.97, 0.54, 0.79),
            (5, 1.97, 1.06, 1.18), (8, 0.92, 0.21, 1.47)]
    print(f"  {'预算':>4}{'效应(pt)':>11}{'sd_none':>10}{'sd_mixup':>10}{'精确式':>9}{'论文式':>9}")
    for ep, eff, s1, s2 in rows:
        ne = required_n(eff, s1, s2, convention="exact")
        np_ = required_n(eff, s1, s2, convention="paper")
        print(f"  {ep:>4}{eff:>11.2f}{s1:>10.2f}{s2:>10.2f}"
              f"{int(math.ceil(ne)):>9}{int(math.ceil(np_)):>9}")
    print()
    print("  「精确式」= 2(z_a/2 + z_b)^2/d^2（本工具的默认）；")
    print("  「论文式」= 论文 3.3 节的 n ~= 16/d^2，约大 2%，论文表 3 用的是这一列。")
    print()
    print("  论文 4.4 节的验证：1/2/3 epoch 分别预测 11/290/8 个种子，")
    print("  补跑至 23 个后，该显著的两点显著、该测不出的一点仍测不出。")
