"""前提验证：用"真实可行解密度"这把统一尺子，比较不同库的约束有效宽窄。

对每个库：按与训练完全相同的方式反推约束（同样的半宽 hw 分布、同样的性质），
然后统计"本库里有多少比例的分子同时落进这些窗口" = 解密度。
密度越低 = 有效窗口越窄 = 任务越难。

纯 CPU、只用 CSV 里的数值列（molwt/logp/tpsa 等），不需要 RDKit。
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics

# 与 chem_utils.RANGE_PROPS 一致的半宽采样范围
RANGE_PROPS = {
    "molwt": (25.0, 60.0),
    "logp": (0.5, 1.2),
    "tpsa": (12.0, 30.0),
}
_NUM = ("molwt", "logp", "tpsa", "qed", "hbd", "hba", "rotatable", "rings")


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(r[k]) for k in _NUM})
            except (KeyError, ValueError):
                continue
    return rows


def solution_density(lib: list[dict], rng: random.Random, trials: int, props: list[str]) -> list[float]:
    """对 trials 个随机分子各反推一组(同 props)区间约束，返回每条约束的解密度(占库比例)。"""
    n = len(lib)
    dens = []
    for _ in range(trials):
        mol = rng.choice(lib)
        windows = {}
        for name in props:
            hw = rng.uniform(*RANGE_PROPS[name])
            windows[name] = (mol[name] - hw, mol[name] + hw)
        cnt = 0
        for m in lib:
            if all(lo <= m[name] <= hi for name, (lo, hi) in windows.items()):
                cnt += 1
        dens.append(cnt / n)
    return dens


def summarize(name: str, lib: list[dict], dens: list[float]) -> None:
    mws = [m["molwt"] for m in lib]
    q = statistics.quantiles(dens, n=4)
    print(f"\n=== {name} (n={len(lib)}) ===")
    print(f"  MW: mean={statistics.mean(mws):.0f}  std={statistics.pstdev(mws):.0f}  "
          f"range=[{min(mws):.0f}, {max(mws):.0f}]")
    print(f"  解密度(占库比例): median={statistics.median(dens):.4f}  "
          f"Q1={q[0]:.4f}  Q3={q[2]:.4f}  mean={statistics.mean(dens):.4f}")
    only_seed = sum(1 for d in dens if d <= 1.0 / len(lib) + 1e-9)
    print(f"  仅种子自己满足(几乎无其它解)的比例: {only_seed / len(dens):.1%}")
    print(f"  解密度<1%的约束比例: {sum(1 for d in dens if d < 0.01) / len(dens):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", required=True, help="name=path 形式，如 v2=data/library_v2.csv")
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--props", nargs="+", default=["molwt", "logp", "tpsa"])
    args = ap.parse_args()

    print(f"用 {args.props} 三性质、同样的半宽分布反推约束，各采 {args.trials} 条，统计本库解密度。")
    for spec in args.libs:
        name, path = spec.split("=", 1)
        lib = load(path)
        rng = random.Random(args.seed)
        dens = solution_density(lib, rng, args.trials, args.props)
        summarize(name, lib, dens)


if __name__ == "__main__":
    main()
