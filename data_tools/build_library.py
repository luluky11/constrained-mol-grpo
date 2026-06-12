"""从 MOSES 构建一个分布均衡的真实分子库。

流程：
1. 流式读取 MOSES SMILES，做 drug-like 过滤、canonical 去重。
2. 按分子量分桶分层采样到约 10000，保证 MW 覆盖面。
3. 统计官能团分布，对稀有官能团定向补强约 1000。
4. 输出 data/library.csv（含性质与官能团命中列）。

用法：
    python molgrpo/build_library.py --src data/moses.csv --out data/library.csv
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "molgrpo"))

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

try:
    from .chem_utils import (
        FUNCTIONAL_GROUPS,
        compute_properties,
        matched_groups,
        passes_basic_filter,
    )
except ImportError:
    from chem_utils import (
        FUNCTIONAL_GROUPS,
        compute_properties,
        matched_groups,
        passes_basic_filter,
    )

# 分子量分桶边界。MOSES 本身被过滤在约 250-350，所以这里用贴合该范围的细分桶；
# 更高分子量（350-550）的覆盖由后续 augment_library.py 的拼接扩增来补。
MW_BUCKETS = [(150, 260), (260, 285), (285, 305), (305, 325), (325, 345), (345, 550)]


def mw_bucket(mw: float) -> int:
    for i, (lo, hi) in enumerate(MW_BUCKETS):
        if lo <= mw < hi:
            return i
    return len(MW_BUCKETS) - 1 if mw >= MW_BUCKETS[-1][0] else 0


def iter_smiles(src: Path):
    with src.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # MOSES 是 SMILES,SPLIT
        for row in reader:
            if row:
                yield row[0]


def build(args: argparse.Namespace) -> None:
    src = Path(args.src)
    rng = random.Random(args.seed)

    # 先把所有 SMILES 读进来再打乱，保证分层采样不偏向文件前部。
    all_smiles = list(iter_smiles(src))
    rng.shuffle(all_smiles)
    print(f"total source smiles: {len(all_smiles)}")

    per_bucket_target = args.target // len(MW_BUCKETS)
    bucket_count: dict[int, int] = defaultdict(int)
    seen: set[str] = set()
    records: list[dict] = []

    processed = 0
    for smiles in all_smiles:
        if len(records) >= args.target:
            break
        if processed >= args.max_scan:
            break
        processed += 1
        mol = Chem.MolFromSmiles(smiles)
        if not passes_basic_filter(mol):
            continue
        cano = Chem.MolToSmiles(mol)
        if cano in seen:
            continue
        props = compute_properties(mol)
        b = mw_bucket(props["molwt"])
        if bucket_count[b] >= per_bucket_target + args.bucket_slack:
            continue
        seen.add(cano)
        bucket_count[b] += 1
        groups = matched_groups(mol)
        records.append({"smiles": cano, "groups": groups, **props})

    print(f"scanned {processed}, collected {len(records)} (stage1)")

    # ---- 分布分析 ---- #
    group_counter: Counter = Counter()
    for r in records:
        group_counter.update(r["groups"])
    print("functional group coverage (stage1):")
    for key in FUNCTIONAL_GROUPS:
        print(f"  {key}: {group_counter.get(key, 0)}")

    # ---- 定向补强稀有官能团 ---- #
    rare_threshold = max(1, int(0.05 * len(records)))
    rare_groups = [k for k in FUNCTIONAL_GROUPS if group_counter.get(k, 0) < rare_threshold]
    print(f"rare groups (<{rare_threshold}): {rare_groups}")

    added = 0
    if rare_groups and args.reinforce > 0:
        for smiles in all_smiles:
            if added >= args.reinforce:
                break
            mol = Chem.MolFromSmiles(smiles)
            if not passes_basic_filter(mol):
                continue
            cano = Chem.MolToSmiles(mol)
            if cano in seen:
                continue
            groups = matched_groups(mol)
            if not any(g in rare_groups for g in groups):
                continue
            props = compute_properties(mol)
            seen.add(cano)
            records.append({"smiles": cano, "groups": groups, **props})
            added += 1
    print(f"reinforced {added} molecules for rare groups")

    # ---- 写出 ---- #
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["smiles", "molwt", "logp", "qed", "tpsa", "hbd", "hba", "rotatable", "rings", "groups"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in fieldnames if k != "groups"}
            row["groups"] = "|".join(r["groups"])
            writer.writerow(row)
    print(f"wrote {len(records)} molecules to {out}")

    # 最终分布
    final_groups: Counter = Counter()
    final_buckets: Counter = Counter()
    for r in records:
        final_groups.update(r["groups"])
        final_buckets[mw_bucket(r["molwt"])] += 1
    print("final MW buckets:", {f"{MW_BUCKETS[i][0]}-{MW_BUCKETS[i][1]}": final_buckets.get(i, 0) for i in range(len(MW_BUCKETS))})
    print("final group coverage:")
    for key in FUNCTIONAL_GROUPS:
        print(f"  {key}: {final_groups.get(key, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/moses.csv")
    parser.add_argument("--out", default="data/library.csv")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--reinforce", type=int, default=1000)
    parser.add_argument("--bucket_slack", type=int, default=400)
    parser.add_argument("--max_scan", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
