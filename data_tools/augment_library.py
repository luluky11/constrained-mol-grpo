"""用 BRICS 片段重组把真实分子库扩增到约 20000。

思路：
1. 读入真实库（data/library.csv）。
2. 对每个分子做 BRICS 分解，汇集片段池。
3. 用 BRICS.BRICSBuild 从片段池随机重组出新分子。
4. sanitize + drug-like 过滤 + canonical 去重（与真实库不重复）。
5. 真实库 + 扩增分子合并到约 20000，输出 data/library_aug.csv。

BRICS 重组天然能拼出更大的分子，因此可以补足 MOSES 偏窄（250-350）的分子量分布。

用法：
    python molgrpo/augment_library.py --src data/library.csv --out data/library_aug.csv --target 20000
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "molgrpo"))

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

try:
    from .chem_utils import FUNCTIONAL_GROUPS, compute_properties, matched_groups
except ImportError:
    from chem_utils import FUNCTIONAL_GROUPS, compute_properties, matched_groups


def read_library(src: Path) -> list[dict]:
    rows = []
    with src.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aug_filter(mol: Chem.Mol) -> bool:
    """扩增分子的过滤：允许分子量到 600，以扩展高分子量覆盖。"""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    smi = Chem.MolToSmiles(mol)
    if "." in smi:
        return False
    if mol.GetNumHeavyAtoms() > 55:
        return False
    from rdkit.Chem import Crippen, Descriptors, QED

    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        qed = QED.qed(mol)
    except Exception:
        return False
    if not (180.0 <= mw <= 600.0):
        return False
    if not (-1.5 <= logp <= 7.0):
        return False
    if qed < 0.25:
        return False
    return True


def augment(args: argparse.Namespace) -> None:
    src = Path(args.src)
    rng = random.Random(args.seed)
    real = read_library(src)
    print(f"real molecules: {len(real)}")

    seen: set[str] = {r["smiles"] for r in real}

    # 收集全局 BRICS 片段池，并缓存每个真实分子的片段分解。
    fragment_pool: list[str] = []
    fragset: set[str] = set()
    base_frags_cache: list[list[str]] = []
    for r in real:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            base_frags_cache.append([])
            continue
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            frags = []
        base_frags_cache.append(frags)
        for f in frags:
            if f not in fragset:
                fragset.add(f)
                fragment_pool.append(f)
    print(f"collected {len(fragment_pool)} BRICS fragments from {len(real)} molecules")

    need = max(0, args.target - len(real))
    print(f"need to build ~{need} augmented molecules")

    augmented: list[dict] = []
    attempts = 0
    max_attempts = need * args.attempt_factor
    # 以随机真实分子为锚：用它的片段 + 少量随机池片段重组，保证扩增分子绑定多样骨架，避免塌缩到少数片段。
    while len(augmented) < need and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(len(real))
        base_frags = base_frags_cache[idx]
        if not base_frags:
            continue
        extra = rng.sample(fragment_pool, min(len(fragment_pool), rng.randint(2, 5)))
        frag_smis = set(base_frags) | set(extra)
        frag_mols = [Chem.MolFromSmiles(f) for f in frag_smis]
        frag_mols = [m for m in frag_mols if m is not None]
        if len(frag_mols) < 2:
            continue
        rng.shuffle(frag_mols)
        try:
            builder = BRICS.BRICSBuild(frag_mols, scrambleReagents=True, maxDepth=args.max_depth)
            mol = next(builder)
        except (StopIteration, Exception):
            continue
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        if not aug_filter(mol):
            continue
        cano = Chem.MolToSmiles(mol)
        if cano in seen:
            continue
        seen.add(cano)
        props = compute_properties(mol)
        groups = matched_groups(mol)
        augmented.append({"smiles": cano, "groups": groups, **props, "source": "aug"})

    print(f"built {len(augmented)} augmented molecules in {attempts} attempts")

    # 合并写出
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["smiles", "molwt", "logp", "qed", "tpsa", "hbd", "hba", "rotatable", "rings", "groups", "source"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in real:
            row = {k: r.get(k, "") for k in fieldnames if k not in ("groups", "source")}
            row["groups"] = r.get("groups", "")
            row["source"] = "real"
            writer.writerow(row)
        for r in augmented:
            row = {k: r[k] for k in fieldnames if k not in ("groups", "source")}
            row["groups"] = "|".join(r["groups"])
            row["source"] = "aug"
            writer.writerow(row)
    total = len(real) + len(augmented)
    print(f"wrote {total} molecules to {out}")

    # 分布
    buckets = Counter()
    for r in real:
        buckets[int(float(r["molwt"]) // 50 * 50)] += 1
    for r in augmented:
        buckets[int(r["molwt"] // 50 * 50)] += 1
    print("MW histogram (bin=50):", dict(sorted(buckets.items())))
    gc = Counter()
    for r in augmented:
        gc.update(r["groups"])
    print("augmented group coverage:")
    for key in FUNCTIONAL_GROUPS:
        print(f"  {key}: {gc.get(key, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/library.csv")
    parser.add_argument("--out", default="data/library_aug.csv")
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--attempt_factor", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    augment(parser.parse_args())


if __name__ == "__main__":
    main()
