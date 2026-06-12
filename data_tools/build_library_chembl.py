"""从 ChEMBL chemreps 构建"全真实、宽分子量"的分子库。

ChEMBL 覆盖宽 MW（不像 MOSES 被卡在 250-350），适合做 all-real 库。
流程：
1. 流式读 chembl_37_chemreps.txt.gz（制表符：chembl_id, canonical_smiles, inchi, inchikey）。
2. drug-like 过滤（宽 MW 200-600、单组分、QED>=0.3）。
3. 按 MW 分桶分层采样到 ~3000。
4. 定向补强 ~1000：高分子量(>=400) + 稀有官能团。
5. 输出 data/library_v3.csv（全 source=real）。

用法：
    python molgrpo/build_library_chembl.py --src data/chembl_37_chemreps.txt.gz --out data/library_v3.csv
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "molgrpo"))

import argparse
import csv
import gzip
import random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

try:
    from .chem_utils import FUNCTIONAL_GROUPS, compute_properties, matched_groups, passes_basic_filter
except ImportError:
    from chem_utils import FUNCTIONAL_GROUPS, compute_properties, matched_groups, passes_basic_filter

MW_BUCKETS = [(200, 260), (260, 320), (320, 380), (380, 440), (440, 500), (500, 600)]


def mw_bucket(mw: float) -> int:
    for i, (lo, hi) in enumerate(MW_BUCKETS):
        if lo <= mw < hi:
            return i
    return -1


def iter_smiles(src: Path):
    # ZINC250k 风格 csv：smiles,logP,qed,SAS（smiles 列可能带引号/换行）
    if src.suffix == ".csv":
        with src.open() as f:
            for row in csv.DictReader(f):
                s = (row.get("smiles") or "").strip()
                if s:
                    yield s
        return
    # ChEMBL chemreps：制表符 chembl_id\tcanonical_smiles\t...
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                yield parts[1]


def wide_filter(mol):
    return passes_basic_filter(mol, mw_range=(200.0, 600.0), logp_range=(-2.0, 7.0),
                               qed_min=0.30, max_heavy_atoms=60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/chembl_37_chemreps.txt.gz")
    ap.add_argument("--out", default="data/library_v3.csv")
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--reinforce", type=int, default=1000)
    ap.add_argument("--bucket_slack", type=int, default=150)
    ap.add_argument("--max_scan", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    print("loading smiles...")
    alls = list(iter_smiles(src))
    print("total chembl smiles:", len(alls))
    rng.shuffle(alls)

    per = args.target // len(MW_BUCKETS)
    bcount = defaultdict(int)
    seen = set()
    recs = []
    scanned = 0
    for smi in alls:
        if len(recs) >= args.target or scanned >= args.max_scan:
            break
        scanned += 1
        mol = Chem.MolFromSmiles(smi)
        if not wide_filter(mol):
            continue
        cano = Chem.MolToSmiles(mol)
        if cano in seen:
            continue
        props = compute_properties(mol)
        b = mw_bucket(props["molwt"])
        if b < 0 or bcount[b] >= per + args.bucket_slack:
            continue
        seen.add(cano); bcount[b] += 1
        recs.append({"smiles": cano, "groups": matched_groups(mol), **props})
    print(f"stage1: scanned {scanned}, collected {len(recs)}")

    # 补强：高 MW(>=400) + 稀有官能团
    gc = Counter()
    for r in recs:
        gc.update(r["groups"])
    rare_thr = max(1, int(0.05 * len(recs)))
    rare = {k for k in FUNCTIONAL_GROUPS if gc.get(k, 0) < rare_thr}
    print("rare groups:", sorted(rare))
    added = 0
    for smi in alls:
        if added >= args.reinforce:
            break
        mol = Chem.MolFromSmiles(smi)
        if not wide_filter(mol):
            continue
        cano = Chem.MolToSmiles(mol)
        if cano in seen:
            continue
        props = compute_properties(mol)
        groups = matched_groups(mol)
        want = props["molwt"] >= 400 or any(g in rare for g in groups)
        if not want:
            continue
        seen.add(cano)
        recs.append({"smiles": cano, "groups": groups, **props})
        added += 1
    print(f"reinforced {added}")

    out = Path(args.out)
    fields = ["smiles", "molwt", "logp", "qed", "tpsa", "hbd", "hba", "rotatable", "rings", "groups", "source"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in recs:
            row = {k: r[k] for k in fields if k not in ("groups", "source")}
            row["groups"] = "|".join(r["groups"]); row["source"] = "real"
            w.writerow(row)
    print(f"wrote {len(recs)} to {out}")
    fb = Counter(mw_bucket(r["molwt"]) for r in recs)
    print("MW buckets:", {f"{MW_BUCKETS[i][0]}-{MW_BUCKETS[i][1]}": fb.get(i, 0) for i in range(len(MW_BUCKETS))})


if __name__ == "__main__":
    main()
