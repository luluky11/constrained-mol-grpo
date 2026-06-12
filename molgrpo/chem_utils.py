"""共享化学工具：性质计算、官能团 SMARTS、过滤、条件描述。

被 build_library / augment_library / dataset / rewards / train_sft_warmup 复用，
保证"反推条件"和"奖励校验"用的是同一套定义，避免训练-评估口径不一致。
"""

from __future__ import annotations

import os
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- #
# 官能团 / 结构片段：key -> (SMARTS, 自然语言描述)
# 反推条件和奖励命中都用这套定义。
# --------------------------------------------------------------------------- #
FUNCTIONAL_GROUPS: dict[str, tuple[str, str]] = {
    "benzene": ("c1ccccc1", "a benzene ring"),
    "aromatic_ring": ("[a]1[a][a][a][a][a]1", "an aromatic six-membered ring"),
    "amide": ("C(=O)N", "an amide group"),
    "carboxylic_acid": ("C(=O)[OX2H1]", "a carboxylic acid group"),
    "ester": ("C(=O)O[#6]", "an ester group"),
    "ether": ("[OD2]([#6])[#6]", "an ether linkage"),
    "hydroxyl": ("[#6][OX2H]", "a hydroxyl group"),
    "primary_amine": ("[NX3;H2][#6]", "a primary amine"),
    "tertiary_amine": ("[NX3;H0]([#6])([#6])[#6]", "a tertiary amine"),
    "halogen": ("[F,Cl,Br,I]", "at least one halogen atom"),
    "fluorine": ("[F]", "a fluorine atom"),
    "nitrogen": ("[#7]", "at least one nitrogen atom"),
    "sulfur": ("[#16]", "a sulfur atom"),
    "aromatic_nitrogen": ("[n]", "an aromatic nitrogen (heteroaromatic ring)"),
    "sulfonamide": ("S(=O)(=O)N", "a sulfonamide group"),
    "nitrile": ("C#N", "a nitrile group"),
    "ketone": ("[#6]C(=O)[#6]", "a ketone group"),
    "piperazine": ("C1CNCCN1", "a piperazine ring"),
    "morpholine": ("C1COCCN1", "a morpholine ring"),
}

# 预编译 SMARTS（None 表示编译失败，运行时跳过）。
_COMPILED_GROUPS: dict[str, Optional[Chem.Mol]] = {
    key: Chem.MolFromSmarts(smarts) for key, (smarts, _desc) in FUNCTIONAL_GROUPS.items()
}


def compute_properties(mol: Chem.Mol) -> dict:
    """计算一组常用分子性质。"""
    return {
        "molwt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "qed": QED.qed(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
    }


def matched_groups(mol: Chem.Mol) -> list[str]:
    """返回分子命中的官能团 key 列表。"""
    hits = []
    for key, (smarts, _desc) in FUNCTIONAL_GROUPS.items():
        patt = _COMPILED_GROUPS.get(key)
        if patt is None:
            patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            hits.append(key)
    return hits


def group_smarts(key: str) -> Optional[str]:
    item = FUNCTIONAL_GROUPS.get(key)
    return item[0] if item else None


def group_description(key: str) -> str:
    item = FUNCTIONAL_GROUPS.get(key)
    return item[1] if item else key


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def passes_basic_filter(
    mol: Chem.Mol,
    mw_range: tuple[float, float] = (150.0, 550.0),
    logp_range: tuple[float, float] = (-1.0, 6.0),
    qed_min: float = 0.30,
    max_heavy_atoms: int = 50,
    require_single_component: bool = True,
) -> bool:
    """drug-like 基础过滤。"""
    if mol is None:
        return False
    if require_single_component and "." in Chem.MolToSmiles(mol):
        return False
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        qed = QED.qed(mol)
    except Exception:
        return False
    if not (mw_range[0] <= mw <= mw_range[1]):
        return False
    if not (logp_range[0] <= logp <= logp_range[1]):
        return False
    if qed < qed_min:
        return False
    return True


# --------------------------------------------------------------------------- #
# 条件反推与评分（dataset / rewards / evaluate 共用，保证口径一致）
#
# 约束 spec 结构：
# {
#   "props": [
#       {"name": "molwt", "type": "range", "low": .., "high": .., "center": ..},
#       {"name": "qed",   "type": "min",   "threshold": ..},
#       {"name": "hbd",   "type": "max",   "value": ..},
#       {"name": "rings", "type": "min",   "value": ..},
#   ],
#   "smarts": ["c1ccccc1", "C(=O)N"],
# }
# 所有条件都由一个真实分子反推得到，保证该分子一定满足（存在可行解）。
# --------------------------------------------------------------------------- #

# 数值区间型性质：name -> (单位描述, 半宽采样范围, center 容差 tol)
RANGE_PROPS = {
    "molwt": ("Da", (25.0, 60.0), 60.0),
    "logp": ("", (0.5, 1.2), 1.2),
    "tpsa": ("A^2", (12.0, 30.0), 30.0),
}
# 计数型性质（取自分子真实值，方向随机），name -> tol
COUNT_PROPS = {
    "hbd": 2.0,
    "hba": 3.0,
    "rotatable": 3.0,
    "rings": 1.5,
}

_PROP_LABELS = {
    "molwt": "molecular weight (MolWt)",
    "logp": "octanol-water partition coefficient (logP)",
    "tpsa": "topological polar surface area (TPSA)",
    "qed": "QED drug-likeness",
    "hbd": "number of hydrogen bond donors",
    "hba": "number of hydrogen bond acceptors",
    "rotatable": "number of rotatable bonds",
    "rings": "number of rings",
}


def derive_constraints(
    props: dict,
    groups: list[str],
    rng: random.Random,
    n_prop_min: int = 1,
    n_prop_max: int = 3,
    n_struct_min: int = 0,
    n_struct_max: int = 3,
) -> dict:
    """从一个真实分子的性质与官能团反推一组可行约束。"""
    spec: dict = {"props": [], "smarts": []}

    # 性质条件候选：molwt/logp/tpsa(区间) + qed(min) + 计数型
    prop_pool = list(RANGE_PROPS.keys()) + ["qed"] + list(COUNT_PROPS.keys())
    k = rng.randint(n_prop_min, n_prop_max)
    chosen = rng.sample(prop_pool, min(k, len(prop_pool)))
    for name in chosen:
        if name in RANGE_PROPS:
            v = float(props[name])
            hw = rng.uniform(*RANGE_PROPS[name][1])
            # 难度旋钮：只对区间窗口半宽乘松紧系数 τ（τ<1 收窄=更难）。
            # 乘在采样之后，不消耗额外随机数 → 不同 τ 的 prompt 完全配对（同分子、同性质）。
            hw *= float(os.environ.get("MOLGRPO_TIGHTNESS", "1.0"))
            spec["props"].append(
                {"name": name, "type": "range", "low": round(v - hw, 1),
                 "high": round(v + hw, 1), "center": round(v, 1)}
            )
        elif name == "qed":
            v = float(props["qed"])
            thr = max(0.30, round(v - rng.uniform(0.03, 0.12), 2))
            spec["props"].append({"name": "qed", "type": "min", "threshold": thr})
        else:  # 计数型
            v = int(round(float(props[name])))
            direction = rng.choice(["max", "min", "exact"])
            if direction == "exact":
                spec["props"].append({"name": name, "type": "exact", "value": v})
            else:
                spec["props"].append({"name": name, "type": direction, "value": v})

    # 结构条件：从该分子真实命中的官能团里抽
    if groups:
        ks = rng.randint(n_struct_min, n_struct_max)
        ks = min(ks, len(groups))
        for key in rng.sample(groups, ks):
            sm = group_smarts(key)
            if sm:
                spec["smarts"].append(sm)

    return spec


def _range_score(value: float, low: float, high: float, center: float, tol: float) -> float:
    """区间型连续打分：中心=1，边界≈0.5，区间外按 tol 衰减到 0。"""
    hw = max(1e-6, (high - low) / 2.0)
    t = abs(value - center) / hw
    return max(0.0, 1.0 - 0.5 * t)


def _min_score(value: float, threshold: float, tol: float) -> float:
    return 1.0 if value >= threshold else max(0.0, 1.0 - (threshold - value) / tol)


def _max_score(value: float, limit: float, tol: float) -> float:
    return 1.0 if value <= limit else max(0.0, 1.0 - (value - limit) / tol)


def _exact_score(value: float, target: float, tol: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / tol)


def property_score(mol_props: dict, spec: dict) -> tuple[float, int]:
    """返回 (性质条件得分之和, 满足的条件数)。每个条件满分 1.0。"""
    total = 0.0
    satisfied = 0
    for c in spec.get("props", []):
        name = c["name"]
        v = float(mol_props[name])
        if c["type"] == "range":
            s = _range_score(v, c["low"], c["high"], c["center"], RANGE_PROPS.get(name, ("", (1, 1), 1.0))[2])
            ok = c["low"] <= v <= c["high"]
        elif c["type"] == "min":
            tol = 0.3 if name == "qed" else COUNT_PROPS.get(name, 2.0)
            thr = c.get("threshold", c.get("value"))
            s = _min_score(v, thr, tol)
            ok = v >= thr
        elif c["type"] == "max":
            s = _max_score(v, c["value"], COUNT_PROPS.get(name, 2.0))
            ok = v <= c["value"]
        else:  # exact
            s = _exact_score(v, c["value"], COUNT_PROPS.get(name, 1.5))
            ok = int(round(v)) == int(c["value"])
        total += s
        satisfied += int(ok)
    return total, satisfied


def smarts_score(mol: Chem.Mol, spec: dict) -> tuple[float, int]:
    """返回 (结构条件命中得分之和, 命中数)。每个命中 1.0。"""
    total = 0.0
    hit = 0
    for sm in spec.get("smarts", []):
        patt = Chem.MolFromSmarts(sm)
        if patt is not None and mol.HasSubstructMatch(patt):
            total += 1.0
            hit += 1
    return total, hit


def all_satisfied(mol_props: dict, mol: Chem.Mol, spec: dict) -> bool:
    _, p_ok = property_score(mol_props, spec)
    _, s_hit = smarts_score(mol, spec)
    return p_ok == len(spec.get("props", [])) and s_hit == len(spec.get("smarts", []))


def render_constraints(spec: dict) -> str:
    """把 spec 渲染成自然语言 prompt。"""
    lines = ["Design a drug-like molecule that satisfies ALL of the following constraints:"]
    for c in spec.get("props", []):
        label = _PROP_LABELS.get(c["name"], c["name"])
        if c["type"] == "range":
            unit = RANGE_PROPS.get(c["name"], ("", None, None))[0]
            unit = f" {unit}" if unit else ""
            lines.append(f"- {label} between {c['low']} and {c['high']}{unit} (ideally near {c['center']})")
        elif c["type"] == "min":
            thr = c.get("threshold", c.get("value"))
            lines.append(f"- {label} at least {thr}")
        elif c["type"] == "max":
            lines.append(f"- {label} at most {c['value']}")
        else:
            lines.append(f"- {label} equal to {c['value']}")
    for sm in spec.get("smarts", []):
        # 用描述而非裸 SMARTS，更可读
        desc = None
        for _k, (s, d) in FUNCTIONAL_GROUPS.items():
            if s == sm:
                desc = d
                break
        lines.append(f"- the molecule must contain {desc or sm}")
    lines.append("- Output only an ASCII SMILES string inside <answer>; do not output formulas or unicode subscripts")
    return "\n".join(lines)
