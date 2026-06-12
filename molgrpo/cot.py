"""化学推理链（Chain-of-Thought）生成器。

用于对比实验：在 SFT 热身时给同一个 (约束 prompt -> 真实分子) 样本配上不同风格的
<reasoning> 内容，从而对比"推理链丰富度"对约束分子生成的影响。

三种风格：
- "none"       直接回答，不推理（no-CoT 基线）
- "short"      通用一句话推理（短 CoT）
- "structured" 基于真实分子 RDKit 数值的多步化学推理（Long-CoT）：
               分析约束 -> 选骨架 -> 调性质 -> 自检

structured 风格的数值全部来自种子分子的真实 RDKit 计算结果，是"grounded CoT"，
不是凭空编造，因此可作为可靠的监督信号。
"""

from __future__ import annotations

import random

try:
    from .chem_utils import FUNCTIONAL_GROUPS, RANGE_PROPS, _PROP_LABELS
except ImportError:
    from chem_utils import FUNCTIONAL_GROUPS, RANGE_PROPS, _PROP_LABELS


def _smarts_desc(smarts: str) -> str:
    for _k, (s, d) in FUNCTIONAL_GROUPS.items():
        if s == smarts:
            return d
    return "the required substructure"


def _constraint_phrases(spec: dict) -> list[str]:
    out = []
    for c in spec.get("props", []):
        label = _PROP_LABELS.get(c["name"], c["name"])
        if c["type"] == "range":
            out.append(f"{label} near {c['center']} (between {c['low']} and {c['high']})")
        elif c["type"] == "min":
            out.append(f"{label} at least {c.get('threshold', c.get('value'))}")
        elif c["type"] == "max":
            out.append(f"{label} at most {c['value']}")
        else:
            out.append(f"{label} equal to {c['value']}")
    for sm in spec.get("smarts", []):
        out.append(_smarts_desc(sm))
    return out


def _structured_reasoning(spec: dict, props: dict, rng: random.Random) -> str:
    phrases = _constraint_phrases(spec)
    constraints_txt = "; ".join(phrases) if phrases else "the given drug-like criteria"

    # Step 2: 骨架/结构选择
    smarts_descs = [_smarts_desc(s) for s in spec.get("smarts", [])]
    if smarts_descs:
        scaffold = (
            "I build the molecule around a core that provides "
            + " and ".join(smarts_descs)
            + ", which directly satisfies the structural requirement."
        )
    else:
        scaffold = (
            "I choose a common drug-like scaffold (an aromatic ring with a small "
            "aliphatic/heteroatom decoration) as the core."
        )

    # Step 3: 性质调节（引用真实数值）
    tune_bits = []
    names = {c["name"] for c in spec.get("props", [])}
    if "molwt" in names or "rings" in names or "rotatable" in names:
        tune_bits.append(
            f"the size is tuned so molecular weight lands around {props['molwt']:.0f} Da"
        )
    if "logp" in names or "tpsa" in names:
        tune_bits.append(
            f"polar groups balance lipophilicity to keep logP near {props['logp']:.1f} "
            f"and TPSA near {props['tpsa']:.0f}"
        )
    if "qed" in names:
        tune_bits.append(f"the overall shape keeps drug-likeness QED at {props['qed']:.2f}")
    if not tune_bits:
        tune_bits.append(
            f"substituents are chosen to keep MolWt {props['molwt']:.0f}, "
            f"logP {props['logp']:.1f}, QED {props['qed']:.2f} in a drug-like range"
        )
    tune = "To meet the numeric targets, " + "; ".join(tune_bits) + "."

    # Step 4: 自检（真实数值）
    verify = (
        f"Checking the result: MolWt {props['molwt']:.0f}, logP {props['logp']:.1f}, "
        f"QED {props['qed']:.2f}, TPSA {props['tpsa']:.0f}, "
        f"H-bond donors {int(props['hbd'])}, acceptors {int(props['hba'])}, "
        f"rings {int(props['rings'])}"
    )
    if smarts_descs:
        verify += ", and it contains " + " and ".join(smarts_descs)
    verify += " — all constraints are satisfied."

    step_labels = rng.choice([
        ("Analyze the constraints", "Choose a scaffold", "Tune the properties", "Verify"),
        ("Read the requirements", "Pick a core", "Adjust substituents", "Double-check"),
    ])
    return (
        f"Step 1 ({step_labels[0]}): the target asks for {constraints_txt}.\n"
        f"Step 2 ({step_labels[1]}): {scaffold}\n"
        f"Step 3 ({step_labels[2]}): {tune}\n"
        f"Step 4 ({step_labels[3]}): {verify}"
    )


_SHORT_VARIANTS = [
    "This molecule is a valid drug-like SMILES whose properties and substructures match the requested constraints.",
    "The proposed SMILES is a valid drug-like structure that fits the requested property and structural constraints.",
    "This is a valid drug-like molecule designed to match the requested molecular properties and substructures.",
]


def _constructive_reasoning(spec: dict, props: dict, smiles: str) -> str:
    """构建式 CoT：化学原生的骨架 + 片段构建步骤（非自然语言）。

    用 RDKit 从目标分子算 Murcko 骨架 + BRICS 片段，模态更贴近 SMILES，理论上更适合小模型。
    """
    from rdkit import Chem
    from rdkit.Chem import BRICS
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold = ""
    frags = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        try:
            # 去掉 BRICS 片段里的 dummy 原子标记，更像可读结构单元
            raw = list(BRICS.BRICSDecompose(mol))
            for f in raw[:4]:
                f2 = f.replace("[*]", "").replace("*", "")
                fm = Chem.MolFromSmiles(f2)
                if fm is not None:
                    frags.append(Chem.MolToSmiles(fm))
        except Exception:
            pass
    if not scaffold:
        scaffold = "a drug-like aromatic core"
    groups = [_smarts_desc(s) for s in spec.get("smarts", [])]
    group_txt = (" containing " + " and ".join(groups)) if groups else ""
    frag_txt = ", ".join(frags) if frags else "small functional substituents"
    target_mw = next((c["center"] for c in spec.get("props", []) if c.get("name") == "molwt"), round(props["molwt"]))
    return (
        f"Core scaffold: {scaffold}\n"
        f"Building blocks: {frag_txt}\n"
        f"Plan: start from the core scaffold and attach the building blocks to assemble a molecule"
        f"{group_txt}, adjusting size toward MolWt ~{target_mw}."
    )


def _budget_reasoning(spec: dict, props: dict, smiles: str) -> str:
    """数值预算 CoT：把目标 MW 拆成片段质量的可加性预算（group-contribution）。

    MW/logP 本质是片段贡献之和，让模型做"决定约束满不满足的那笔算术"。
    """
    from rdkit import Chem
    from rdkit.Chem import BRICS, Crippen, Descriptors

    target_mw = next((c["center"] for c in spec.get("props", []) if c.get("name") == "molwt"), round(props["molwt"]))
    parts = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            for f in list(BRICS.BRICSDecompose(mol))[:5]:
                f2 = f.replace("[*]", "").replace("*", "")
                fm = Chem.MolFromSmiles(f2)
                if fm is not None:
                    parts.append((Chem.MolToSmiles(fm), Descriptors.MolWt(fm), Crippen.MolLogP(fm)))
        except Exception:
            pass
    groups = [_smarts_desc(s) for s in spec.get("smarts", [])]
    group_txt = (" while including " + " and ".join(groups)) if groups else ""
    if parts:
        sum_mw = sum(p[1] for p in parts)
        budget = "; ".join(f"{p[0]} (MW {p[1]:.0f}, logP {p[2]:+.1f})" for p in parts)
        return (
            f"Budget the target MolWt ~{target_mw} by adding fragment contributions{group_txt}.\n"
            f"Building blocks and their additive contributions: {budget}.\n"
            f"Their masses sum to ~{sum_mw:.0f}, which lands near the target; "
            f"combine them into one connected molecule."
        )
    return (
        f"Budget the target MolWt ~{target_mw} additively{group_txt}: pick a core ring (~78), "
        f"a linker, and 1-2 substituents whose masses sum near the target, then connect them."
    )


def build_reasoning(style: str, spec: dict, props: dict, rng: random.Random, smiles: str = "") -> str:
    """返回 <reasoning> 标签内的文本（不含标签本身）。"""
    if style == "none":
        return "Direct answer."
    if style == "short":
        return rng.choice(_SHORT_VARIANTS)
    if style == "structured":
        return _structured_reasoning(spec, props, rng)
    if style == "constructive":
        return _constructive_reasoning(spec, props, smiles)
    if style == "budget":
        return _budget_reasoning(spec, props, smiles)
    raise ValueError(f"unknown cot style: {style}")
