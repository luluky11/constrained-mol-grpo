"""约束分子生成的奖励函数（条件版）。

奖励分三类：
1) 格式类：保证 <reasoning>/<answer> 结构与 ASCII SMILES。
2) 化学类：合法性 + 性质条件(含 center 加权) + 结构 SMARTS 命中 + 全满足奖励 + 类药。
3) 多样性：批内去重，抑制塌缩到单一"万能分子"。

约束 spec 通过 constraints_json 列以 **kwargs 透传进来（与 dataset.py 对齐）。
性质/结构评分逻辑统一在 chem_utils，保证训练-评估口径一致。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Optional

from rdkit import Chem, RDLogger

# 表示方式：smiles(默认) 或 selfies。由环境变量门控，不影响默认 SMILES 流程。
_REPR = os.environ.get("MOLGRPO_REPR", "smiles").lower()
if _REPR == "selfies":
    import selfies as _sf


def _decode_to_smiles(token: str) -> Optional[str]:
    """把 <answer> 里的 token 转成 SMILES。selfies 模式先解码。"""
    if not token:
        return None
    if _REPR == "selfies":
        try:
            smi = _sf.decoder(token)
        except Exception:
            return None
        return smi or None
    return token
from rdkit.Chem import Crippen, Descriptors, QED

try:
    from .chem_utils import all_satisfied, compute_properties, property_score, smarts_score
except ImportError:
    from chem_utils import all_satisfied, compute_properties, property_score, smarts_score

RDLogger.DisableLog("rdApp.*")

SMILES_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+-[]()=#$/\\.%")


# --------------------------------------------------------------------------- #
# 提取
# --------------------------------------------------------------------------- #
def extract_xml_answer(text: str) -> str:
    # 取第一个 <answer> 块。模型有时会在收尾后继续啰嗦出残缺的第二段，
    # 取最后一段会拿到垃圾，因此固定取第一段（更符合模型本意，对训练/评估都更稳）。
    if "<answer>" not in text:
        return ""
    after = text.split("<answer>", 1)[1]
    return after.split("</answer>")[0].strip()


def extract_smiles_candidate(text: str) -> str:
    # 优先从 <answer> 块取；模型有时放弃 XML 直接吐 SMILES，则回退到第一行非空 token。
    raw = extract_xml_answer(text).strip()
    if raw and raw.split():
        return raw.split()[0]
    for line in text.strip().splitlines():
        line = line.strip()
        if line and line.lower() not in ("user", "assistant", "system", "<reasoning>", "</reasoning>"):
            return line.split()[0]
    return ""


def parse_smiles(text: str) -> Optional[Chem.Mol]:
    smiles = _decode_to_smiles(extract_smiles_candidate(text))
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def _format_ok(text: str) -> bool:
    return (
        "<reasoning>" in text and "</reasoning>" in text
        and "<answer>" in text and "</answer>" in text
    )


def parse_smiles_gated(text: str) -> Optional[Chem.Mol]:
    """训练用：要求模型保持 <reasoning>/<answer> 格式，否则化学奖励为 0。

    这样可以防止 GRPO 把 XML 包装丢掉、直接吐 SMILES 并编造假对话。
    """
    if not _format_ok(text):
        return None
    raw = extract_xml_answer(text).strip()
    if not raw or not raw.split():
        return None
    smi = _decode_to_smiles(raw.split()[0])
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def _specs(kwargs) -> Optional[list]:
    cj = kwargs.get("constraints_json")
    if cj is None:
        return None
    return [json.loads(s) for s in cj]


# --------------------------------------------------------------------------- #
# 化学奖励
# --------------------------------------------------------------------------- #
def validity_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    return [0.5 if parse_smiles_gated(r) is not None else 0.0 for r in responses]


def smiles_charset_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        answer = extract_xml_answer(completion[0]["content"]).strip()
        parts = answer.split()
        if len(parts) != 1:
            rewards.append(0.0)
            continue
        token = parts[0]
        ok = token.isascii() and all(ch in SMILES_CHARS for ch in token) and 3 <= len(token) <= 160
        rewards.append(0.2 if ok else 0.0)
    return rewards


def property_reward_func(completions, **kwargs) -> list[float]:
    """性质条件连续打分（含 center 加权），每条约束满分 1.0。非法分子 0 分。"""
    specs = _specs(kwargs)
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for i, r in enumerate(responses):
        mol = parse_smiles_gated(r)
        if mol is None or specs is None:
            rewards.append(0.0)
            continue
        try:
            props = compute_properties(mol)
            score, _ = property_score(props, specs[i])
        except Exception:
            score = 0.0
        rewards.append(float(score))
    return rewards


def smarts_reward_func(completions, **kwargs) -> list[float]:
    """结构条件命中奖励，每命中一个 SMARTS 给 0.8。无结构条件则 0。"""
    specs = _specs(kwargs)
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for i, r in enumerate(responses):
        mol = parse_smiles_gated(r)
        if mol is None or specs is None:
            rewards.append(0.0)
            continue
        try:
            score, _ = smarts_score(mol, specs[i])
        except Exception:
            score = 0.0
        rewards.append(0.8 * float(score))
    return rewards


def all_satisfied_reward_func(completions, **kwargs) -> list[float]:
    """所有性质+结构条件全部满足，给 1.0 奖励。"""
    specs = _specs(kwargs)
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for i, r in enumerate(responses):
        mol = parse_smiles_gated(r)
        if mol is None or specs is None:
            rewards.append(0.0)
            continue
        try:
            props = compute_properties(mol)
            ok = all_satisfied(props, mol, specs[i])
        except Exception:
            ok = False
        rewards.append(1.0 if ok else 0.0)
    return rewards


def druglike_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        mol = parse_smiles_gated(r)
        if mol is None:
            rewards.append(0.0)
            continue
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
        except Exception:
            rewards.append(0.0)
            continue
        ok = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
        rewards.append(0.3 if ok else 0.0)
    return rewards


_NUM = r"(-?\d+(?:\.\d+)?)"


def consistency_reward_func(completions, **kwargs) -> list[float]:
    """过程奖励：推理里声称的 MW/logP/QED 数值需与最终分子的真实 RDKit 值一致。

    逼"推理"变成对答案有用的、可验证的陈述，而不是装饰性文本。
    无可解析的数值声称 -> 0（鼓励做出可验证的推理）。满分 0.3。
    """
    tol = {"molwt": 40.0, "logp": 1.0, "qed": 0.15}
    rewards = []
    for c in completions:
        text = c[0]["content"]
        mol = parse_smiles_gated(text)
        if mol is None:
            rewards.append(0.0)
            continue
        reasoning = ""
        if "<reasoning>" in text and "</reasoning>" in text:
            reasoning = text.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0].lower()
        try:
            props = compute_properties(mol)
        except Exception:
            rewards.append(0.0)
            continue
        claims = []
        for m in re.findall(r"(?:molwt|molecular weight|\bmw\b)[^0-9\-]{0,14}" + _NUM, reasoning):
            claims.append(("molwt", float(m)))
        for m in re.findall(r"logp[^0-9\-]{0,14}" + _NUM, reasoning):
            claims.append(("logp", float(m)))
        for m in re.findall(r"qed[^0-9\-]{0,14}" + _NUM, reasoning):
            claims.append(("qed", float(m)))
        if not claims:
            rewards.append(0.0)
            continue
        ok = sum(1 for k, v in claims if abs(props[k] - v) <= tol[k])
        rewards.append(0.3 * ok / len(claims))
    return rewards


def conciseness_reward_func(completions, **kwargs) -> list[float]:
    """收尾奖励：输出到 </answer> 后立即停止（几乎无多余文本）给 0.3。

    针对 GRPO 中模型不发 EOS、一直凑到 max_completion_length 的问题（clipped_ratio=1.0）。
    """
    rewards = []
    for c in completions:
        t = c[0]["content"]
        if "</answer>" in t:
            trailing = t.split("</answer>")[-1].strip()
            rewards.append(0.3 if len(trailing) <= 2 else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def diversity_reward_func(completions, **kwargs) -> list[float]:
    """批内去重多样性奖励：同一 batch 里某合法 SMILES 出现 k 次，则每个得 0.4/k。

    模型若塌缩到单一分子，这个奖励会被稀释；输出不同合法分子能拿满 0.4。
    无状态，依赖 GRPO 一个 batch 内多条 completions。
    """
    cano = []
    for c in completions:
        mol = parse_smiles_gated(c[0]["content"])
        cano.append(Chem.MolToSmiles(mol) if mol is not None else None)
    counts = Counter(s for s in cano if s is not None)
    rewards = []
    for s in cano:
        if s is None:
            rewards.append(0.0)
        else:
            rewards.append(0.4 / counts[s])
    return rewards


# --------------------------------------------------------------------------- #
# 格式奖励
# --------------------------------------------------------------------------- #
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [c[0]["content"] for c in completions]
    return [count_xml(c) for c in contents]
