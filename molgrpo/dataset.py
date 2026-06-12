"""约束分子生成的训练数据集（条件反推版）。

与早期"随机区间"不同，这里每条 prompt 的约束都从分子库里的一个真实分子反推得到：
取它的性质和官能团，随机抽 1-3 个性质条件 + 0-3 个结构条件。
这样每条样本都至少有一个已知可行解，且固定的"万能分子"无法同时满足所有 prompt。

奖励函数通过 constraints_json 列拿到约束 spec（见 chem_utils 的评分函数）。
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from datasets import Dataset

try:
    from .chem_utils import derive_constraints, render_constraints
except ImportError:
    from chem_utils import derive_constraints, render_constraints

_SYSTEM_PROMPT_SMILES = """\
You are an expert medicinal chemist. Given a set of molecular property and structural \
constraints, design ONE valid molecule that satisfies all of them.

Respond strictly in the following format:
<reasoning>
...your brief reasoning about which scaffold/functional groups satisfy the constraints...
</reasoning>
<answer>
SMILES
</answer>

Rules for <answer>: output a single valid ASCII SMILES string only, no extra text, no explanation.
Do NOT output molecular formulas, IUPAC names, unicode subscripts, or condensed structures such as CH3CH2... .

Examples of valid <answer> contents:
- CC(=O)Nc1ccccc1
- CCOc1ccc(C(=O)N)cc1
- CN1CCN(C(=O)c2ccccc2)CC1
"""

_SYSTEM_PROMPT_SELFIES = """\
You are an expert medicinal chemist. Given a set of molecular property and structural \
constraints, design ONE valid molecule that satisfies all of them.

Respond strictly in the following format:
<reasoning>
...your brief reasoning about which scaffold/functional groups satisfy the constraints...
</reasoning>
<answer>
SELFIES
</answer>

Rules for <answer>: output a single SELFIES string only (tokens like [C][=Branch1][Ring2]...),
no SMILES, no extra text, no explanation.

Examples of valid <answer> contents:
- [C][C][=Branch1][C][=O][N][C][=C][C][=C][C][=C][Ring1][=Branch1]
- [C][C][O][C][=C][C][=C][Branch1][C][C][=Branch1][C][=O][N][C][=C][Ring1][#Branch1]
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_SELFIES if os.environ.get("MOLGRPO_REPR", "smiles").lower() == "selfies" else _SYSTEM_PROMPT_SMILES

_PROP_KEYS = ("molwt", "logp", "qed", "tpsa", "hbd", "hba", "rotatable", "rings")


def _load_library(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rec = {"smiles": r["smiles"], "groups": [g for g in r.get("groups", "").split("|") if g]}
            for k in _PROP_KEYS:
                rec[k] = float(r[k])
            rows.append(rec)
    return rows


def make_constraint_dataset(
    n: int = 20000,
    seed: int = 42,
    library_path: str = "data/library_aug.csv",
    n_prop_min: int = 1,
    n_prop_max: int = 3,
    n_struct_min: int = 0,
    n_struct_max: int = 3,
) -> Dataset:
    """从分子库反推约束，生成 n 条约束分子生成样本。

    Returns:
        Dataset，含列：
          - prompt: chat 格式（system + user）
          - constraints_json: 约束 spec 的 JSON 字符串（透传给奖励函数）
          - seed_smiles: 反推所用的真实分子（仅作参考，不展示给模型）
    """
    rng = random.Random(seed)
    lib = _load_library(Path(library_path))
    if not lib:
        raise RuntimeError(f"empty library: {library_path}")

    records = []
    for _ in range(n):
        mol = rng.choice(lib)
        props = {k: mol[k] for k in _PROP_KEYS}
        spec = derive_constraints(
            props, mol["groups"], rng,
            n_prop_min=n_prop_min, n_prop_max=n_prop_max,
            n_struct_min=n_struct_min, n_struct_max=n_struct_max,
        )
        user_prompt = render_constraints(spec)
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "constraints_json": json.dumps(spec),
                "seed_smiles": mol["smiles"],
            }
        )
    return Dataset.from_list(records)


if __name__ == "__main__":
    ds = make_constraint_dataset(n=5)
    for i in range(len(ds)):
        ex = ds[i]
        print("-" * 50)
        print(ex["prompt"][1]["content"])
        print("seed:", ex["seed_smiles"])
        print("spec:", ex["constraints_json"])
