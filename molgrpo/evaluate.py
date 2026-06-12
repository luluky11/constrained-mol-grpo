"""批量评估约束分子生成模型（条件版）。

统计：
- RDKit 合法率
- 性质条件平均满足比例 / 结构条件平均命中比例
- 全部条件满足率 (all_ok_rate)
- 多样性：唯一 canonical SMILES 数 / unique rate / 最高频分子占比
- 新颖性 novelty：合法分子中不在训练库里的占比（区分"真生成" vs "背库"）
- best-of-N：每个 prompt 采样 N 个，pass@N = 至少一个全满足的 prompt 比例

示例：
    python molgrpo/evaluate.py --model_name_or_path outputs/xxx --num_examples 1000
    python molgrpo/evaluate.py --model_name_or_path outputs/xxx --num_examples 300 --best_of_n 8
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from rdkit import Chem
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chem_utils import all_satisfied, compute_properties, property_score, smarts_score
    from .dataset import make_constraint_dataset
    from .rewards import extract_smiles_candidate, parse_smiles
except ImportError:
    from chem_utils import all_satisfied, compute_properties, property_score, smarts_score
    from dataset import make_constraint_dataset
    from rewards import extract_smiles_candidate, parse_smiles


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="outputs/Qwen2.5-0.5B-molgrpo-v6")
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--library_path", default="data/library_aug.csv")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--best_of_n", type=int, default=1, help=">1 时每个 prompt 采样 N 个，算 pass@N（强制采样）。")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--out", default="outputs/eval_molgrpo.json")
    return parser.parse_args()


def load_library_smiles(path: str) -> set:
    s = set()
    p = Path(path)
    if not p.exists():
        return s
    with p.open() as f:
        for r in csv.DictReader(f):
            if r.get("smiles"):
                s.add(r["smiles"])
    return s


def eval_one(response: str, spec: dict, lib_set: set) -> dict:
    smiles = extract_smiles_candidate(response)
    mol = parse_smiles(response)
    n_prop = len(spec.get("props", []))
    n_struct = len(spec.get("smarts", []))
    row = {"smiles": smiles, "valid": mol is not None, "n_prop": n_prop, "n_struct": n_struct}
    if mol is not None:
        props = compute_properties(mol)
        _, p_ok = property_score(props, spec)
        _, s_hit = smarts_score(mol, spec)
        cano = Chem.MolToSmiles(mol)
        row["prop_sat_ratio"] = p_ok / n_prop if n_prop else 1.0
        row["struct_hit_ratio"] = s_hit / n_struct if n_struct else 1.0
        row["all_ok"] = bool(all_satisfied(props, mol, spec))
        row["canonical"] = cano
        row["novel"] = cano not in lib_set
    else:
        row["prop_sat_ratio"] = 0.0
        row["struct_hit_ratio"] = 0.0
        row["all_ok"] = False
        row["canonical"] = None
        row["novel"] = False
    return row


def main() -> None:
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = max(1, args.best_of_n)
    do_sample = args.do_sample or n > 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    lib_set = load_library_smiles(args.library_path)
    ds = make_constraint_dataset(n=args.num_examples, seed=args.seed, library_path=args.library_path)

    rows = []                # 每个生成样本一行（共 num_examples*N）
    pass_at_n = []           # 每个 prompt：N 个里是否至少一个全满足
    for start in range(0, len(ds), args.batch_size):
        batch = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
        texts = [tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True) for ex in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.eos_token_id,
                      "num_return_sequences": n,
                      # 生成到 </answer> 即停：又快又干净，避免模型不发 EOS 一直啰嗦到上限。
                      "stop_strings": ["</answer>"], "tokenizer": tokenizer}
        if do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[:, inputs.input_ids.shape[-1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # decoded 顺序: [p0s0..p0s(N-1), p1s0..]
        for bi, ex in enumerate(batch):
            spec = json.loads(ex["constraints_json"])
            group = decoded[bi * n:(bi + 1) * n]
            group_rows = [eval_one(resp, spec, lib_set) for resp in group]
            for gr in group_rows:
                gr["prompt"] = ex["prompt"][1]["content"]
            rows.extend(group_rows)
            pass_at_n.append(any(gr["all_ok"] for gr in group_rows))

    total = len(rows)
    valid_rows = [r for r in rows if r["valid"]]
    cano_counts = Counter(r["canonical"] for r in rows if r["canonical"])
    most_common = cano_counts.most_common(1)[0] if cano_counts else (None, 0)
    summary = {
        "model": args.model_name_or_path,
        "num_prompts": len(ds),
        "best_of_n": n,
        "samples_total": total,
        "valid_rate": sum(r["valid"] for r in rows) / total,
        "avg_prop_sat_ratio": sum(r["prop_sat_ratio"] for r in rows) / total,
        "avg_struct_hit_ratio": sum(r["struct_hit_ratio"] for r in rows) / total,
        "all_ok_rate": sum(r["all_ok"] for r in rows) / total,
        "pass_at_n": sum(pass_at_n) / len(pass_at_n),
        "unique_smiles": len(cano_counts),
        "unique_rate": len(cano_counts) / max(1, len(valid_rows)),
        "novelty_rate": sum(r["novel"] for r in valid_rows) / max(1, len(valid_rows)),
        "top1_share": most_common[1] / max(1, len(valid_rows)),
    }

    payload = {"summary": summary, "examples": rows[:2000]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
