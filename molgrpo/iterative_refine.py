"""反馈引导的迭代精修（self-refine）推理实验，对照 best-of-N。

对每条约束 prompt：
- 先生成一个候选分子；若未全满足，用 RDKit 算出"哪条差、差多少"，作为反馈追加成多轮对话，
  让模型修正，最多 R 轮 → 记录"第 k 轮内首次全满足"的累计成功率（refine@k）。
- 同时用同样的生成预算做 best-of-R（R 个独立采样），算 pass@R 作对照。

这测试"利用反馈定向改"是否比"盲采样"更省样本、更高命中。纯推理、不训练。

用法：
    python molgrpo/iterative_refine.py --model_name_or_path outputs/r3_none --num_examples 200 --rounds 4 --library_path data/library_v3.csv
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chem_utils import all_satisfied, compute_properties, RANGE_PROPS, group_smarts, FUNCTIONAL_GROUPS
    from .dataset import SYSTEM_PROMPT, make_constraint_dataset
    from .rewards import extract_smiles_candidate, parse_smiles
except ImportError:
    from chem_utils import all_satisfied, compute_properties, RANGE_PROPS, group_smarts, FUNCTIONAL_GROUPS
    from dataset import SYSTEM_PROMPT, make_constraint_dataset
    from rewards import extract_smiles_candidate, parse_smiles


def _g(smarts):
    for _k, (s, d) in FUNCTIONAL_GROUPS.items():
        if s == smarts:
            return d
    return smarts


def feedback(spec: dict, mol) -> str:
    """根据当前分子与约束的偏差，给出修正反馈。mol 为 None 表示非法。"""
    if mol is None:
        return "The previous answer was not a valid molecule. Output one valid molecule that satisfies all the constraints."
    props = compute_properties(mol)
    msgs = []
    for c in spec.get("props", []):
        n = c["name"]; v = props[n]
        if c["type"] == "range" and not (c["low"] <= v <= c["high"]):
            msgs.append(f"{n} is {v:.1f} but must be in [{c['low']},{c['high']}]")
        elif c["type"] == "min" and v < c.get("threshold", c.get("value", 0)):
            msgs.append(f"{n} is {v:.2f} but must be >= {c.get('threshold', c.get('value'))}")
        elif c["type"] == "max" and v > c["value"]:
            msgs.append(f"{n} is {v} but must be <= {c['value']}")
        elif c["type"] == "exact" and int(round(v)) != int(c["value"]):
            msgs.append(f"{n} is {int(round(v))} but must equal {c['value']}")
    from rdkit import Chem
    for sm in spec.get("smarts", []):
        patt = Chem.MolFromSmarts(sm)
        if patt is not None and not mol.HasSubstructMatch(patt):
            msgs.append(f"it is missing {_g(sm)}")
    if not msgs:
        return "Good. Keep it."
    return ("Your previous molecule does not satisfy: " + "; ".join(msgs)
            + ". Revise it to satisfy ALL constraints, output the corrected molecule in the same format.")


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", default="outputs/r3_none")
    p.add_argument("--num_examples", type=int, default=200)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--library_path", default="data/library_v3.csv")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--out", default="outputs/refine.json")
    return p.parse_args()


def gen(model, tokenizer, messages, args, device, n=1):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer([text], return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                         temperature=args.temperature, top_p=args.top_p,
                         num_return_sequences=n, pad_token_id=tokenizer.eos_token_id,
                         stop_strings=["</answer>"], tokenizer=tokenizer)
    new = out[:, inp.input_ids.shape[-1]:]
    return tokenizer.batch_decode(new, skip_special_tokens=True)


def main():
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True).to(device)
    model.eval()
    ds = make_constraint_dataset(n=args.num_examples, seed=args.seed, library_path=args.library_path)

    refine_first = [None] * len(ds)   # 第几轮首次全满足
    bo_first = [None] * len(ds)       # best-of-R: 第几个样本首次全满足
    for i, ex in enumerate(ds):
        spec = json.loads(ex["constraints_json"])
        # ---- 迭代精修 ----
        msgs = list(ex["prompt"])
        for r in range(args.rounds):
            resp = gen(model, tok, msgs, args, device, n=1)[0]
            mol = parse_smiles(resp)
            ok = mol is not None and all_satisfied(compute_properties(mol), mol, spec)
            if ok and refine_first[i] is None:
                refine_first[i] = r + 1
                break
            msgs = msgs + [{"role": "assistant", "content": resp},
                           {"role": "user", "content": feedback(spec, mol)}]
        # ---- best-of-R（同预算独立采样）----
        for r in range(args.rounds):
            resp = gen(model, tok, ex["prompt"], args, device, n=1)[0]
            mol = parse_smiles(resp)
            if mol is not None and all_satisfied(compute_properties(mol), mol, spec):
                bo_first[i] = r + 1
                break

    def cum(first, k):
        return sum(1 for f in first if f is not None and f <= k) / len(first)

    summary = {"model": args.model_name_or_path, "num_examples": len(ds), "rounds": args.rounds,
               "refine_at_k": {k: cum(refine_first, k) for k in range(1, args.rounds + 1)},
               "bestof_at_k": {k: cum(bo_first, k) for k in range(1, args.rounds + 1)}}
    from pathlib import Path
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
