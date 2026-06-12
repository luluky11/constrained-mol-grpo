"""训练后推理与逐条约束校验（条件版）。

示例：
    python molgrpo/infer.py --model_name_or_path outputs/xxx --num_examples 5
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chem_utils import all_satisfied, compute_properties, property_score, smarts_score
    from .dataset import make_constraint_dataset
    from .rewards import extract_smiles_candidate, extract_xml_answer, parse_smiles
except ImportError:
    from chem_utils import all_satisfied, compute_properties, property_score, smarts_score
    from dataset import make_constraint_dataset
    from rewards import extract_smiles_candidate, extract_xml_answer, parse_smiles


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="outputs/Qwen2.5-0.5B-molgrpo-v3-after-sft")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--library_path", default="data/library_aug.csv")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    ds = make_constraint_dataset(n=args.num_examples, seed=args.seed, library_path=args.library_path)
    for i, ex in enumerate(ds):
        spec = json.loads(ex["constraints_json"])
        print("=" * 80)
        print(f"Example {i + 1}")
        print(ex["prompt"][1]["content"])

        text = tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["</answer>"],
                tokenizer=tokenizer,
            )
        new_tokens = output_ids[:, inputs.input_ids.shape[-1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        smiles = extract_smiles_candidate(response)
        mol = parse_smiles(response)

        print("\nModel response:")
        print(response)
        print(f"\nExtracted <answer>: {extract_xml_answer(response)!r}")
        print(f"RDKit candidate: {smiles!r}")
        if mol is None:
            print("RDKit: INVALID")
            continue
        props = compute_properties(mol)
        _, p_ok = property_score(props, spec)
        _, s_hit = smarts_score(mol, spec)
        print("RDKit: valid")
        print(f"props: MolWt={props['molwt']:.1f} logP={props['logp']:.2f} QED={props['qed']:.3f} "
              f"TPSA={props['tpsa']:.1f} HBD={props['hbd']} HBA={props['hba']} "
              f"RotB={props['rotatable']} Rings={props['rings']}")
        print(f"property constraints satisfied: {p_ok}/{len(spec.get('props', []))}")
        print(f"structural constraints hit: {s_hit}/{len(spec.get('smarts', []))}")
        print(f"ALL satisfied: {all_satisfied(props, mol, spec)}")


if __name__ == "__main__":
    main()
