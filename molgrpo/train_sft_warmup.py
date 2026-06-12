"""SFT 热身（条件版）：用分子库 + 反推条件构造监督样本。

每条样本：
- prompt = 从某真实分子反推的约束（1-3 性质 + 0-3 结构）
- answer = 该真实分子的 canonical SMILES（一定满足约束）

这样模型先学会"按条件输出合法且多样的 SMILES"，再进 GRPO 就不容易塌缩。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

_REPR = os.environ.get("MOLGRPO_REPR", "smiles").lower()
if _REPR == "selfies":
    import selfies as _sf


def _answer_repr(smiles: str) -> str:
    """SFT 答案表示：selfies 模式编码为 SELFIES，否则原样 SMILES。"""
    if _REPR == "selfies":
        try:
            return _sf.encoder(smiles)
        except Exception:
            return ""
    return smiles

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from .chem_utils import derive_constraints, render_constraints
    from .cot import build_reasoning
    from .dataset import SYSTEM_PROMPT
except ImportError:
    from chem_utils import derive_constraints, render_constraints
    from cot import build_reasoning
    from dataset import SYSTEM_PROMPT

_PROP_KEYS = ("molwt", "logp", "qed", "tpsa", "hbd", "hba", "rotatable", "rings")


def load_library(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rec = {"smiles": r["smiles"], "groups": [g for g in r.get("groups", "").split("|") if g]}
            for k in _PROP_KEYS:
                rec[k] = float(r[k])
            rows.append(rec)
    return rows


def _assistant(reasoning: str, smiles: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            "<reasoning>\n" f"{reasoning}\n" "</reasoning>\n"
            "<answer>\n" f"{smiles}\n" "</answer>"
        ),
    }


def build_sft_dataset_from_cot_file(tokenizer, cot_file: str, cot_style: str = "gemini", seed: int = 7) -> Dataset:
    """从同一份种子文件按指定风格构造 SFT 样本（保证四变体用相同分子对，最大化公平）。

    文件为 jsonl，每行含 constraints_json / user_prompt / smiles / props (+ reasoning)。
    - cot_style == "gemini": 直接用文件里的 reasoning（教师蒸馏）。
    - 其它风格: 用文件里的 spec/props 现场生成 none/short/structured 推理。
    """
    rng = random.Random(seed)
    rows = []
    for l in Path(cot_file).open():
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except Exception:
            continue  # 跳过损坏行（例如生成中途被读取的残行）
    records = []
    for r in rows:
        if not r.get("smiles") or not r.get("user_prompt"):
            continue
        if cot_style == "gemini":
            reasoning = r.get("reasoning", "")
            if not reasoning:
                continue
        else:
            spec = json.loads(r["constraints_json"])
            props = r["props"]
            reasoning = build_reasoning(cot_style, spec, props, rng, smiles=r["smiles"])
        ans = _answer_repr(r["smiles"])
        if not ans:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["user_prompt"]},
            _assistant(reasoning, ans),
        ]
        records.append({"text": tokenizer.apply_chat_template(messages, tokenize=False)})
    rng.shuffle(records)
    return Dataset.from_list(records)


def build_sft_dataset(tokenizer, library_path: str, n_samples: int, cot_style: str = "short", seed: int = 7) -> Dataset:
    rng = random.Random(seed)
    lib = load_library(Path(library_path))
    if not lib:
        raise RuntimeError(f"empty library: {library_path}")

    records = []
    for _ in range(n_samples):
        mol = rng.choice(lib)
        props = {k: mol[k] for k in _PROP_KEYS}
        spec = derive_constraints(props, mol["groups"], rng)
        user_prompt = render_constraints(spec)
        reasoning = build_reasoning(cot_style, spec, props, rng, smiles=mol["smiles"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\n"
                    f"{reasoning}\n"
                    "</reasoning>\n"
                    "<answer>\n"
                    f"{_answer_repr(mol['smiles'])}\n"
                    "</answer>"
                ),
            },
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        records.append({"text": text})

    rng.shuffle(records)
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="./Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/Qwen2.5-0.5B-molgrpo-sftwarm2")
    parser.add_argument("--library_path", default="data/library_aug.csv")
    parser.add_argument("--cot_style", choices=["none", "short", "structured", "gemini", "constructive", "budget"], default="short")
    parser.add_argument("--cot_file", default="", help="种子文件 jsonl；提供时四变体用同一份分子对(按 cot_style 取/生成推理)。")
    parser.add_argument("--n_samples", type=int, default=8000)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=640)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if args.cot_file:
        dataset = build_sft_dataset_from_cot_file(tokenizer, args.cot_file, cot_style=args.cot_style, seed=args.seed)
        print(f"sft samples: {len(dataset)} | from cot_file={args.cot_file} | style={args.cot_style}")
    else:
        dataset = build_sft_dataset(tokenizer, args.library_path, args.n_samples, cot_style=args.cot_style, seed=args.seed)
        print(f"sft samples: {len(dataset)} | cot_style={args.cot_style}")
    print(dataset[0]["text"][-600:])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        # LoRA + gradient_checkpointing 需要让输入 embedding 参与梯度，否则报
        # "element 0 of tensors does not require grad"。
        model.enable_input_require_grads()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    if args.use_lora:
        # 合并 LoRA 回基座并保存为完整模型，便于 GRPO 阶段直接加载再叠新 LoRA。
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved SFT warm model to {args.output_dir}")


if __name__ == "__main__":
    main()
