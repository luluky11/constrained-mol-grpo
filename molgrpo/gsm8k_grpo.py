"""GSM8K 上的 CoT vs 直接回答 对照（GRPO，小模型）。

目的：验证"CoT 是否随规模涌现"——在 0.5B/1.5B 这种小模型上，CoT 是否真的比直接回答好。
- --cot on : system 要求"逐步推理后给出 #### 答案"，允许长推理。
- --cot off: system 要求"只给 #### 答案"，不要推理。
两者奖励完全相同（答案正确性为主 + 轻格式），唯一变量是允不允许/要求推理。

用法：
    python molgrpo/gsm8k_grpo.py --cot on  --output_dir outputs/gsm8k_cot
    python molgrpo/gsm8k_grpo.py --cot off --output_dir outputs/gsm8k_direct
然后：
    python molgrpo/gsm8k_grpo.py --eval_only --model_name_or_path outputs/gsm8k_cot --cot on
"""

from __future__ import annotations

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SYS_COT = ("Solve the math problem. Think step by step, then give the final answer "
           "on the last line in the exact form '#### <integer>'.")
SYS_DIRECT = ("Solve the math problem. Output ONLY the final answer on a single line in the exact "
              "form '#### <integer>'. Do not show any reasoning or steps.")


def gold_answer(ans: str):
    if "####" not in ans:
        return None
    return ans.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str):
    # 取最后一个 #### 后的数；没有则取文本中最后一个整数
    if "####" in text:
        tail = text.split("####")[-1]
        m = re.findall(r"-?\d[\d,]*", tail)
        if m:
            return m[0].replace(",", "")
    m = re.findall(r"-?\d[\d,]*", text)
    return m[-1].replace(",", "") if m else None


def make_dataset(split, cot: bool):
    sysp = SYS_COT if cot else SYS_DIRECT
    data = load_dataset("json", data_files=f"data/gsm8k_{split}.jsonl")["train"]
    return data.map(lambda x: {
        "prompt": [{"role": "system", "content": sysp}, {"role": "user", "content": x["question"]}],
        "gold": gold_answer(x["answer"]),
    })


def correctness_reward(completions, gold, **kwargs):
    out = []
    for c, g in zip(completions, gold):
        p = extract_pred(c[0]["content"])
        out.append(2.0 if (p is not None and g is not None and p == g) else 0.0)
    return out


def format_reward(completions, **kwargs):
    return [0.3 if "####" in c[0]["content"] else 0.0 for c in completions]


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", default="./Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="outputs/gsm8k_cot")
    p.add_argument("--cot", choices=["on", "off"], default="on")
    p.add_argument("--num_samples", type=int, default=4000)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_n", type=int, default=500)
    return p.parse_args()


def evaluate(model_path, cot, n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32, trust_remote_code=True).to(device)
    model.eval()
    ds = make_dataset("test", cot == "on").select(range(n))
    correct = 0; bs = 16
    for s in range(0, len(ds), bs):
        batch = [ds[i] for i in range(s, min(s + bs, len(ds)))]
        texts = [tok.apply_chat_template(b["prompt"], tokenize=False, add_generation_prompt=True) for b in batch]
        inp = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.batch_decode(out[:, inp.input_ids.shape[-1]:], skip_special_tokens=True)
        for b, d in zip(batch, dec):
            if extract_pred(d) == b["gold"]:
                correct += 1
    acc = correct / len(ds)
    print(f"[EVAL] {model_path} cot={cot} acc={acc:.4f} ({correct}/{len(ds)})")
    return acc


def main():
    args = build_args()
    if args.eval_only:
        evaluate(args.model_name_or_path, args.cot, args.eval_n)
        return
    cot = args.cot == "on"
    ds = make_dataset("train", cot)
    ds = ds.select(range(min(args.num_samples, len(ds))))
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GRPOConfig(
        output_dir=args.output_dir, learning_rate=args.learning_rate, adam_beta1=0.9, adam_beta2=0.99,
        weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine", logging_steps=2, fp16=True, bf16=False,
        per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations, max_prompt_length=256, max_completion_length=args.max_completion_length,
        num_train_epochs=1, save_strategy="no", max_grad_norm=0.1, use_vllm=False, report_to="none",
        gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[correctness_reward, format_reward], args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.output_dir); tok.save_pretrained(args.output_dir)
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
