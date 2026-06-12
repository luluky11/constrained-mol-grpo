"""用 GRPO 训练约束分子生成模型。

默认配置面向 Bohrium 单卡 Tesla T4 15GB：
- fp16=True，bf16=False（T4 不支持 bf16）
- use_vllm=False
- 默认 report_to=none，避免服务器联网受限时 wandb 卡住

示例：
    python molgrpo/train_grpo.py \
      --model_name_or_path ./Qwen2.5-0.5B-Instruct \
      --output_dir outputs/Qwen2.5-0.5B-molgrpo \
      --num_samples 2000
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from .dataset import make_constraint_dataset
    from .rewards import (
        all_satisfied_reward_func,
        conciseness_reward_func,
        diversity_reward_func,
        druglike_reward_func,
        property_reward_func,
        smarts_reward_func,
        smiles_charset_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        validity_reward_func,
        xmlcount_reward_func,
    )
except ImportError:
    # 允许直接 `python molgrpo/train_grpo.py` 执行。
    from dataset import make_constraint_dataset
    from rewards import (
        all_satisfied_reward_func,
        conciseness_reward_func,
        diversity_reward_func,
        druglike_reward_func,
        property_reward_func,
        smarts_reward_func,
        smiles_charset_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        validity_reward_func,
        xmlcount_reward_func,
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="./Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/Qwen2.5-0.5B-molgrpo")
    parser.add_argument("--run_name", default="Qwen2.5-0.5B-molgrpo")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--library_path", default="data/library_aug.csv")
    parser.add_argument("--n_prop_min", type=int, default=1)
    parser.add_argument("--n_prop_max", type=int, default=3)
    parser.add_argument("--n_struct_min", type=int, default=0)
    parser.add_argument("--n_struct_max", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=96)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04, help="KL 系数。调大可锚定会收尾的 SFT 模型，缓解策略漂移导致的不收尾。")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--process_reward", action="store_true", help="加入推理-分子一致性过程奖励。")
    parser.add_argument(
        "--report_to",
        default="none",
        help='默认 "none"。如果服务器可联网且已登录 wandb，可设为 "wandb"。',
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="显存紧张时打开，T4 上更稳但会慢一些。",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()

    dataset = make_constraint_dataset(
        n=args.num_samples,
        seed=args.seed,
        library_path=args.library_path,
        n_prop_min=args.n_prop_min,
        n_prop_max=args.n_prop_max,
        n_struct_min=args.n_struct_min,
        n_struct_max=args.n_struct_max,
    )
    print(f"train samples: {len(dataset)}")
    print("first prompt:")
    print(dataset[0]["prompt"][1]["content"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 注意：fp16 训练时不要把可训练权重直接加载成 torch.float16，
    # 否则 Accelerate 的 GradScaler 会在反缩放梯度时报
    # "Attempting to unscale FP16 gradients"。这里用 float32 加载，
    # 由 Trainer/AMP 在前向中自动使用 fp16。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        smiles_charset_reward_func,
        validity_reward_func,
        druglike_reward_func,
        property_reward_func,
        smarts_reward_func,
        all_satisfied_reward_func,
        diversity_reward_func,
        conciseness_reward_func,
    ]
    if args.process_reward:
        try:
            from .rewards import consistency_reward_func
        except ImportError:
            from rewards import consistency_reward_func
        reward_funcs.append(consistency_reward_func)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        fp16=True,
        bf16=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        # 不存中间 checkpoint（含优化器状态，单个就十几 G），只在结束 save_model，避免写满磁盘。
        save_strategy="no",
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    if peft_config is not None:
        # 合并 LoRA 回基座并存为完整模型：评估可直接加载，也便于课程学习链式续训。
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
