#!/usr/bin/env bash
# CoT 对比实验驱动：none / short / structured 三种推理风格，配置完全一致，串行跑。
# 每个变体 = SFT 热身 + GRPO。日志各自落盘。不要用 set -e，避免一个变体失败拖垮整链。
cd ~/grpo_mol_project
mkdir -p logs

SFT_N=6000
GRPO_N=3000
MAXLEN=160

for STYLE in none short structured; do
  echo "[$(date)] === SFT $STYLE ==="
  python -u molgrpo/train_sft_warmup.py \
    --model_name_or_path ./Qwen2.5-0.5B-Instruct \
    --output_dir outputs/sft_${STYLE} \
    --library_path data/library_aug.csv \
    --cot_style ${STYLE} --n_samples ${SFT_N} --num_train_epochs 1 \
    > logs/sft_${STYLE}.log 2>&1
  echo "[$(date)] === GRPO $STYLE ==="
  python -u molgrpo/train_grpo.py \
    --model_name_or_path outputs/sft_${STYLE} \
    --output_dir outputs/molgrpo_${STYLE} \
    --library_path data/library_aug.csv \
    --num_samples ${GRPO_N} \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
    --num_generations 2 --max_completion_length ${MAXLEN} \
    --gradient_checkpointing --temperature 1.0 \
    > logs/grpo_${STYLE}.log 2>&1
  echo "[$(date)] === EVAL $STYLE ==="
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${STYLE} \
    --num_examples 200 --batch_size 8 --max_new_tokens ${MAXLEN} \
    --out outputs/eval_${STYLE}_greedy.json > logs/eval_${STYLE}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${STYLE} \
    --num_examples 200 --batch_size 8 --do_sample --max_new_tokens ${MAXLEN} \
    --out outputs/eval_${STYLE}_sample.json > logs/eval_${STYLE}_sample.log 2>&1
  # 省磁盘：删掉该变体的 SFT 中间产物（保留最终 GRPO 模型）
  rm -rf outputs/sft_${STYLE}/checkpoint-* 2>/dev/null
done
echo "[$(date)] ALL_DONE"
