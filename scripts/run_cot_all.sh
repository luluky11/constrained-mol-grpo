#!/usr/bin/env bash
# 幂等可续跑的 CoT 对比总驱动。
# 变体：none / short / structured（程序化）+ gemini（教师蒸馏，需 data/cot_gemini.jsonl）。
# 已产出 eval 的变体自动跳过；已存在的模型自动复用。安全可重复执行。
cd ~/grpo_mol_project
mkdir -p logs outputs

SFT_N=6000
GRPO_N=3000
MAXLEN=160

run_variant () {
  local STYLE="$1"        # 标签
  local SFT_ARGS="$2"     # 传给 train_sft_warmup 的差异化参数
  local SFT_DIR="outputs/sft_${STYLE}"
  local GRPO_DIR="outputs/molgrpo_${STYLE}"

  if [ -f "outputs/eval_${STYLE}_sample.json" ]; then
    echo "[$(date)] skip ${STYLE}: eval exists"; return
  fi
  if [ ! -f "${SFT_DIR}/model.safetensors" ]; then
    echo "[$(date)] === SFT ${STYLE} ==="
    python -u molgrpo/train_sft_warmup.py --model_name_or_path ./Qwen2.5-0.5B-Instruct \
      --output_dir ${SFT_DIR} ${SFT_ARGS} --num_train_epochs 1 > logs/sft_${STYLE}.log 2>&1
  fi
  if [ ! -f "${GRPO_DIR}/model.safetensors" ]; then
    echo "[$(date)] === GRPO ${STYLE} ==="
    python -u molgrpo/train_grpo.py --model_name_or_path ${SFT_DIR} \
      --output_dir ${GRPO_DIR} --library_path data/library_aug.csv --num_samples ${GRPO_N} \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --num_generations 2 \
      --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 \
      > logs/grpo_${STYLE}.log 2>&1
  fi
  echo "[$(date)] === EVAL ${STYLE} ==="
  python -u molgrpo/evaluate.py --model_name_or_path ${GRPO_DIR} --num_examples 200 --batch_size 8 \
    --max_new_tokens ${MAXLEN} --out outputs/eval_${STYLE}_greedy.json > logs/eval_${STYLE}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path ${GRPO_DIR} --num_examples 200 --batch_size 8 \
    --do_sample --max_new_tokens ${MAXLEN} --out outputs/eval_${STYLE}_sample.json > logs/eval_${STYLE}_sample.log 2>&1
  rm -rf ${SFT_DIR}/checkpoint-* 2>/dev/null
}

run_variant none       "--library_path data/library_aug.csv --cot_style none --n_samples ${SFT_N}"
run_variant short      "--library_path data/library_aug.csv --cot_style short --n_samples ${SFT_N}"
run_variant structured "--library_path data/library_aug.csv --cot_style structured --n_samples ${SFT_N}"
if [ -f data/cot_gemini.jsonl ]; then
  run_variant gemini   "--cot_file data/cot_gemini.jsonl"
fi

echo "[$(date)] === COMPARISON TABLE ==="
python -u analysis/compare_results.py | tee logs/comparison_table.md
echo "[$(date)] ALL_DONE"
