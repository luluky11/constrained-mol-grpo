#!/usr/bin/env bash
# 第二波：利用一波任务空出的 GPU 补跑新变体（不污染一波对比，均为单卡同配置的新增实验）。
#   budget        : SMILES + 数值预算 CoT (group-contribution 加法推理)
#   selfies       : SELFIES 表示 + none(无CoT)  —— 表示轴对照
#   selfiesbudget : SELFIES + 数值预算 CoT       —— 表示×推理是否叠加
# 每个变体轮询等待其 GPU 空闲后再启动；幂等(已出 bo8 则跳过)。
cd ~/grpo_mol_project
mkdir -p logs outputs
MODEL=./Qwen2.5-1.5B-Instruct
SEED=data/cot_seed_4k.jsonl
GRPO_N=4000
MAXLEN=192

wait_gpu () {
  local g="$1"
  while true; do
    local u
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 2000 ] && break
    sleep 120
  done
  sleep 10
}

run_extra () {
  local NAME="$1"; local GPU="$2"; local REPR="$3"; local STYLE="$4"
  [ -f "outputs/eval_${NAME}_bo8.json" ] && { echo "[$(date)] skip ${NAME}"; return; }
  wait_gpu "$GPU"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export MOLGRPO_REPR="$REPR"
  echo "[$(date)][${NAME}][gpu${GPU}][repr=${REPR}][style=${STYLE}] SFT"
  python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL --output_dir outputs/sft_${NAME} \
    --cot_file $SEED --cot_style ${STYLE} --use_lora --lora_r 16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --num_train_epochs 2 \
    --learning_rate 1e-4 --max_length 640 > logs/sft_${NAME}.log 2>&1
  echo "[$(date)][${NAME}] GRPO"
  python -u molgrpo/train_grpo.py --model_name_or_path outputs/sft_${NAME} --output_dir outputs/molgrpo_${NAME} \
    --library_path data/library_aug.csv --num_samples ${GRPO_N} --use_lora --lora_r 16 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --num_generations 4 \
    --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 --beta 0.1 > logs/grpo_${NAME}.log 2>&1
  echo "[$(date)][${NAME}] EVAL"
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${NAME} --num_examples 1000 \
    --batch_size 16 --max_new_tokens ${MAXLEN} --out outputs/eval_${NAME}_greedy.json > logs/eval_${NAME}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${NAME} --num_examples 300 \
    --batch_size 4 --best_of_n 8 --max_new_tokens ${MAXLEN} --out outputs/eval_${NAME}_bo8.json > logs/eval_${NAME}_bo8.log 2>&1
  echo "[$(date)][${NAME}] DONE"
}

run_extra budget 1 smiles budget &
run_extra selfies 3 selfies none &
run_extra selfiesbudget 2 selfies budget &
wait
echo "WAVE2_DONE $(date)"
