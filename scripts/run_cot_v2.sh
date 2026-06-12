#!/usr/bin/env bash
# v2 单库 CoT 风格主线对照：6 种 reasoning 风格在 library_v2 上各 SFT+GRPO+评测。
# SFT：全部用同一份种子文件(cot_gemini_4k.jsonl)，按风格取/生成推理 → 同一批分子对，只差推理文本(最严格配对)。
#      max_length 640, epochs 2, lr 1e-4（沿用原 1.5B CoT 对照配方）。
# GRPO：library_v2, NS=4000, gen=8, maxlen=192, beta=0.1, LoRA r16（对齐 round3/难度实验）。
# 评测：greedy 1000 + bo8 300×8, max_new_tokens 192。训完即删模型省盘。
# 全新输出名(cv2_*) 避免旧 json 误跳；4 卡并行(wave1=4风格, wave2=2风格)；幂等。
cd ~/grpo_mol_project
mkdir -p logs outputs
MODEL=./Qwen2.5-1.5B-Instruct
LIB=data/library_v2.csv
SEED=data/cot_gemini_4k.jsonl
MAXLEN=192
GEN=8
NS=4000
GRPO_COMMON="--use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 --beta 0.1 --num_generations ${GEN}"

cell () { # style gpu
  local STYLE="$1" GPU="$2"
  local NAME=cv2_${STYLE}
  [ -f "outputs/eval_${NAME}_bo8.json" ] && { echo "skip ${NAME}"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "[$(date)][${NAME}] SFT"
  python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL \
    --output_dir outputs/sft_${NAME} --cot_file $SEED --cot_style ${STYLE} \
    --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 2 --learning_rate 1e-4 --max_length 640 > logs/sft_${NAME}.log 2>&1
  echo "[$(date)][${NAME}] GRPO"
  python -u molgrpo/train_grpo.py --model_name_or_path outputs/sft_${NAME} --output_dir outputs/${NAME} \
    --library_path $LIB --num_samples ${NS} ${GRPO_COMMON} > logs/grpo_${NAME}.log 2>&1
  rm -rf outputs/sft_${NAME}   # SFT 基座用完即删
  python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} --num_examples 1000 --batch_size 16 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_greedy.json > logs/eval_${NAME}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} --num_examples 300 --batch_size 4 --best_of_n 8 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_bo8.json > logs/eval_${NAME}_bo8.log 2>&1
  [ -f "outputs/eval_${NAME}_bo8.json" ] && rm -rf outputs/${NAME}   # 评测出了删模型省盘
  echo "[$(date)][${NAME}] DONE"
}

# base-1.5B 在 v2 上的"未训练直接"基线(不训练，仅评测)
base_eval () {
  [ -f "outputs/eval_cv2_base_bo8.json" ] && { echo "skip base"; return; }
  export CUDA_VISIBLE_DEVICES=$1
  python -u molgrpo/evaluate.py --model_name_or_path $MODEL --num_examples 1000 --batch_size 16 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_cv2_base_greedy.json > logs/eval_cv2_base_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path $MODEL --num_examples 300 --batch_size 4 --best_of_n 8 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_cv2_base_bo8.json > logs/eval_cv2_base_bo8.log 2>&1
}

# Wave 1: 4 风格
cell none        0 &
cell short       1 &
cell structured  2 &
cell constructive 3 &
wait
# Wave 2: 2 风格 + base 基线
cell budget 0 &
cell gemini 1 &
base_eval 2 &
wait

echo "=== v2 CoT 风格对照 greedy ===" | tee logs/table_cot_v2.md
python analysis/compare_results.py outputs/eval_cv2_none_greedy.json outputs/eval_cv2_short_greedy.json \
  outputs/eval_cv2_structured_greedy.json outputs/eval_cv2_constructive_greedy.json \
  outputs/eval_cv2_budget_greedy.json outputs/eval_cv2_gemini_greedy.json outputs/eval_cv2_base_greedy.json 2>/dev/null | tee -a logs/table_cot_v2.md
echo "--- best-of-8 ---" | tee -a logs/table_cot_v2.md
python analysis/compare_results.py outputs/eval_cv2_none_bo8.json outputs/eval_cv2_short_bo8.json \
  outputs/eval_cv2_structured_bo8.json outputs/eval_cv2_constructive_bo8.json \
  outputs/eval_cv2_budget_bo8.json outputs/eval_cv2_gemini_bo8.json outputs/eval_cv2_base_bo8.json 2>/dev/null | tee -a logs/table_cot_v2.md
echo "COT_V2_DONE $(date)" | tee -a logs/table_cot_v2.md
