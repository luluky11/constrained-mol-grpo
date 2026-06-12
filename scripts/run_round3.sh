#!/usr/bin/env bash
# 第三轮（library_v3 全真实宽MW, gen8）：
#   r3_none      : 无 CoT 基线
#   r3_budget    : 数值预算 CoT（上一轮最优 CoT）
#   r3_budgetpr  : 数值预算 CoT + 过程奖励(推理-分子一致性)
#   r3_curric    : 课程学习（none：先单约束→再全约束，两阶段 GRPO）
# 4 卡并行；幂等(已出 bo8 跳过)。
cd ~/grpo_mol_project
mkdir -p logs outputs
MODEL=./Qwen2.5-1.5B-Instruct
LIB=data/library_v3.csv
MAXLEN=192
GEN=8
NS=4000
SFTN=outputs/sft_none_v3
SFTB=outputs/sft_budget_v3
COMMON="--use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 --beta 0.1 --num_generations ${GEN}"

sft () { # style outdir gpu
  local STYLE="$1" OUT="$2" GPU="$3"
  [ -f "${OUT}/config.json" ] && return
  CUDA_VISIBLE_DEVICES=$GPU python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL \
    --output_dir ${OUT} --library_path $LIB --cot_style ${STYLE} --n_samples 6000 \
    --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 1 --learning_rate 1e-4 --max_length 640 > logs/$(basename ${OUT}).log 2>&1
}

evald () { # name gpu
  local NAME="$1" GPU="$2"
  CUDA_VISIBLE_DEVICES=$GPU python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} \
    --num_examples 1000 --batch_size 16 --max_new_tokens ${MAXLEN} --library_path $LIB \
    --out outputs/eval_${NAME}_greedy.json > logs/eval_${NAME}_greedy.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} \
    --num_examples 300 --batch_size 4 --best_of_n 8 --max_new_tokens ${MAXLEN} --library_path $LIB \
    --out outputs/eval_${NAME}_bo8.json > logs/eval_${NAME}_bo8.log 2>&1
}

variant_simple () { # name gpu sftbase extra
  local NAME="$1" GPU="$2" SFT="$3" EXTRA="$4"
  [ -f "outputs/eval_${NAME}_bo8.json" ] && { echo "skip ${NAME}"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "[$(date)][${NAME}] GRPO"
  python -u molgrpo/train_grpo.py --model_name_or_path ${SFT} --output_dir outputs/${NAME} \
    --library_path $LIB --num_samples ${NS} ${COMMON} ${EXTRA} > logs/grpo_${NAME}.log 2>&1
  evald ${NAME} ${GPU}
  echo "[$(date)][${NAME}] DONE"
}

curriculum () { # gpu sftbase
  local GPU="$1" SFT="$2"
  [ -f "outputs/eval_r3_curric_bo8.json" ] && { echo "skip curric"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "[$(date)][curric] phase A (<=1 prop, <=1 struct)"
  python -u molgrpo/train_grpo.py --model_name_or_path ${SFT} --output_dir outputs/r3_curric_A \
    --library_path $LIB --num_samples 2000 --n_prop_max 1 --n_struct_max 1 ${COMMON} > logs/grpo_r3_curricA.log 2>&1
  echo "[$(date)][curric] phase B (full)"
  python -u molgrpo/train_grpo.py --model_name_or_path outputs/r3_curric_A --output_dir outputs/r3_curric \
    --library_path $LIB --num_samples ${NS} ${COMMON} > logs/grpo_r3_curricB.log 2>&1
  evald r3_curric ${GPU}
  echo "[$(date)][curric] DONE"
}

# 先并行两套 SFT base
sft none ${SFTN} 0 &
sft budget ${SFTB} 1 &
wait

# 4 卡并行四变体
variant_simple r3_none     0 ${SFTN} "" &
variant_simple r3_budget   1 ${SFTB} "" &
variant_simple r3_budgetpr 2 ${SFTB} "--process_reward" &
curriculum 3 ${SFTN} &
wait

echo "=== ROUND3 (library_v3) greedy ===" | tee logs/table_round3.md
python analysis/compare_results.py outputs/eval_r3_none_greedy.json outputs/eval_r3_budget_greedy.json outputs/eval_r3_budgetpr_greedy.json outputs/eval_r3_curric_greedy.json 2>/dev/null | tee -a logs/table_round3.md
echo "--- best-of-8 ---" | tee -a logs/table_round3.md
python analysis/compare_results.py outputs/eval_r3_none_bo8.json outputs/eval_r3_budget_bo8.json outputs/eval_r3_budgetpr_bo8.json outputs/eval_r3_curric_bo8.json 2>/dev/null | tee -a logs/table_round3.md
echo "ROUND3_DONE $(date)" | tee -a logs/table_round3.md
