#!/usr/bin/env bash
# 训练预算标定：共享一个 none-SFT base，只变 GRPO 的三个预算旋钮（数据/步数/探索）。
# 全部在 library_v2 上、none 风格、SMILES、LoRA，唯一变量是预算。
cd ~/grpo_mol_project
mkdir -p logs outputs
MODEL=./Qwen2.5-1.5B-Instruct
LIB=data/library_v2.csv
SFTBASE=outputs/sft_v2base
MAXLEN=192

# 1) 共享 SFT（none 风格，固定，不计入预算变量）
if [ ! -f ${SFTBASE}/config.json ]; then
  echo "[$(date)] shared SFT base"
  CUDA_VISIBLE_DEVICES=0 python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL \
    --output_dir ${SFTBASE} --library_path $LIB --cot_style none --n_samples 6000 \
    --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 1 --learning_rate 1e-4 --max_length 512 > logs/sft_v2base.log 2>&1
fi

run_grpo () { # name gpu num_samples epochs gen
  local NAME="$1" GPU="$2" NS="$3" EP="$4" GEN="$5"
  [ -f "outputs/eval_${NAME}_bo8.json" ] && { echo "skip ${NAME}"; return; }
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "[$(date)][${NAME}][gpu${GPU}] GRPO ns=${NS} ep=${EP} gen=${GEN}"
  python -u molgrpo/train_grpo.py --model_name_or_path ${SFTBASE} --output_dir outputs/molgrpo_${NAME} \
    --library_path $LIB --num_samples ${NS} --num_train_epochs ${EP} --num_generations ${GEN} \
    --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
    --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 --beta 0.1 > logs/grpo_${NAME}.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${NAME} --num_examples 1000 \
    --batch_size 16 --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_greedy.json > logs/eval_${NAME}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${NAME} --num_examples 300 \
    --batch_size 4 --best_of_n 8 --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_bo8.json > logs/eval_${NAME}_bo8.log 2>&1
  echo "[$(date)][${NAME}] DONE"
}

run_grpo b_anchor 0 4000 1 4 &
run_grpo b_epoch  1 4000 2 4 &
run_grpo b_gen    2 4000 1 8 &
run_grpo b_data   3 8000 1 4 &
wait
echo "=== BUDGET CALIBRATION (none, library_v2) ===" | tee logs/table_budget.md
python analysis/compare_results.py outputs/eval_b_anchor_greedy.json outputs/eval_b_epoch_greedy.json outputs/eval_b_gen_greedy.json outputs/eval_b_data_greedy.json 2>/dev/null | tee -a logs/table_budget.md
echo "--- best-of-8 ---" | tee -a logs/table_budget.md
python analysis/compare_results.py outputs/eval_b_anchor_bo8.json outputs/eval_b_epoch_bo8.json outputs/eval_b_gen_bo8.json outputs/eval_b_data_bo8.json 2>/dev/null | tee -a logs/table_budget.md
echo "BUDGET_DONE $(date)" | tee -a logs/table_budget.md
