#!/usr/bin/env bash
# GSM8K 上 CoT vs 直接回答 对照（0.5B, T4 单卡顺序）。
# 先看 base(未训练) 两种 prompt 的正确率，再各 GRPO 后评估。
cd ~/grpo_mol_project
mkdir -p logs outputs
M=./Qwen2.5-0.5B-Instruct
NS=2000; GEN=6; ML=256; EVALN=500

echo "[$(date)] base cot-prompt"
python -u molgrpo/gsm8k_grpo.py --eval_only --model_name_or_path $M --cot on  --eval_n $EVALN > logs/gsm8k_base_cot.log 2>&1
echo "[$(date)] base direct-prompt"
python -u molgrpo/gsm8k_grpo.py --eval_only --model_name_or_path $M --cot off --eval_n $EVALN > logs/gsm8k_base_direct.log 2>&1

echo "[$(date)] GRPO cot"
python -u molgrpo/gsm8k_grpo.py --cot on --output_dir outputs/gsm8k_cot --num_samples $NS --num_generations $GEN --max_completion_length $ML > logs/gsm8k_train_cot.log 2>&1
python -u molgrpo/gsm8k_grpo.py --eval_only --model_name_or_path outputs/gsm8k_cot --cot on --eval_n $EVALN > logs/gsm8k_eval_cot.log 2>&1

echo "[$(date)] GRPO direct"
python -u molgrpo/gsm8k_grpo.py --cot off --output_dir outputs/gsm8k_direct --num_samples $NS --num_generations $GEN --max_completion_length $ML > logs/gsm8k_train_direct.log 2>&1
python -u molgrpo/gsm8k_grpo.py --eval_only --model_name_or_path outputs/gsm8k_direct --cot off --eval_n $EVALN > logs/gsm8k_eval_direct.log 2>&1

echo "=== GSM8K (0.5B) accuracy ===" | tee logs/table_gsm8k.md
grep -h '\[EVAL\]' logs/gsm8k_base_cot.log logs/gsm8k_base_direct.log logs/gsm8k_eval_cot.log logs/gsm8k_eval_direct.log | tee -a logs/table_gsm8k.md
echo "GSM8K_DONE $(date)" | tee -a logs/table_gsm8k.md
