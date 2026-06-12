#!/usr/bin/env bash
# 难度反转受控实验（v2 库不变，唯一变量 = 约束"精度/解密度"）。
# 做法：保持 round3/v2 那套出题(1-3性质+0-3结构)与奖励完全不动，
#       只用 MOLGRPO_TIGHTNESS 给区间型性质的窗口半宽乘 τ（τ<1 收窄=更难）。
#       τ 乘在采样之后、不耗随机数 → 各档 prompt 完全配对(同分子/同性质，只窗口宽窄不同)。
# 难度四档(τ -> 实测3性质解密度)：
#   t1 τ=1.2 -> ~0.25  (易锚点)
#   t2 τ=0.7 -> ~0.067 (≈v3)
#   t3 τ=0.5 -> ~0.030 (比v3难~2x)
#   t4 τ=0.35-> ~0.0096(比v3难~5.6x, 未触底)
# 每档跑 none vs budget；产出 (budget-none) 增益 vs 难度 曲线。
# 训练/评测参数全部对齐 round3：NS=4000, gen=8, maxlen=192, beta=0.1, lr=5e-6(默认), 评测1000/300×8。
# SFT 对齐 v2：n_samples=6000, max_length=512。结构约束默认 0-3(对齐 v2/v3)。
# 4 卡并行；幂等(已出 bo8 跳过)。训完即删模型只留 json(省盘)。
cd ~/grpo_mol_project
mkdir -p logs outputs
MODEL=./Qwen2.5-1.5B-Instruct
LIB=data/library_v2.csv
MAXLEN=192
GEN=8
NS=4000
COMMON="--use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_completion_length ${MAXLEN} --gradient_checkpointing --temperature 1.0 --beta 0.1 --num_generations ${GEN}"
declare -A TAU=( [t1]=1.2 [t2]=0.7 [t3]=0.5 [t4]=0.35 )
SFTN=outputs/sft_none_diff
SFTB=outputs/sft_budget_diff

# SFT base：不设 TIGHTNESS（自然窗 τ=1，等同 v2 的 SFT），对齐 v2 配方 6000/512。
sft () { # style outdir gpu
  local STYLE="$1" OUT="$2" GPU="$3"
  [ -f "${OUT}/config.json" ] && { echo "skip sft ${OUT}"; return; }
  CUDA_VISIBLE_DEVICES=$GPU python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL \
    --output_dir ${OUT} --library_path $LIB --cot_style ${STYLE} --n_samples 6000 \
    --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 1 --learning_rate 1e-4 --max_length 512 > logs/$(basename ${OUT}).log 2>&1
}

# 一个 (风格×难度) 单元：GRPO + greedy + bo8，全程绑定本档 τ；评测完删模型省盘。
cell () { # name style sftbase tier gpu
  local NAME="$1" STYLE="$2" SFT="$3" TIER="$4" GPU="$5"
  local T=${TAU[$TIER]}
  [ -f "outputs/eval_${NAME}_bo8.json" ] && { echo "skip ${NAME}"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU MOLGRPO_TIGHTNESS=$T
  echo "[$(date)][${NAME}] GRPO tau=${T}"
  python -u molgrpo/train_grpo.py --model_name_or_path ${SFT} --output_dir outputs/${NAME} \
    --library_path $LIB --num_samples ${NS} ${COMMON} > logs/grpo_${NAME}.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} --num_examples 1000 --batch_size 16 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_greedy.json > logs/eval_${NAME}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/${NAME} --num_examples 300 --batch_size 4 --best_of_n 8 \
    --max_new_tokens ${MAXLEN} --library_path $LIB --out outputs/eval_${NAME}_bo8.json > logs/eval_${NAME}_bo8.log 2>&1
  [ -f "outputs/eval_${NAME}_bo8.json" ] && rm -rf outputs/${NAME}   # 评测出了就删模型省盘
  echo "[$(date)][${NAME}] DONE"
}

# 两套 SFT base 并行
sft none ${SFTN} 0 &
sft budget ${SFTB} 1 &
wait

# Wave 1: none × 四档
cell dn_t1 none ${SFTN} t1 0 &
cell dn_t2 none ${SFTN} t2 1 &
cell dn_t3 none ${SFTN} t3 2 &
cell dn_t4 none ${SFTN} t4 3 &
wait
# Wave 2: budget × 四档
cell db_t1 budget ${SFTB} t1 0 &
cell db_t2 budget ${SFTB} t2 1 &
cell db_t3 budget ${SFTB} t3 2 &
cell db_t4 budget ${SFTB} t4 3 &
wait

echo "=== DIFFICULTY greedy (none vs budget, 四档) ===" | tee logs/table_difficulty.md
python analysis/compare_results.py outputs/eval_dn_t1_greedy.json outputs/eval_db_t1_greedy.json \
  outputs/eval_dn_t2_greedy.json outputs/eval_db_t2_greedy.json \
  outputs/eval_dn_t3_greedy.json outputs/eval_db_t3_greedy.json \
  outputs/eval_dn_t4_greedy.json outputs/eval_db_t4_greedy.json 2>/dev/null | tee -a logs/table_difficulty.md
echo "--- best-of-8 ---" | tee -a logs/table_difficulty.md
python analysis/compare_results.py outputs/eval_dn_t1_bo8.json outputs/eval_db_t1_bo8.json \
  outputs/eval_dn_t2_bo8.json outputs/eval_db_t2_bo8.json \
  outputs/eval_dn_t3_bo8.json outputs/eval_db_t3_bo8.json \
  outputs/eval_dn_t4_bo8.json outputs/eval_db_t4_bo8.json 2>/dev/null | tee -a logs/table_difficulty.md
echo "DIFFICULTY_DONE $(date)" | tee -a logs/table_difficulty.md
