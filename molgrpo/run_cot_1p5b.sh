#!/usr/bin/env bash
# 1.5B + LoRA 的 CoT 四变体对比，四卡并行（一卡一变体）。
# 每变体：SFT(用同一份种子文件，按风格取/生成推理) -> GRPO(LoRA, 加大beta, EOS修复) -> 评估(greedy/sample@1000 + bo8@300)。
# 幂等：已产出 eval 的变体自动跳过。
cd ~/grpo_mol_project
mkdir -p logs outputs

MODEL=./Qwen2.5-1.5B-Instruct
GRPO_N=4000
MAXLEN=192

run_one () {
  local STYLE="$1"; local GPU="$2"; local SEED="$3"
  export CUDA_VISIBLE_DEVICES=$GPU
  if [ -f "outputs/eval_${STYLE}_bo8.json" ]; then echo "[$(date)] skip ${STYLE}"; return; fi
  if [ ! -f "outputs/sft_${STYLE}/model.safetensors" ] && [ ! -f "outputs/sft_${STYLE}/config.json" ]; then
    echo "[$(date)][$STYLE][gpu$GPU] SFT"
    python -u molgrpo/train_sft_warmup.py --model_name_or_path $MODEL \
      --output_dir outputs/sft_${STYLE} --cot_file $SEED --cot_style ${STYLE} \
      --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
      --num_train_epochs 2 --learning_rate 1e-4 --max_length 640 > logs/sft_${STYLE}.log 2>&1
  fi
  if [ ! -f "outputs/molgrpo_${STYLE}/model.safetensors" ] && [ ! -f "outputs/molgrpo_${STYLE}/config.json" ]; then
    echo "[$(date)][$STYLE][gpu$GPU] GRPO"
    python -u molgrpo/train_grpo.py --model_name_or_path outputs/sft_${STYLE} \
      --output_dir outputs/molgrpo_${STYLE} --library_path data/library_aug.csv --num_samples ${GRPO_N} \
      --use_lora --lora_r 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
      --num_generations 4 --max_completion_length ${MAXLEN} --gradient_checkpointing \
      --temperature 1.0 --beta 0.1 > logs/grpo_${STYLE}.log 2>&1
  fi
  echo "[$(date)][$STYLE][gpu$GPU] EVAL"
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${STYLE} --num_examples 1000 \
    --batch_size 16 --max_new_tokens ${MAXLEN} --out outputs/eval_${STYLE}_greedy.json > logs/eval_${STYLE}_greedy.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${STYLE} --num_examples 1000 \
    --batch_size 16 --do_sample --max_new_tokens ${MAXLEN} --out outputs/eval_${STYLE}_sample.json > logs/eval_${STYLE}_sample.log 2>&1
  python -u molgrpo/evaluate.py --model_name_or_path outputs/molgrpo_${STYLE} --num_examples 300 \
    --batch_size 4 --best_of_n 8 --max_new_tokens ${MAXLEN} --out outputs/eval_${STYLE}_bo8.json > logs/eval_${STYLE}_bo8.log 2>&1
  echo "[$(date)][$STYLE][gpu$GPU] DONE"
}

# 5 个变体，4 张卡：4 个非 gemini 立刻并行（用完整种子文件）；gemini 接在 GPU0 的 none 之后（届时其数据已生成好）。
SEED_PLAIN=data/cot_seed_4k.jsonl
SEED_GEMINI=data/cot_gemini_4k.jsonl
( run_one none 0 $SEED_PLAIN; run_one gemini 0 $SEED_GEMINI ) &
run_one short 1 $SEED_PLAIN &
run_one structured 2 $SEED_PLAIN &
run_one constructive 3 $SEED_PLAIN &
wait

for DEC in greedy sample bo8; do
  echo "=== ${DEC} ===" | tee -a logs/table_1p5b.md
  python molgrpo/compare_results.py \
    outputs/eval_none_${DEC}.json outputs/eval_short_${DEC}.json outputs/eval_structured_${DEC}.json \
    outputs/eval_gemini_${DEC}.json outputs/eval_constructive_${DEC}.json 2>/dev/null | tee -a logs/table_1p5b.md
done
echo "ALL_DONE_1P5B" | tee -a logs/table_1p5b.md
