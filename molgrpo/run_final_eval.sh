#!/usr/bin/env bash
# 等四变体训练完成后，对每个最终模型做大规模评估：
#   greedy@1000 / sample@1000 / best-of-8@300（含 novelty 与 pass@N）
# 然后输出对比表。会先等待 run_cot_all.sh 结束（轮询 ALL_DONE）。
cd ~/grpo_mol_project
mkdir -p logs outputs

echo "[$(date)] waiting for training (ALL_DONE in logs/cot_all.log) ..."
while ! grep -q "ALL_DONE" logs/cot_all.log 2>/dev/null; do
  sleep 60
done
echo "[$(date)] training done, start final eval"

for STYLE in none short structured gemini; do
  M="outputs/molgrpo_${STYLE}"
  [ -f "${M}/model.safetensors" ] || { echo "skip ${STYLE}: no model"; continue; }
  echo "[$(date)] final eval ${STYLE} (greedy@1000)"
  python -u molgrpo/evaluate.py --model_name_or_path ${M} --num_examples 1000 --batch_size 8 \
    --max_new_tokens 160 --out outputs/final_${STYLE}_greedy.json > logs/final_${STYLE}_greedy.log 2>&1
  echo "[$(date)] final eval ${STYLE} (sample@1000)"
  python -u molgrpo/evaluate.py --model_name_or_path ${M} --num_examples 1000 --batch_size 8 --do_sample \
    --max_new_tokens 160 --out outputs/final_${STYLE}_sample.json > logs/final_${STYLE}_sample.log 2>&1
  echo "[$(date)] final eval ${STYLE} (best-of-8@300)"
  python -u molgrpo/evaluate.py --model_name_or_path ${M} --num_examples 300 --batch_size 2 --best_of_n 8 \
    --max_new_tokens 160 --out outputs/final_${STYLE}_bo8.json > logs/final_${STYLE}_bo8.log 2>&1
done

echo "[$(date)] === GREEDY@1000 ===" | tee logs/final_table.md
python -u molgrpo/compare_results.py outputs/final_none_greedy.json outputs/final_short_greedy.json outputs/final_structured_greedy.json outputs/final_gemini_greedy.json 2>/dev/null | tee -a logs/final_table.md
echo "[$(date)] === SAMPLE@1000 ===" | tee -a logs/final_table.md
python -u molgrpo/compare_results.py outputs/final_none_sample.json outputs/final_short_sample.json outputs/final_structured_sample.json outputs/final_gemini_sample.json 2>/dev/null | tee -a logs/final_table.md
echo "[$(date)] === BEST-OF-8@300 ===" | tee -a logs/final_table.md
python -u molgrpo/compare_results.py outputs/final_none_bo8.json outputs/final_short_bo8.json outputs/final_structured_bo8.json outputs/final_gemini_bo8.json 2>/dev/null | tee -a logs/final_table.md
echo "[$(date)] FINAL_EVAL_DONE" | tee -a logs/final_table.md
