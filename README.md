# Constrained Molecule Generation with GRPO

Fine-tuning a small LLM (**Qwen2.5-1.5B-Instruct**) to design valid molecules that
satisfy a set of **property + structural constraints**, trained end-to-end with
**GRPO** (Group Relative Policy Optimization) and **RDKit-computed verifiable rewards** —
no human labels, no reward model. The project then runs a controlled study of **how
chain-of-thought (CoT) style affects this task**, with cross-domain (math) and
difficulty controls.

> Personal research project. The training/eval framework is adapted from a GRPO
> math-reasoning baseline; the task, reward design, data pipeline, and the CoT /
> ablation experiments are my own.

## TL;DR findings

1. **Training works, and SFT + GRPO are synergistic (super-additive).**
   An untrained base model almost can't do the task (greedy all-ok 1.5%).
   SFT-only (5.2%) and GRPO-only (7.7%) are each weak; **SFT+GRPO together reach 30.2%** —
   far more than the sum. SFT supplies a format/legal-SMILES prior; GRPO then optimizes
   constraint satisfaction on top of it.
2. **On this task, CoT is at best neutral, and heavy CoT hurts.**
   `none / short / budget` tie at the top (within single-seed noise); making the reasoning
   longer / more natural-language (`constructive → structured → gemini`) monotonically
   degrades performance, with the most verbose style also dropping *validity* (0.98 → 0.90).
3. **Same CoT is clearly useful on math (GSM8K): 29.2% vs 4.0% direct** —
   a 7× gain. So **CoT's value is strongly task-dependent**: math needs chainable
   multi-step derivation; constrained generation behaves more like one-shot
   constrained sampling where long reasoning becomes a burden.
4. **Difficulty control (honest null):** varying only constraint tightness within a
   single library shows **no robust difficulty-dependent CoT effect** (differences are
   within seed noise).

## Key results

**Training vs. untrained (1.5B, library_v2)**

| model | greedy all-ok | pass@8 | valid |
|---|---|---|---|
| base-1.5B (untrained) | 0.015 | 0.083 | 0.17 |
| SFT + GRPO (none) | **0.302** | **0.570** | **0.976** |

**CoT styles (same library, same recipe, shared SFT seeds — only the reasoning text differs)**

| reasoning style | greedy all-ok | pass@8 | valid |
|---|---|---|---|
| budget (numeric budgeting) | **0.317** | **0.577** | 0.986 |
| short | 0.309 | 0.527 | 0.987 |
| none | 0.302 | 0.570 | 0.976 |
| constructive | 0.286 | 0.503 | 0.983 |
| structured | 0.263 | 0.487 | 0.978 |
| gemini (verbose NL) | 0.212 | 0.463 | 0.900 |

**2×2 ablation (all v2, same eval protocol, same 1000-step GRPO budget) — `all_ok / pass@8 / valid`**

|  | no GRPO | + GRPO |
|---|---|---|
| **no SFT** | base: 0.015 / 0.083 / 0.171 | GRPO-only: 0.077 / 0.217 / 0.761 |
| **SFT (none)** | SFT-only: 0.052 / 0.263 / 0.344 | **SFT+GRPO: 0.302 / 0.570 / 0.976** |

Full numbers, protocol, and the difficulty study: see [`report/findings_zh.md`](report/findings_zh.md) (Chinese).

## Method

- **Task.** Given property constraints (1–3 of MolWt / logP / QED / TPSA / HBD / HBA /
  rotatable bonds / ring count) plus 0–3 structural constraints (functional-group SMARTS),
  generate **one** valid SMILES satisfying all of them.
- **Verifiable reward (RLVR).** Whether constraints are met is decided programmatically by
  RDKit — no annotation, and the answer is not unique (any satisfying molecule is correct).
  Reward terms: format, validity, continuous property score (center-weighted),
  structural match, drug-likeness, an all-satisfied bonus, in-batch diversity, conciseness.
- **GRPO** (TRL): sample several completions per prompt, update with group-relative advantage.
- **Pipeline.** `base →[SFT (LoRA r=16)]→ merge →[GRPO (LoRA)]→ merge → eval`. SFT teaches the
  `<reasoning>/<answer>` format and a given CoT style; GRPO does the heavy lifting on
  validity and constraint satisfaction.
- **Constraints are derived from real molecules** in the library (sample a molecule, read off
  its properties/groups, randomly keep 1–3 property + 0–3 structural conditions) — so every
  prompt has at least one known feasible solution, and no single "universal molecule" can
  satisfy all prompts, which prevents mode collapse.

Output format:

```text
<reasoning>
...
</reasoning>
<answer>
SMILES
</answer>
```

## Repo layout

```
molgrpo/
  chem_utils.py          # properties, SMARTS, derive_constraints, scoring, prompt rendering
  dataset.py             # build training prompts by reverse-deriving constraints
  rewards.py             # all reward functions (RDKit-based)
  train_sft_warmup.py    # SFT warm-start (constraint -> molecule + CoT style)
  train_grpo.py          # GRPO training entrypoint
  evaluate.py / infer.py # batch eval (valid / property / structural / all-ok / diversity) and demo
  cot.py                 # CoT-style generators (none/short/structured/constructive/budget)
  build_library.py       # build a balanced real library from MOSES
  augment_library.py     # BRICS recombination to extend the library
  gsm8k_grpo.py          # cross-domain control: CoT vs direct on GSM8K
  run_*.sh               # experiment orchestration (CoT comparison, difficulty, etc.)
report/
  findings_zh.md         # detailed write-up (Chinese)
requirements.txt
```

> Note: the Gemini-distilled CoT generator depends on internal tooling and is omitted from
> this public repo; the `gemini` row above was produced with it.

## Quickstart

```bash
pip install -r requirements.txt   # transformers==4.51.3, trl==0.17.0, datasets, rdkit, torch

# 1) build a real library from MOSES (download dataset_v1.csv first)
python molgrpo/build_library.py --src data/moses.csv --out data/library.csv \
  --target 10000 --reinforce 1000 --bucket_slack 1500 --max_scan 400000
# 2) BRICS augmentation
python molgrpo/augment_library.py --src data/library.csv --out data/library_aug.csv --target 20000
# 3) SFT warm-start
python molgrpo/train_sft_warmup.py --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/sft --library_path data/library_aug.csv --n_samples 8000
# 4) GRPO
python molgrpo/train_grpo.py --model_name_or_path outputs/sft \
  --output_dir outputs/grpo --library_path data/library_aug.csv --num_samples 4000
# 5) evaluate (greedy + best-of-N)
python molgrpo/evaluate.py --model_name_or_path outputs/grpo \
  --num_examples 1000 --out outputs/eval_greedy.json
```

Pin `transformers==4.51.3` and `trl==0.17.0`: newer TRL pulls transformers 5.x, which breaks
on older torch/CUDA stacks.

## Notes & limitations

- Main comparisons are single-seed; top `none/short/budget` and the difficulty deltas are
  within noise — turning them into strong claims needs 2–3 seeds with error bars.
- Capped at 1.5B; whether heavy-CoT harm changes with model scale is left as future work.
- Trained/evaluated on Tesla T4 / V100 (fp16; under fp16, load weights in float32 to avoid
  the GradScaler "unscale FP16 gradients" error).
