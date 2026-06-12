"""汇总各 CoT 变体的评估 JSON，输出对比表（markdown）。

用法：
    python molgrpo/compare_results.py outputs/eval_none_greedy.json outputs/eval_short_greedy.json ...
若不传参数，默认扫描 outputs/eval_*_{greedy,sample}.json。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

COLS = [
    ("valid_rate", "合法率"),
    ("avg_prop_sat_ratio", "性质满足"),
    ("avg_struct_hit_ratio", "结构命中"),
    ("all_ok_rate", "全满足"),
    ("pass_at_n", "pass@N"),
    ("unique_rate", "唯一率"),
    ("novelty_rate", "新颖率"),
    ("top1_share", "top1占比"),
]


def label_of(path: str) -> str:
    name = Path(path).stem.replace("eval_", "")
    return name


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("outputs/eval_*_greedy.json")) + sorted(glob.glob("outputs/eval_*_sample.json"))
    rows = []
    for p in paths:
        try:
            s = json.loads(Path(p).read_text())["summary"]
        except Exception as e:  # noqa: BLE001
            print(f"skip {p}: {e}")
            continue
        rows.append((label_of(p), s))

    header = "| 变体 | " + " | ".join(c[1] for c in COLS) + " |"
    sep = "|------|" + "|".join(["------"] * len(COLS)) + "|"
    print(header)
    print(sep)
    for label, s in rows:
        cells = []
        for key, _ in COLS:
            v = s.get(key)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
