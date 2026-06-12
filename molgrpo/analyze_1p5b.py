"""汇总 1.5B 八变体结果：对比表 + pass@k 曲线 + 约束数 vs 成功率。读取 results/1p5b/。"""
import json, os

RES = os.path.join(os.path.dirname(__file__), "results", "1p5b")
ORDER = ["none", "short", "structured", "constructive", "gemini", "budget", "selfies", "selfiesbudget"]


def load(v, dec):
    p = os.path.join(RES, f"eval_{v}_{dec}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def passk_from_bo8(payload, ks=(1, 2, 4, 8)):
    n = payload["summary"].get("best_of_n", 8)
    rows = payload["examples"]
    groups = [rows[i:i + n] for i in range(0, len(rows), n)]
    groups = [g for g in groups if len(g) == n]
    out = {}
    for k in ks:
        out[k] = sum(1 for g in groups if any(r["all_ok"] for r in g[:k])) / max(1, len(groups))
    return out


def constraint_breakdown(payload):
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in payload["examples"]:
        tot = r.get("n_prop", 0) + r.get("n_struct", 0)
        agg[tot][0] += int(r["all_ok"]); agg[tot][1] += 1
    return {k: agg[k][0] / agg[k][1] for k in sorted(agg) if agg[k][1] >= 5}


lines = []
def P(s=""):
    lines.append(s); print(s)

P("## 1.5B 八变体对比（贪心 + best-of-8）\n")
P("| 变体 | 合法率 | 全满足(贪心) | 唯一率(贪心) | pass@8 | 新颖率 |")
P("|------|------|------|------|------|------|")
for v in ORDER:
    g = load(v, "greedy"); b = load(v, "bo8")
    if not g:
        continue
    gs = g["summary"]; bs = b["summary"] if b else None
    passn = f"{bs['pass_at_n']:.3f}" if bs else "-"
    novel = bs["novelty_rate"] if bs else gs["novelty_rate"]
    P(f"| {v} | {gs['valid_rate']:.3f} | {gs['all_ok_rate']:.3f} | {gs['unique_rate']:.3f} | {passn} | {novel:.3f} |")

P("\n## pass@k 曲线（从 best-of-8 重组，k=1/2/4/8）\n")
P("| 变体 | pass@1 | pass@2 | pass@4 | pass@8 |")
P("|------|------|------|------|------|")
for v in ORDER:
    b = load(v, "bo8")
    if not b:
        continue
    pk = passk_from_bo8(b)
    P(f"| {v} | {pk[1]:.3f} | {pk[2]:.3f} | {pk[4]:.3f} | {pk[8]:.3f} |")

P("\n## 约束数 vs 全满足率（贪心；约束越多越难）\n")
P("| 变体 | 1个 | 2个 | 3个 | 4个 | 5个 |")
P("|------|------|------|------|------|------|")
for v in ORDER:
    g = load(v, "greedy")
    if not g:
        continue
    cb = constraint_breakdown(g)
    cells = " | ".join(f"{cb.get(k, float('nan')):.2f}" if k in cb else "-" for k in (1, 2, 3, 4, 5))
    P(f"| {v} | {cells} |")

open(os.path.join(RES, "REPORT_1p5b.md"), "w").write("\n".join(lines) + "\n")
print("\nwrote", os.path.join(RES, "REPORT_1p5b.md"))
