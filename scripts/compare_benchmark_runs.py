"""对比两次 benchmark 运行的日志，逐题 diff（召回率 A/B 用）。

为什么按"位置"对齐而不是按题目文本：pytest 的 -s 输出经过 PowerShell 管道时，
子进程的 UTF-8 字节会被按控制台代码页（中文 Windows 是 GBK）解码，中文字符会变成
乱码，但 `[OK]/[FAIL]/rank=N` 这些 ASCII 结构是完好的。而 _run_one_method 是按
BENCHMARK_CASES 的顺序遍历的，所以第 i 行结构就对应第 i 道题。

用法：
    python scripts/compare_benchmark_runs.py <基线文件> <新文件>
    python scripts/compare_benchmark_runs.py \
        test/baselines/20260912_before_chunking.txt \
        test/baselines/20260912_after_semantic_chunking.txt
"""
import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from test.benchmark_cases_rag_book import BENCHMARK_CASES  # noqa: E402

METHODS = ["BM25", "Vector", "Fusion(RRF)"]
# 一行结构长这样（中文部分是乱码，正好不参与匹配）：
#   [medium] <junk>[OK] <junk>[FAIL] rank=None  <junk>
# 注意结构框里是 [_OK_] 而不是 [OK]，日志里是方括号包住 OK
_ROW_RE = re.compile(r"\[(OK|FAIL)\][^\[]*\[(OK|FAIL)\]\s*rank=(None|\d+)")


def parse_rows(path):
    """返回 [(关键词命中, 答案块命中, rank)]，按日志出现顺序。"""
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    rows = []
    for kw, strict, rank in _ROW_RE.findall(text):
        rows.append((kw == "OK", strict == "OK", None if rank == "None" else int(rank)))
    return rows


def split_methods(rows, path):
    n = len(rows)
    if n % 3 != 0:
        sys.exit(f"{path}: 解析到 {n} 行，不是 3 的倍数，日志可能被截断")
    per = n // 3
    if per != len(BENCHMARK_CASES):
        print(f"警告: {path} 每个方法解析到 {per} 行，"
              f"但用例数是 {len(BENCHMARK_CASES)}，可能有个别行没匹配上")
    return {m: rows[i * per:(i + 1) * per] for i, m in enumerate(METHODS)}


def summarize(rows):
    n = len(rows)
    kw = sum(1 for r in rows if r[0]) / n
    strict = sum(1 for r in rows if r[1]) / n
    mrr = sum(1.0 / r[2] for r in rows if r[2]) / n
    return kw, strict, mrr


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("new")
    args = parser.parse_args()

    old = split_methods(parse_rows(args.baseline), args.baseline)
    new = split_methods(parse_rows(args.new), args.new)

    print(f"基线: {args.baseline}")
    print(f"新结果: {args.new}")
    print("=" * 72)
    print(f"{'Method':<12} {'关键词(旧→新)':>22} {'答案块(旧→新)':>22} {'MRR@5(旧→新)':>20}")
    print("-" * 72)
    for m in METHODS:
        a, b = summarize(old[m]), summarize(new[m])
        print(f"{m:<12} {a[0]*100:>8.1f}%→{b[0]*100:<6.1f}%"
              f"{a[1]*100:>8.1f}%→{b[1]*100:<6.1f}%"
              f"{a[2]:>10.3f}→{b[2]:<6.3f}")

    # ── Fusion 逐题 diff ──
    print("\n" + "=" * 72)
    print("融合(RRF) 逐题 diff：答案块命中 + 排名变化")
    print("=" * 72)

    improved, regressed, fixed, broke = [], [], [], []
    for case, o, n in zip(BENCHMARK_CASES, old["Fusion(RRF)"], new["Fusion(RRF)"]):
        o_rank, n_rank = o[2], n[2]
        if o_rank == n_rank:
            continue
        item = (case["query"], case["difficulty"], o_rank, n_rank)
        if o_rank is None and n_rank is not None:
            fixed.append(item)
        elif o_rank is not None and n_rank is None:
            broke.append(item)
        elif o_rank is not None and n_rank is not None:
            (improved if n_rank < o_rank else regressed).append(item)

    def show(title, items):
        print(f"\n{title}（{len(items)} 题）")
        for q, d, o, n in items:
            o_s = f"rank={o}" if o else "未命中"
            n_s = f"rank={n}" if n else "未命中"
            print(f"  [{d:<6}] {o_s:<8} → {n_s:<8}  {q}")

    show("[+] 从「未命中」变成「命中」", fixed)
    show("[-] 从「命中」变成「未命中」", broke)
    show("↑ 命中且排名上升", improved)
    show("↓ 命中但排名下降", regressed)

    same = sum(1 for o, n in zip(old["Fusion(RRF)"], new["Fusion(RRF)"]) if o[2] == n[2])
    print(f"\n排名没变的题: {same}/{len(BENCHMARK_CASES)}")


if __name__ == "__main__":
    main()
