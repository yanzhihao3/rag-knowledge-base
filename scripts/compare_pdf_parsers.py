"""对比 pdfplumber 与 pymupdf4llm 在同一份 PDF 上的解析结果。

用途：决定"要不要换 PDF 解析器"。看三件事：
  1. 标题有没有还原成结构（Markdown 的 # 标题行）
  2. 段落是不是用空行分开（而不是每个视觉行一个 \\n）
  3. 词还会不会被换行切断（`Co-\\noccurrence`、`一定数\\n量`）

用法：
    python scripts/compare_pdf_parsers.py --pdf "upload_files\\xxx.pdf" --pages 40,120,200
    # 不指定 --pages 就均匀抽 3 页

前提：pip install pymupdf4llm
"""
import argparse
import re
import sys


def _stats(name, text):
    lines = text.split("\n")
    nonblank = [l for l in lines if l.strip()]
    blank = len(lines) - len(nonblank)
    avg = (sum(len(l) for l in nonblank) / len(nonblank)) if nonblank else 0
    headings = sum(1 for l in lines if re.match(r"^#{1,6}\s", l))
    # "疑似被切断的词"：行尾是字母/汉字（不是标点）且下一行紧接着继续
    broken = 0
    for a, b in zip(lines, lines[1:]):
        if a.strip() and b.strip() and not re.search(r"[。！？.!?；;：:）)\]]$", a.rstrip()):
            if re.match(r"^[a-z\u4e00-\u9fff]", b.strip()):
                broken += 1
    return {
        "name": name,
        "chars": len(text),
        "lines": len(nonblank),
        "blank_lines": blank,
        "avg_line": avg,
        "headings": headings,
        "broken": broken,
    }


def _show(s, sample):
    print(f"    {s['name']:<12} 字符={s['chars']:<7} 非空行={s['lines']:<5} "
          f"空行={s['blank_lines']:<4} 平均行长={s['avg_line']:.1f}")
    print(f"    {'':<12} Markdown标题行={s['headings']:<4} "
          f"疑似词被切断={s['broken']}")
    print(f"{sample}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--pages", default=None, help="0 基页码，逗号分隔")
    parser.add_argument("--sample", type=int, default=420, help="每页样本字符数")
    args = parser.parse_args()

    import pdfplumber
    try:
        import pymupdf4llm
    except ImportError:
        sys.exit("没装 pymupdf4llm，先执行：\n"
                 r"    D:\miniconda3\envs\rag310\python.exe -m pip install pymupdf4llm")

    with pdfplumber.open(args.pdf) as pdf:
        n_pages = len(pdf.pages)
        if args.pages:
            page_ids = [int(x) for x in args.pages.split(",")]
        else:
            step = max(1, n_pages // 4)
            page_ids = [p for p in range(0, n_pages, step) if p][:3]

        print(f"PDF: {args.pdf}")
        print(f"总页数: {n_pages}   对比页: {page_ids}（0 基）")
        print("=" * 78)

        totals = {"pdfplumber": 0, "pymupdf4llm": 0}
        for pid in page_ids:
            if pid >= n_pages:
                continue
            old = pdf.pages[pid].extract_text() or ""
            new = pymupdf4llm.to_markdown(args.pdf, pages=[pid])

            print(f"\n########## 第 {pid} 页 ##########")
            _show(_stats("pdfplumber", old),
                  "    样本: " + repr(old[:args.sample]))
            print()
            _show(_stats("pymupdf4llm", new),
                  "    样本: " + repr(new[:args.sample]))
            totals["pdfplumber"] += _stats("", old)["headings"]
            totals["pymupdf4llm"] += _stats("", new)["headings"]

        print("\n" + "=" * 78)
        print("怎么判读：")
        print("  · Markdown标题行 > 0  → 它把标题还原成结构了（你现在得靠正则猜）")
        print("  · 空行数明显 > 0      → 段落边界是真的（你的书现在是 0）")
        print("  · 疑似词被切断 ≈ 0    → 不用再写 join_visual_lines 拼行")
        print("  · 三者都满足 → 值得换解析器；否则留在 pdfplumber 逐条修下游")
        print(f"\n抽样页 Markdown 标题行合计: pdfplumber={totals['pdfplumber']}  "
              f"pymupdf4llm={totals['pymupdf4llm']}")


if __name__ == "__main__":
    main()
