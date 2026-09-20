"""PDF 分块诊断：看 pdfplumber 的换行到底是"段落边界"还是"视觉行边界"。

为什么需要这个：分块函数要靠 `\\n` 判断段落。如果 pdfplumber 在每个视觉行尾
都插了 `\\n`（中文书排版常见），那"按段落切"就退化成"按行切"，块长度会失控。
这个脚本给出三个判断信号：

  1. 平均行长 —— 中文书一个视觉行约 35~45 字。平均行长落在这个区间说明
     `\\n` 是视觉行；明显更长（80+）说明一个 `\\n` 跨越了多行。
  2. 空行数 / 段首缩进行数 —— 中文书段落通常以两个全角空格（　　）缩进开头，
     或用空行分段。这两个信号是"真正的段落边界"。
  3. 空文本页比例 —— 如果大面积为空，说明是扫描版 PDF（没有文本层），
     这时候分块参数怎么调都没用，必须先做 OCR。

用法：
    python scripts/diagnose_pdf_layout.py --pdf upload_files/document_1_xxx.pdf
    python scripts/diagnose_pdf_layout.py --pdf xxx.pdf --pages 10,60,120 --sample 400
"""
import argparse
import re
import sys


def _verdict(avg_line_len, blank_lines, indent_lines, total_lines):
    """根据三个信号给一个人话判断。"""
    if total_lines <= 1:
        return "只有一段或根本没有换行 → pdfplumber 把整页当一行，需要看 layout 模式"
    reasons = []
    if 30 <= avg_line_len <= 50:
        reasons.append(f"平均行长 {avg_line_len:.0f} 字，正中中文书视觉行的区间")
    elif avg_line_len > 80:
        reasons.append(f"平均行长 {avg_line_len:.0f} 字，一个 \\n 跨越了多个视觉行")
    if blank_lines:
        reasons.append(f"有 {blank_lines} 个空行（段落分隔信号）")
    if indent_lines:
        reasons.append(f"有 {indent_lines} 行段首缩进（段落起始信号）")
    return "；".join(reasons) if reasons else "信号不明显，看下面的原文样本人工判断"


def _is_indent(line):
    return line.startswith(("\u3000", "  ", "\t"))


def diagnose_page(text, sample_chars):
    lines = text.split("\n")
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    indent_lines = sum(1 for l in lines if l.strip() and _is_indent(l))
    avg_line_len = len(text) / total_lines if total_lines else 0
    return {
        "chars": len(text),
        "lines": total_lines,
        "blank_lines": blank_lines,
        "indent_lines": indent_lines,
        "avg_line_len": avg_line_len,
        "verdict": _verdict(avg_line_len, blank_lines, indent_lines, total_lines),
        "sample": text[:sample_chars],
    }


_SENT_END = re.compile(r"(?<=[。！？!?；;])")


def sentence_stats(pdf, page_ids, max_pages=30):
    """把抽样页拼成连续文本流后按句切，统计句长分布。

    为什么看句长：分块的 target_size 要按"一句多长"来定。如果平均句长 40 字，
    target=350 就是 8~9 句，块内语义完整；如果平均句长 100 字，那 target=350
    会让每个块只装 3 句，甚至一句被硬切。
    """
    flow_parts = []
    for pid in page_ids[:max_pages]:
        text = pdf.pages[pid].extract_text() or ""
        flow_parts.append(text.replace("\n", ""))  # 拼行：中文行尾断裂不补空格
    flow = "".join(flow_parts)
    sents = [s.strip() for s in _SENT_END.split(flow)]
    sents = [s for s in sents if len(s) >= 2]
    if not sents:
        return None
    lengths = sorted(len(s) for s in sents)
    n = len(lengths)
    return {
        "flow_chars": len(flow),
        "n_sent": n,
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "max": lengths[-1],
        "mean": sum(lengths) / n,
    }


def main():
    # Windows 控制台默认 GBK，中文 + 各种符号容易炸，这里强制成 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, help="待诊断的 PDF 路径")
    parser.add_argument("--pages", default=None,
                        help="抽样页码，逗号分隔（0 基）。默认均匀抽 5 页")
    parser.add_argument("--sample", type=int, default=300, help="每页原文样本字符数")
    args = parser.parse_args()

    try:
        import pdfplumber
    except ImportError:
        sys.exit("没装 pdfplumber：请用项目环境 rag310 运行，"
                 "例如 D:\\miniconda3\\envs\\rag310\\python.exe scripts\\diagnose_pdf_layout.py ...")

    with pdfplumber.open(args.pdf) as pdf:
        n_pages = len(pdf.pages)
        if args.pages:
            page_ids = [int(x) for x in args.pages.split(",")]
        else:
            step = max(1, n_pages // 6)
            page_ids = list(range(0, n_pages, step))[1:6]  # 跳过封面/版权页

        print(f"PDF: {args.pdf}")
        print(f"总页数: {n_pages}")
        print(f"抽样页: {page_ids}（0 基）")
        print("=" * 70)

        empty_pages = 0
        for pid in page_ids:
            if pid >= n_pages:
                continue
            page = pdf.pages[pid]
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages += 1
                print(f"\n[页 {pid}] 文本为空 —— 这一页没有文本层（扫描页或纯图页）")
                # 有图但没字 → 扫描件；顺便报一下页面尺寸，方便判断要不要 OCR
                print(f"  页面尺寸: {page.width:.0f}x{page.height:.0f}pt  "
                      f"图片数: {len(page.images)}")
                continue

            d = diagnose_page(text, args.sample)
            print(f"\n[页 {pid}] 字符={d['chars']}  换行={d['lines']}  "
                  f"空行={d['blank_lines']}  段首缩进行={d['indent_lines']}  "
                  f"平均行长={d['avg_line_len']:.1f}")
            print(f"  判断: {d['verdict']}")
            print(f"  原文前 {args.sample} 字（repr，\\n 会显示成换行转义）:")
            print("  " + repr(d["sample"]))

        print("\n" + "=" * 70)
        sampled = len([p for p in page_ids if p < n_pages])
        if empty_pages == sampled:
            print("结论: 抽样页全部无文本层 → 这是扫描版 PDF，"
                  "当前 pdfplumber 链路拿不到任何内容，必须先做 OCR。")
        elif empty_pages:
            print(f"结论: {empty_pages}/{sampled} 页无文本层 → 混合型 PDF，"
                  "需要在解析里做逐页判断 + OCR 兜底。")
        else:
            print("结论: 抽样页都有文本层。接下来看上面每页的『判断』一行，"
                  "确认 \\n 是段落边界还是视觉行边界，再决定分块函数怎么写。")

        # ── 句子统计：为分块参数提供依据 ──
        st = sentence_stats(pdf, [p for p in page_ids if p < n_pages])
        if st:
            print("\n" + "=" * 70)
            print("句子统计（抽样页拼成文本流后按 。！？； 切句）")
            print(f"  总字符={st['flow_chars']}  句数={st['n_sent']}  "
                  f"平均句长={st['mean']:.1f}  中位={st['p50']}  "
                  f"P90={st['p90']}  最长={st['max']}")
            suggested = max(200, min(600, int(st["p90"] * 5) // 50 * 50))
            print(f"  建议 target_size ≈ {suggested} 字"
                  f"（约 {suggested / max(1, st['mean']):.0f} 句/块，"
                  f"按 P90 句长 {st['p90']} 的 5 倍取整）")


if __name__ == "__main__":
    main()
