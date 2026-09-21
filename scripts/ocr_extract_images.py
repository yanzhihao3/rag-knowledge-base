"""抠出 PDF 里的图片跑 OCR，验证"能不能还原代码和表格"。

背景：这本《大模型RAG实战》250 张图占 20.9% 版面，代码示例、表格、架构图
全是图片，纯文本解析器（pdfplumber、pymupdf4llm）一个字都拿不到。

这个脚本是**验证工具，不改主流程**：
  1. 用 PyMuPDF 抠出页面里的图片（原生分辨率，不重新渲染）
  2. 用 RapidOCR（本地 CPU、免费、中文友好）识别文字 + 坐标
  3. 按坐标把文字聚成行、分列，拼成 Markdown 表格
  4. 打印原图和结果，人眼判断质量

用法：
    python scripts/ocr_extract_images.py --pdf "xxx.pdf" --pages 108,109,200
    python scripts/ocr_extract_images.py --pdf "xxx.pdf" --pick 10     # 自动挑 10 张

前提：pip install rapidocr-onnxruntime
"""
import argparse
import io
import os
import statistics
import sys
import time


_MODEL_CACHE = {}


def _load_ocr():
    """RapidOCR 有两个包名，新版叫 rapidocr，老版叫 rapidocr_onnxruntime，都兼容。"""
    try:
        from rapidocr import RapidOCR  # 新版
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR  # 老版（更稳）
        except ImportError:
            sys.exit("没装 OCR 引擎，先执行：\n"
                     r"    D:\miniconda3\envs\rag310\python.exe -m pip install rapidocr-onnxruntime")
    if "engine" not in _MODEL_CACHE:
        _MODEL_CACHE["engine"] = RapidOCR()
    return _MODEL_CACHE["engine"]


def _classify(w, h):
    r = w / h if h else 0
    if r > 4:
        return "宽扁(代码行)"
    if r > 1.5:
        return "中等(代码块/表)"
    if r > 0.8:
        return "方形(架构图)"
    return "竖长(整页)"


def _iter_images(doc, pages):
    """产出 (页码, 图片序号, 宽, 高, 形状, 图片bytes)。"""
    for pno in pages:
        page = doc[pno]
        idx = 0
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 1:
                continue
            w, h = block.get("width", 0), block.get("height", 0)
            if w < 60 or h < 20:
                continue
            idx += 1
            yield pno, idx, w, h, _classify(w, h), block["image"]


def _pick_images(doc, n, n_pages):
    """均匀抽样，尽量覆盖不同形状。"""
    all_imgs = list(_iter_images(doc, range(n_pages)))
    if not all_imgs:
        return []
    step = max(1, len(all_imgs) // n)
    picked = all_imgs[::step][:n]
    return picked


def _run_ocr(engine, img_bytes, tmp_path):
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    out = engine(tmp_path)
    # 两种包的返回结构略有差异，统一成 result 列表
    result = out[0] if isinstance(out, tuple) else out
    items = []
    for row in (result or []):
        box, text, score = row[0], row[1], row[2]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": str(text),
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
            "h": max(ys) - min(ys),
            "score": float(score),
        })
    return items


def _group_rows(items, tol_ratio=0.6):
    if not items:
        return []
    med_h = statistics.median([it["h"] for it in items]) or 10
    items = sorted(items, key=lambda it: it["cy"])
    rows, cur = [], [items[0]]
    for it in items[1:]:
        row_cy = sum(x["cy"] for x in cur) / len(cur)
        if abs(it["cy"] - row_cy) <= tol_ratio * med_h:
            cur.append(it)
        else:
            rows.append(sorted(cur, key=lambda x: x["cx"]))
            cur = [it]
    rows.append(sorted(cur, key=lambda x: x["cx"]))
    return rows


def _render(rows):
    """多列行占比够高就当表格，否则当普通文字。"""
    if not rows:
        return "（没识别出任何文字）"
    multi = sum(1 for r in rows if len(r) >= 2)
    as_table = multi / len(rows) >= 0.5
    lines = []
    if as_table:
        ncol = max(len(r) for r in rows)
        for i, r in enumerate(rows):
            cells = [c["text"] for c in r] + [""] * (ncol - len(r))
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("|" + "---|" * ncol)
    else:
        for r in rows:
            lines.append(" ".join(c["text"] for c in r))
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--pages", default=None, help="0 基页码，逗号分隔")
    parser.add_argument("--pick", type=int, default=0, help="自动均匀挑 N 张图")
    parser.add_argument("--from-json", default=None,
                        help="只处理 find_table_images.py 产出的 JSON 里列出的候选图")
    args = parser.parse_args()

    import fitz  # PyMuPDF

    doc = fitz.open(args.pdf)
    n_pages = len(doc)
    print(f"PDF: {args.pdf}   共 {n_pages} 页")

    if args.pages:
        targets = list(_iter_images(doc, [int(x) for x in args.pages.split(",")]))
    elif args.from_json:
        import json
        with open(args.from_json, "r", encoding="utf-8") as f:
            wanted = {tuple(x) for x in json.load(f)["unique_images"]}
        targets = [t for t in _iter_images(doc, range(n_pages))
                   if (t[0], t[1]) in wanted]
    else:
        n = args.pick or 10
        targets = _pick_images(doc, n, n_pages)

    print(f"本次处理 {len(targets)} 张图")
    engine = _load_ocr()
    tmp = os.path.join(os.environ.get("TEMP", "."), "_ocr_tmp.png")

    total_t = 0.0
    for pno, idx, w, h, kind, img in targets:
        t0 = time.monotonic()
        items = _run_ocr(engine, img, tmp)
        dt = time.monotonic() - t0
        total_t += dt
        rows = _group_rows(items)
        print("\n" + "=" * 76)
        print(f"第 {pno} 页 图{idx}  {w}x{h}px  {kind}  识别 {len(items)} 个文本块  "
              f"耗时 {dt:.1f}s")
        print("-" * 76)
        print(_render(rows))

    print("\n" + "=" * 76)
    print(f"平均每张 {total_t / max(1, len(targets)):.1f}s，"
          f"全部 {len(targets)} 张共 {total_t:.0f}s")
    print("怎么看：多列行占比高 → 被识别成表格；只有单列 → 当普通文字输出")


if __name__ == "__main__":
    main()
