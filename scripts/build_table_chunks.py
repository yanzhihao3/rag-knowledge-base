"""阶段 1：把表格图片变成"可检索的块文本"。

流程：候选图 → OCR → 判定是不是表格 → 取表题和引用句 → 拼成块文本

默认只**预览**（打印块文本，不写 ES）；加 --apply 才写入。

为什么块文本要带表题和引用句：实测 ES/IK 分词显示，
  「表4-1 LangChain提供的特殊文本切块方法」和用户问句「LangChain支持哪些特殊文本切块方法」
  分词高度重合 → BM25 能命中；
  而表格内容本身（| 文本格式 | 类名 | ...）和问句几乎不重合。
孤立的表格内容检索不到，必须带语义锚点。

用法：
    python scripts/build_table_chunks.py --pdf xxx.pdf --tables test/baselines/table_images.json
    python scripts/build_table_chunks.py ... --apply        # 真正写入 ES
"""
import argparse
import json
import os
import re
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

_TABLE_NO = re.compile(r"表\s*(\d+)\s*[-–—]\s*(\d+)")

# 引用句常见的开头（这些不是表题，是正文在"引用"这张表）
_BAD_CAPTION = re.compile(r"^(所示|展示|给出|列出|如下|见|中|里|为|是|说明|描述)")


def _ocr_rows(engine, img_bytes, tmp_path):
    """OCR 一张图，返回按行分组的结果 [[{text,cx,cy,h}, ...], ...]。"""
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    out = engine(tmp_path)
    result = out[0] if isinstance(out, tuple) else out
    items = []
    for row in (result or []):
        box, text = row[0], str(row[1])
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": text, "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
                      "h": max(ys) - min(ys)})
    if not items:
        return []
    med = statistics.median([i["h"] for i in items]) or 10
    items.sort(key=lambda x: x["cy"])
    rows, cur = [], [items[0]]
    for it in items[1:]:
        row_cy = sum(x["cy"] for x in cur) / len(cur)
        if abs(it["cy"] - row_cy) <= 0.6 * med:
            cur.append(it)
        else:
            rows.append(sorted(cur, key=lambda x: x["cx"]))
            cur = [it]
    rows.append(sorted(cur, key=lambda x: x["cx"]))
    return rows


def _rows_to_markdown(rows):
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        cells = [c["text"] for c in r] + [""] * (ncol - len(r))
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("|" + "---|" * ncol)
    return "\n".join(out)


def _is_table(rows):
    """判定是不是真表格。

    只按"多列行占比"会误判：折线图 OCR 出来是 12 列的网格，公式图是 2 行 2 列，
    代码截图也会被切成多列。所以综合四条判断：
      · 不含代码特征（代码被切成列之后形状和表格很像）
      · 含中文（公式图/符号图几乎没有汉字）
      · 只有两行的候选必须中文足够多（表尾/表头片段 vs 两行公式，形状一样）
      · 列数集中在 2~8 之间且占比 >= 50%（折线图列数忽多忽少、12 列）
    """
    if not rows or len(rows) < 2:
        return False
    if _looks_like_code(rows):
        return False
    cjk = _cjk_count(_text_of(rows))
    if cjk < 3:
        return False
    if len(rows) == 2 and cjk < 8:
        return False
    widths = [len(r) for r in rows]
    if max(widths) > 8:
        return False                       # 折线图那种十几列的网格
    multi_ratio = sum(1 for w in widths if w >= 2) / len(rows)
    # 表头通常是 3 列以上；有这个特征时，容忍"单元格文字太长被折成单列"的行
    # （表1-1 的对比表就是 11 行里只有 6 行是多列，严格 0.5 会被误杀）
    threshold = 0.4 if max(widths) >= 3 else 0.5
    if multi_ratio < threshold:
        return False
    return True


# 代码里常见、而中文表格里几乎没有的片段
_CODE_MARKERS = ("import", "def ", "self.", "print(", "return", "()")


def _text_of(rows):
    return " ".join(c["text"] for r in rows for c in r)


def _cjk_count(text):
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _looks_like_code(rows):
    """代码被 OCR 切成多列后，形状和表格几乎一样，只能靠内容特征区分。"""
    return any(m in _text_of(rows) for m in _CODE_MARKERS)


def _extract_caption(page_text, table_key):
    """取表题，例如 '表4-1 LangChain提供的特殊文本切块方法' → 'LangChain提供的特殊文本切块方法'。

    坑：正文里先出现的往往是**引用句**（"如表1-1清晰地展示了两者的区别"），
    直接取第一个匹配会拿到"清晰地展示了两者的区别。"。所以要跳过带动词的候选。
    """
    pat = re.compile(re.escape(table_key) + r"\s*([^\n（(]{2,50})")
    for m in pat.finditer(page_text or ""):
        cand = m.group(1).strip(" 。，、")
        if not cand or len(cand) < 4:
            continue
        if _BAD_CAPTION.match(cand):          # 以"所示/展示/给出"开头 → 引用句
            continue
        if any(w in cand for w in ("所示", "展示", "如下", "见表", "如 表")):
            continue
        return cand
    return ""


def _extract_context(page_text, table_key):
    """取引用句：含 '如表4-1所示' 的那一句（排除表题本身那一行）。"""
    text = page_text or ""
    for sent in re.split(r"(?<=[。！？；])", text):
        s = sent.strip()
        if not s or s.startswith(table_key):
            continue
        if table_key in s and any(k in s for k in ("所示", "如下", "见表", "如表")):
            return s.replace("\n", "")[:120]
    return ""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--tables", default="test/baselines/table_images.json")
    ap.add_argument("--apply", action="store_true", help="写入 ES（默认只预览）")
    ap.add_argument("--document-id", type=int, default=3)
    ap.add_argument("--knowledge-id", type=int, default=2)
    args = ap.parse_args()

    import pymupdf
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        sys.exit("没装 OCR 引擎：pip install rapidocr-onnxruntime")

    with open(args.tables, "r", encoding="utf-8") as f:
        tables = json.load(f)["tables"]

    doc = pymupdf.open(args.pdf)
    page_texts = {i: (doc[i].get_text() or "") for i in range(len(doc))}

    # 页 + 图序号 → 图片 bytes
    img_index = {}
    for pno in range(len(doc)):
        idx = 0
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 1:
                continue
            if b.get("width", 0) < 60 or b.get("height", 0) < 20:
                continue
            idx += 1
            img_index[(pno, idx)] = b["image"]

    engine = RapidOCR()
    tmp = os.path.join(os.environ.get("TEMP", "."), "_table_ocr.png")

    blocks = []
    for entry in tables:
        key = entry["table"]
        m = _TABLE_NO.search(key)
        table_no = f"表{m.group(1)}-{m.group(2)}" if m else key

        parts, pages, used_imgs = [], [], []
        for im in entry["images"]:
            img = img_index.get((im["page"], im["idx"]))
            if img is None:
                continue
            rows = _ocr_rows(engine, img, tmp)
            if not _is_table(rows):
                continue          # 不是表格（代码/公式/图表）→ 丢掉
            parts.append(_rows_to_markdown(rows))
            pages.append(im["page"])
            used_imgs.append([im["page"], im["idx"]])
        if not parts:
            continue

        caption, context = "", ""
        for p in entry["ref_pages"]:
            caption = caption or _extract_caption(page_texts.get(p, ""), table_no)
            context = context or _extract_context(page_texts.get(p, ""), table_no)

        block_text = "\n".join(filter(None, [
            f"【表题】{table_no} {caption}".strip(),
            f"【正文】{context}".strip() if context else "",
            "【表格】",
            "\n\n".join(parts),
        ]))
        blocks.append({
            "table": table_no,
            "page": min(pages) if pages else 0,
            "pages": sorted(set(pages)),
            "images": len(parts),
            "used_imgs": used_imgs,
            "text": block_text,
        })

    print(f"候选 {len(tables)} 个表号 → 确认 {len(blocks)} 张真表格\n")
    # 把"确认是表格的图"落盘，供代码通道排除用（用候选清单会误伤代码图）
    out_confirmed = os.path.join(PROJECT_ROOT, "test", "baselines",
                                 "confirmed_table_images.json")
    with open(out_confirmed, "w", encoding="utf-8") as f:
        json.dump(sorted({tuple(x) for b in blocks for x in b["used_imgs"]}), f,
                  ensure_ascii=False)
    for b in blocks:
        print("=" * 78)
        print(f"{b['table']}  第 {b['pages']} 页  由 {b['images']} 张图拼成  "
              f"块文本 {len(b['text'])} 字")
        print("-" * 78)
        print(b["text"][:500])
        if len(b["text"]) > 500:
            print(f"...（共 {len(b['text'])} 字）")

    if args.apply:
        from es_api import es
        from rag_api import RAG
        rag = RAG()
        # 幂等：先清掉该文档以前的表格块（按 chunk_type 字段，不能用 wildcard 查 _id）
        resp = es.delete_by_query(
            index="chunk_info",
            query={"bool": {"must": [{"term": {"document_id": args.document_id}},
                                     {"term": {"chunk_type.keyword": "table"}}]}},
            refresh=True,
        )
        print(f"  清理旧表格块: {resp.get('deleted', 0)} 条")
        for b in blocks:
            vec = rag.get_embedding(b["text"])
            es.index(index="chunk_info", id=f"doc_{args.document_id}_table_{b['table']}",
                     document={
                         "document_id": args.document_id,
                         "knowledge_id": args.knowledge_id,
                         "owner_id": 1, "department_id": 1,
                         "page_number": b["page"],
                         "chunk_id": 0,
                         "chunk_type": "table",
                         "chunk_content": b["text"],
                         "chunk_images": [], "chunk_tables": [],
                         "embedding_vector": [float(x) for x in list(vec)],
                     })
            print(f"  已写入 {b['table']}")
    else:
        print("\n（预览模式，没有写入 ES。确认格式后加 --apply）")


if __name__ == "__main__":
    main()
