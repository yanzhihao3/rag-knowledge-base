"""阶段 2a：把"高置信度代码图"变成可检索的块。

定位思路（不靠猜，靠书自己说的话）：正文里凡是有
  「代码如下所示」「示例代码」「测试代码如下」「如下代码」…
的地方，它附近的图大概率是代码 —— 这是高置信度信号。

候选 = 引用句所在页 + 后一页的图（排除已经作为表格入库的图）
判定 = 复用 classify_code_images.classify（表格→代码→公式→架构图→其他）
拼块 = 【小节标题】+【引用句】+【代码】

为什么块文本要带小节标题和引用句：和表格同理 —— 孤立的代码检索不到，
要靠自然语言的锚点才能被"怎么实现 X"这类问句召回。

用法：
    python scripts/build_code_chunks.py --pdf xxx.pdf              # 预览
    python scripts/build_code_chunks.py --pdf xxx.pdf --apply      # 写入 ES
"""
import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import build_table_chunks as T          # noqa: E402  OCR 工具
import classify_code_images as C        # noqa: E402  分类规则

# 小节标题：4.2.2 基于NLTK的文本切分
_TITLE = re.compile(r"(\d+(?:\.\d+)+)\s+([^\n]{2,40})")
# 引用代码的那句话
_REF_SENT = re.compile(r"[^。！？；\n]*?(?:代码如下|示例代码|测试代码|如下代码|以下代码"
                       r"|上述代码|代码如|代码所示|代码实现|实现代码|源码如下)[^。！？；\n]*[。：:；]?")


def _nearest_title(page_text, pos):
    """取引用句之前最近的一个小节标题。"""
    best = ""
    for m in _TITLE.finditer(page_text[:pos] if pos else page_text):
        best = f"{m.group(1)} {m.group(2).strip()}"
    return best


def _ref_sentence(page_text):
    """取第一句引用代码的句子 + 它在原文里的位置。"""
    m = _REF_SENT.search(page_text or "")
    if not m:
        return "", 0
    return m.group(0).strip().replace("\n", "")[:150], m.start()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--apply", action="store_true", help="写入 ES（默认只预览）")
    ap.add_argument("--document-id", type=int, default=3)
    ap.add_argument("--knowledge-id", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    import pymupdf

    doc = pymupdf.open(args.pdf)
    n_pages = len(doc)

    # 每页图片
    page_imgs = {}
    for pno in range(n_pages):
        idx, items = 0, []
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 1:
                continue
            if b.get("width", 0) < 60 or b.get("height", 0) < 20:
                continue
            idx += 1
            items.append((idx, b))
        page_imgs[pno] = items

    # 只排除"已确认为表格的图"（不能用候选清单，那里面混着代码）
    import json
    table_imgs = set()
    confirmed = os.path.join(PROJECT_ROOT, "test", "baselines", "confirmed_table_images.json")
    if os.path.exists(confirmed):
        with open(confirmed, "r", encoding="utf-8") as f:
            table_imgs = {tuple(x) for x in json.load(f)}

    ref_pages = [p for p in range(n_pages) if C._CODE_REF.search(doc[p].get_text() or "")]
    print(f"含代码引用的页 {len(ref_pages)} 页")

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        sys.exit("没装 OCR 引擎：pip install rapidocr-onnxruntime")
    eng = RapidOCR()
    tmp = os.path.join(os.environ.get("TEMP", "."), "_code_ocr.png")

    blocks, stat, scanned = [], {}, 0
    seen = set()          # 相邻两个引用页会覆盖到同一张图，去重避免重复 OCR
    for p in ref_pages:
        page_text = doc[p].get_text() or ""
        ref_sent, ref_pos = _ref_sentence(page_text)
        title = _nearest_title(page_text, ref_pos)
        for q in range(p, min(p + 2, n_pages)):          # 引用句所在页 + 后一页
            for idx, b in page_imgs.get(q, []):
                if (q, idx) in table_imgs:
                    continue
                if (q, idx) in seen:
                    continue
                seen.add((q, idx))
                scanned += 1
                rows = T._ocr_rows(eng, b["image"], tmp)
                kind, why = C.classify(rows)
                stat[kind] = stat.get(kind, 0) + 1
                if kind != "代码":
                    continue
                code_text = " ".join(T._text_of(rows).split())
                block = "\n".join(filter(None, [
                    f"【小节】{title}" if title else "",
                    f"【正文】{ref_sent}" if ref_sent else "",
                    "【代码】",
                    code_text,
                ]))
                blocks.append({"page": q, "idx": idx, "text": block,
                               "raw": code_text, "why": why})
                if args.limit and len(blocks) >= args.limit:
                    break
            if args.limit and len(blocks) >= args.limit:
                break
        if args.limit and len(blocks) >= args.limit:
            break

    print(f"OCR 了 {scanned} 张，分类分布: {stat}")
    print(f"确认代码块 {len(blocks)} 个，平均 {sum(len(b['text']) for b in blocks)//max(1,len(blocks))} 字\n")
    for b in blocks[:8]:
        print("=" * 74)
        print(f"第{b['page']}页 图{b['idx']}  依据: {b['why']}  {len(b['text'])} 字")
        print("-" * 74)
        print(b["text"][:320] + ("…" if len(b["text"]) > 320 else ""))

    if not args.apply:
        print(f"\n（预览模式，共 {len(blocks)} 个块未写入。确认后加 --apply）")
        return

    from es_api import es
    from rag_api import RAG
    rag = RAG()
    # 幂等：先清掉该文档以前的代码块，再重写。
    # 注意不能用 wildcard 查 _id（ES 会报 400：_id 是 [_id] 类型，不支持 wildcard），
    # 所以按 chunk_type 字段删 —— 这也正好是方案里要加的字段。
    resp = es.delete_by_query(
        index="chunk_info",
        query={"bool": {"must": [{"term": {"document_id": args.document_id}},
                                 {"term": {"chunk_type.keyword": "code"}}]}},
        refresh=True,
    )
    print(f"  清理旧代码块: {resp.get('deleted', 0)} 条")
    for i, b in enumerate(blocks, 1):
        vec = rag.get_embedding(b["text"])
        es.index(index="chunk_info", id=f"doc_{args.document_id}_code_{b['page']}_{b['idx']}",
                 document={
                     "document_id": args.document_id,
                     "knowledge_id": args.knowledge_id,
                     "owner_id": 1, "department_id": 1,
                     "page_number": b["page"],
                     "chunk_id": 0,
                     "chunk_type": "code",
                     "chunk_content": b["text"],
                     "chunk_images": [], "chunk_tables": [],
                     "embedding_vector": [float(x) for x in list(vec)],
                 })
        if i % 20 == 0:
            print(f"  已写入 {i}/{len(blocks)}")
    print(f"完成：写入 {len(blocks)} 个代码块")


if __name__ == "__main__":
    main()
