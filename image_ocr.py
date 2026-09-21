"""图片内容提取：把 PDF 里"图片型的表格 / 代码"转成可检索的文本块。

**为什么需要这个模块**
实测那本《大模型RAG实战》：251 张图占 20.9% 版面，代码示例和表格**全是图片**，
`pdfplumber.extract_tables()` 全书 240 页抽到 0 张 —— 任何纯文本解析器都拿不到这些内容。
结果就是用户问"代码怎么写""表里有什么"时，系统一个字都搜不到。

**流程**
  ① 定位候选：表格用正文里的"表X-Y"引用反查；代码用"代码如下所示"这类引用
  ② OCR（本地 RapidOCR，纯 CPU，0 成本）
  ③ 分类确认：表格 / 代码 / 公式 / 架构图 / 其他
  ④ 拼块文本：【表题 / 小节标题】+【引用句】+【内容】

**为什么块文本一定要带标题和引用句**
实测 ES/IK 分词：表题「表4-1 LangChain提供的特殊文本切块方法」和用户问句
「LangChain支持哪些特殊文本切块方法」分词高度重合（langchain/特殊/文本/切块/方法），
而表格内容本身和问句几乎不重合。孤立的表格/代码检索不到，必须带语义锚点。

对外主要接口：`extract_image_blocks(doc)` → `(table_blocks, code_blocks)`
"""
import os
import re
import statistics

# 太小的图是装饰元素，不是内容
MIN_IMG_W, MIN_IMG_H = 60, 20

# ---------- 定位用的正则 ----------
_TABLE_REF = re.compile(r"表\s*(\d+)\s*[-–—]\s*(\d+)")
_CODE_REF = re.compile(
    r"代码如下|示例代码|测试代码|如下代码|以下代码|上述代码|代码如|代码所示"
    r"|代码实现|实现代码|源码如下"
)
_TITLE = re.compile(r"(\d+(?:\.\d+)+)\s+([^\n]{2,40})")          # 4.2.2 基于NLTK的文本切分
_REF_SENT = re.compile(r"[^。！？；\n]*?(?:代码如下|示例代码|测试代码|如下代码|以下代码"
                       r"|上述代码|代码如|代码所示|代码实现|实现代码|源码如下)[^。！？；\n]*[。：:；]?")
_BAD_CAPTION = re.compile(r"^(所示|展示|给出|列出|如下|见|中|里|为|是|说明|描述)")

# ---------- 分类用的特征 ----------
_CODE_KW = ("import", "def ", "class ", "self.", "print(", "return", "()",
            "#include", "console.log", "function ", "async ", "await ")
# 代码语句：赋值（x = ...）或属性访问（self.x / nn.Linear）
# 不能用"标识符后跟括号"当特征 —— 示意图里的数学标注 w(t-2) 完全符合这个形状
_CODE_STMT = re.compile(r"[A-Za-z_]\w*\s*=\s*\S|[A-Za-z_]\w*\.[A-Za-z_]\w*")
_CODE_HARD_SYM = re.compile(r"[=_.]")
_MATH_SYM = set("∑√∫±×÷^≈≤≥∞")
_CODE_SYM = set("=(){}[]:;_")


# ============================================================
# OCR（引擎懒加载：只有真的要用时才吃内存）
# ============================================================
_OCR_ENGINE = None
_OCR_TMP = os.path.join(os.environ.get("TEMP", "."), "_rag_ocr_tmp.png")


def get_ocr():
    """加载 RapidOCR（第一次调用时才加载）。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:      # pragma: no cover
            raise RuntimeError(
                "没装 OCR 引擎，请先执行：pip install rapidocr-onnxruntime"
            ) from e
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def ocr_rows(engine, img_bytes, tmp_path=None):
    """OCR 一张图，返回按 y 坐标聚成的行：[[{text, cx, cy, h}, ...], ...]。"""
    tmp_path = tmp_path or _OCR_TMP
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    out = engine(tmp_path)
    result = out[0] if isinstance(out, tuple) else out

    items = []
    for row in (result or []):
        box, text = row[0], str(row[1])
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": text, "cx": sum(xs) / len(xs),
                      "cy": sum(ys) / len(ys), "h": max(ys) - min(ys)})
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


# ============================================================
# 分类：这段文字是"表格 / 代码 / 公式 / 架构图 / 其他"？
# ============================================================
def text_of(rows):
    return " ".join(c["text"] for r in rows for c in r)


def cjk_count(text):
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def cjk_outside_comment(rows):
    """注释（# 之后，含行内注释）以外有多少个汉字。

    这是区分"真代码"和"示意图"的关键：真代码截图里的中文只出现在注释里，
    而示意图的中文是独立标签（"输人 输出 w(t-2) w(t-1)…"）。
    """
    return sum(cjk_count(" ".join(c["text"] for c in r).split("#")[0]) for r in rows)


def looks_like_code(rows):
    return any(m in text_of(rows) for m in _CODE_KW)


def is_table(rows):
    """判定是不是真表格。

    只按"多列行占比"会误判：折线图 OCR 出来是十几列的网格，公式图只有 2 行，
    代码截图也会被切成多列。所以综合四条判断。
    """
    if not rows or len(rows) < 2:
        return False
    if looks_like_code(rows):
        return False
    cjk = cjk_count(text_of(rows))
    if cjk < 3:
        return False                       # 公式/符号图几乎没有汉字
    if len(rows) == 2 and cjk < 8:
        return False                       # 两行且中文很少 → 大概率是公式片段
    widths = [len(r) for r in rows]
    if max(widths) > 8:
        return False                       # 折线图那种十几列的网格
    multi_ratio = sum(1 for w in widths if w >= 2) / len(rows)
    # 表头通常 3 列以上；有这个特征时容忍"单元格文字太长被折成单列"的行
    threshold = 0.4 if max(widths) >= 3 else 0.5
    return multi_ratio >= threshold


def classify(rows):
    """返回 (类型, 依据)。判断顺序很重要，顺序错了会互相抢。"""
    if not rows:
        return "空白", "无文本"
    text = text_of(rows)
    csym = sum(1 for c in text if c in _CODE_SYM)
    msym = sum(1 for c in text if c in _MATH_SYM)
    kw = [k for k in _CODE_KW if k in text]

    if is_table(rows):
        return "表格", "多列结构"
    if kw:
        return "代码", f"含关键词 {kw[:2]}"
    cjk_out = cjk_outside_comment(rows)
    if _CODE_STMT.search(text) and cjk_out <= 2:
        return "代码", "代码语句模式"
    density = csym / len(text) if text else 0
    if density > 0.10 and cjk_out <= 2 and _CODE_HARD_SYM.search(text):
        return "代码", f"符号密度 {density:.0%}"
    if msym >= 3 and cjk_count(text) < 3:
        return "公式", f"数学符号 {msym} 个"
    avg = len(text) / len(rows)
    if len(rows) <= 8 or avg <= 8:
        return "架构图", f"{len(rows)} 行 / 平均 {avg:.0f} 字"
    return "其他", f"{len(rows)} 行 / 中文 {cjk_count(text)}"


def rows_to_markdown(rows):
    """把 OCR 的行拼成 Markdown 表格。"""
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


# ============================================================
# 定位候选
# ============================================================
def page_image_map(doc):
    """返回 {(页, 图序号): block}，图序号从 1 开始，过滤掉过小的装饰图。"""
    out = {}
    for pno in range(len(doc)):
        idx = 0
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 1:
                continue
            if b.get("width", 0) < MIN_IMG_W or b.get("height", 0) < MIN_IMG_H:
                continue
            idx += 1
            out[(pno, idx)] = b
    return out


def find_table_candidates(doc, lookahead=1):
    """用正文里的"表X-Y"引用反查候选图。

    返回 [{table: '表4-1', ref_pages: [...], images: [(页, 序号), ...]}, ...]
    """
    n = len(doc)
    refs = {}
    for pno in range(n):
        for m in _TABLE_REF.finditer(doc[pno].get_text() or ""):
            refs.setdefault(f"表{m.group(1)}-{m.group(2)}", set()).add(pno)
    img_map = page_image_map(doc)
    out = []
    for key in sorted(refs, key=lambda k: [int(x) for x in re.findall(r"\d+", k)]):
        pages = sorted(refs[key])
        cand_pages = {q for p in pages
                      for q in range(p, min(p + lookahead + 1, n))}
        imgs = [(q, i) for q in sorted(cand_pages) for (q2, i) in img_map if q2 == q]
        out.append({"table": key, "ref_pages": pages, "images": imgs})
    return out


def find_code_candidates(doc, lookahead=1):
    """用正文里的"代码如下所示"等引用定位候选图（引用句所在页 + 后 lookahead 页）。"""
    n = len(doc)
    img_map = page_image_map(doc)
    ref_pages = [p for p in range(n) if _CODE_REF.search(doc[p].get_text() or "")]
    seen, out = set(), []
    for p in ref_pages:
        for q in range(p, min(p + lookahead + 1, n)):
            for (q2, i) in img_map:
                if q2 != q or (q, i) in seen:
                    continue
                seen.add((q, i))
                out.append((q, i))
    return out, ref_pages


# ============================================================
# 拼块
# ============================================================
def _extract_caption(page_text, table_key):
    """取表题，如 '表4-1 LangChain提供的特殊文本切块方法' → 'LangChain提供的特殊文本切块方法'。

    坑：正文里先出现的往往是引用句（"如表1-1清晰地展示了两者的区别"），
    直接取第一个匹配会拿到"清晰地展示了两者的区别。"，所以要跳过带动词的候选。
    """
    pat = re.compile(re.escape(table_key) + r"\s*([^\n（(]{2,50})")
    for m in pat.finditer(page_text or ""):
        cand = m.group(1).strip(" 。，、")
        if not cand or len(cand) < 4 or _BAD_CAPTION.match(cand):
            continue
        if any(w in cand for w in ("所示", "展示", "如下", "见表")):
            continue
        return cand
    return ""


def _extract_context(page_text, table_key):
    """取引用句（含"如表X所示"的那一句），排除表题本身那一行。"""
    for sent in re.split(r"(?<=[。！？；])", page_text or ""):
        s = sent.strip()
        if not s or s.startswith(table_key):
            continue
        if table_key in s and any(k in s for k in ("所示", "如下", "见表", "如表")):
            return s.replace("\n", "")[:120]
    return ""


def _nearest_title(page_text, pos):
    """取引用句之前最近的一个小节标题。"""
    best = ""
    for m in _TITLE.finditer((page_text or "")[:pos] if pos else (page_text or "")):
        best = f"{m.group(1)} {m.group(2).strip()}"
    return best


def _ref_sentence(page_text):
    m = _REF_SENT.search(page_text or "")
    return (m.group(0).strip().replace("\n", "")[:150], m.start()) if m else ("", 0)


def build_table_blocks(doc, engine, log=None, lookahead=1):
    """表格：定位 → OCR → 判定 → 拼块。返回 block 列表。"""
    img_map = page_image_map(doc)
    candidates = find_table_candidates(doc, lookahead)
    blocks = []
    for entry in candidates:
        key = entry["table"]
        parts, pages, used = [], [], []
        for (pno, idx) in entry["images"]:
            blk = img_map.get((pno, idx))
            if blk is None:
                continue
            rows = ocr_rows(engine, blk["image"])
            if not is_table(rows):
                continue          # 不是表格（代码/公式/图表）→ 丢掉
            parts.append(rows_to_markdown(rows))
            pages.append(pno)
            used.append((pno, idx))
        if not parts:
            continue

        caption = context = ""
        for p in entry["ref_pages"]:
            txt = doc[p].get_text() or ""
            caption = caption or _extract_caption(txt, key)
            context = context or _extract_context(txt, key)

        text = "\n".join(filter(None, [
            f"【表题】{key} {caption}".strip(),
            f"【正文】{context}".strip() if context else "",
            "【表格】",
            "\n\n".join(parts),
        ]))
        blocks.append({"kind": "table", "key": key, "page": min(pages),
                       "pages": sorted(set(pages)), "used_images": used, "text": text})
        if log:
            log(f"[image_ocr] 表格 {key}（第 {min(pages)} 页，{len(parts)} 张图拼成）")
    return blocks


def build_code_blocks(doc, engine, exclude_images=None, log=None, lookahead=1):
    """代码：定位 → OCR → 判定 → 拼块。返回 block 列表。"""
    exclude_images = exclude_images or set()
    img_map = page_image_map(doc)
    cand, ref_pages = find_code_candidates(doc, lookahead)
    blocks = []
    for (pno, idx) in cand:
        if (pno, idx) in exclude_images:
            continue
        blk = img_map.get((pno, idx))
        if blk is None:
            continue
        rows = ocr_rows(engine, blk["image"])
        kind, why = classify(rows)
        if kind != "代码":
            continue
        # 引用句取"该页（或前一页）里最近的代码引用"
        ref_sent, title = "", ""
        for p in (pno, pno - 1):
            if p < 0 or not _CODE_REF.search(doc[p].get_text() or ""):
                continue
            ref_sent, pos = _ref_sentence(doc[p].get_text() or "")
            title = _nearest_title(doc[p].get_text() or "", pos)
            break
        code_text = " ".join(text_of(rows).split())
        text = "\n".join(filter(None, [
            f"【小节】{title}" if title else "",
            f"【正文】{ref_sent}" if ref_sent else "",
            "【代码】",
            code_text,
        ]))
        blocks.append({"kind": "code", "page": pno, "idx": idx, "text": text, "why": why})
        if log:
            log(f"[image_ocr] 代码 第{pno}页图{idx}（{why}）")
    return blocks


def extract_image_blocks(doc, log=None, lookahead=1):
    """一站式入口：返回 (table_blocks, code_blocks)。

    代码通道会排除"已确认为表格的图"，避免同一张图既当表格又当代码。
    """
    engine = get_ocr()
    tables = build_table_blocks(doc, engine, log=log, lookahead=lookahead)
    exclude = {im for b in tables for im in b["used_images"]}
    codes = build_code_blocks(doc, engine, exclude_images=exclude,
                              log=log, lookahead=lookahead)
    return tables, codes
