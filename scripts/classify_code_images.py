"""阶段 2 第 1 步：抽样判定"哪些图片是代码"（不跑全量，用来调规则）。

定位候选：正文里出现"代码如下所示""示例代码""测试代码如下"等引用的页，
取其**当前页 + 后一页**的图片（代码一般跟在引用句下方）。

分类规则（在 OCR 结果上判断）：
  表格 → 复用 build_table_chunks 的判定
  代码 → 含代码关键词，或代码符号占比高
  公式 → 数学符号多、中文极少
  架构图 → 文本块很少，或平均字长很短

用法：
    python scripts/classify_code_images.py --pdf xxx.pdf --sample 24
    python scripts/classify_code_images.py --pdf xxx.pdf --sample 24 --seed 7
"""
import argparse
import os
import random
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import build_table_chunks as T  # noqa: E402  复用 OCR 与判定逻辑

# 正文里"引用一段代码"的说法
_CODE_REF = re.compile(
    r"代码如下|示例代码|测试代码|如下代码|以下代码|上述代码|代码如|代码所示"
    r"|代码实现|实现代码|代码如下所示|源码如下"
)

_CODE_KW = ("import", "def ", "class ", "self.", "print(", "return", "()",
            "#include", "console.log", "function ", "async ", "await ")
# 代码语句特征：赋值（x = ...）或属性访问（self.x / nn.Linear）
# 注意不能用"标识符后跟括号"当特征 —— 示意图里的数学标注 w(t-2) 完全符合这个形状，
# 但真代码几乎都会出现 = 赋值或 . 属性访问。
_CODE_STMT = re.compile(r"[A-Za-z_]\w*\s*=\s*\S|[A-Za-z_]\w*\.[A-Za-z_]\w*")
# 代码里的"硬特征"符号：赋值、下划线命名、属性访问
_CODE_HARD_SYM = re.compile(r"[=_.]")
_MATH_SYM = set("∑√∫±×÷^≈≤≥∞")
_CODE_SYM = set("=(){}[]:;_")


def _cjk_outside_comment(rows):
    """注释行（以 # 开头）以外有多少个汉字。

    这是区分"真代码"和"示意图"的关键：真代码截图里的中文只出现在注释里，
    而示意图/流程图的中文是独立标签（"输人 输出 w(t-2)…"）。

    要同时处理**行内注释**：`config = LoraConfig(...)  #不同的模型，我们需要调整`
    这种中文在行内，只按"行首是不是 #"判断会漏掉 → 真代码被判成非代码。
    """
    n = 0
    for r in rows:
        line = " ".join(c["text"] for c in r).split("#")[0]
        n += T._cjk_count(line)
    return n


def classify(rows):
    """返回 (类型, 依据)。判断顺序很重要：

    1. 表格优先（结构特征最明确）
    2. 代码关键词（最可靠，`import`/`def`/`self.` 不会出现在示意图里）
    3. 符号密度 —— **但要求几乎不含中文**。示意图（"输人 输出 w(t-2)…"）的括号
       会把符号密度顶上去，而它明显带中文标注；真正的代码截图基本没有中文
       （只有注释，那种情况第 2 步的关键词已经抓住了）。
    4. 公式 → 5. 架构图 → 6. 其他
    """
    if not rows:
        return "空白", "无文本"
    text = T._text_of(rows)
    cjk = T._cjk_count(text)
    csym = sum(1 for c in text if c in _CODE_SYM)
    msym = sum(1 for c in text if c in _MATH_SYM)
    kw = [k for k in _CODE_KW if k in text]

    if T._is_table(rows):
        return "表格", "多列结构"
    if kw:
        return "代码", f"含关键词 {kw[:2]}"
    cjk_out = _cjk_outside_comment(rows)
    if _CODE_STMT.search(text) and cjk_out <= 2:
        return "代码", "代码语句模式"
    density = csym / len(text) if text else 0
    if density > 0.10 and cjk_out <= 2 and _CODE_HARD_SYM.search(text):
        return "代码", f"符号密度 {density:.0%}"
    if msym >= 3 and cjk < 3:
        return "公式", f"数学符号 {msym} 个"
    avg = len(text) / len(rows)
    if len(rows) <= 8 or avg <= 8:
        return "架构图", f"{len(rows)} 行 / 平均 {avg:.0f} 字"
    return "其他", f"{len(rows)} 行 / 中文 {cjk}"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lookahead", type=int, default=1)
    args = ap.parse_args()

    import pymupdf
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        sys.exit("没装 OCR 引擎：pip install rapidocr-onnxruntime")

    doc = pymupdf.open(args.pdf)
    n_pages = len(doc)

    # 每页图片
    page_imgs = {}
    for pno in range(n_pages):
        idx = 0
        items = []
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 1:
                continue
            if b.get("width", 0) < 60 or b.get("height", 0) < 20:
                continue
            idx += 1
            items.append((idx, b))
        page_imgs[pno] = items

    # 只排除"已**确认**是表格的图"。
    # 千万别用表格**候选**清单（table_images.json）来排除 —— 它里面混着代码/公式/图表，
    # 会把第108页图1 那种真代码一起排掉。
    table_imgs = set()
    confirmed = os.path.join(PROJECT_ROOT, "test", "baselines", "confirmed_table_images.json")
    if os.path.exists(confirmed):
        import json
        with open(confirmed, "r", encoding="utf-8") as f:
            table_imgs = {tuple(x) for x in json.load(f)}

    # 引用句定位候选
    ref_pages = [p for p in range(n_pages) if _CODE_REF.search(doc[p].get_text() or "")]
    cand = []
    for p in ref_pages:
        for q in range(p, min(p + args.lookahead + 1, n_pages)):
            for idx, b in page_imgs.get(q, []):
                if (q, idx) in table_imgs:
                    continue
                w, h = b.get("width", 0), b.get("height", 0)
                cand.append((q, idx, w, h, b["image"]))
    seen, uniq = set(), []
    for c in cand:
        if (c[0], c[1]) in seen:
            continue
        seen.add((c[0], c[1]))
        uniq.append(c)

    print(f"全书图片 {sum(len(v) for v in page_imgs.values())} 张")
    print(f"含代码引用的页 {len(ref_pages)} 页 → 候选 {len(uniq)} 张")

    random.seed(args.seed)
    sample = uniq if len(uniq) <= args.sample else random.sample(uniq, args.sample)
    sample.sort(key=lambda x: (x[0], x[1]))
    print(f"抽样 {len(sample)} 张做分类（seed={args.seed}）\n")

    eng = RapidOCR()
    tmp = os.path.join(os.environ.get("TEMP", "."), "_cls_code.png")
    stat = {}
    for pno, idx, w, h, img in sample:
        rows = T._ocr_rows(eng, img, tmp)
        kind, why = classify(rows)
        stat[kind] = stat.get(kind, 0) + 1
        preview = T._text_of(rows)[:110].replace("\n", " ")
        print(f"第{pno:>3}页 图{idx}  {w:.0f}x{h:.0f}  →  【{kind}】({why})")
        print(f"          {preview}")

    print("\n分类分布:", stat)


if __name__ == "__main__":
    main()
