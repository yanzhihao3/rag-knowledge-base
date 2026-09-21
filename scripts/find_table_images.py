"""定位 PDF 里的表格图片（不跑 OCR）：用正文里的"表X-Y"表题反查图片位置。

为什么需要这个：这本书 250 张图里真正是表格的只有十几张。如果对全部 250 张
跑 OCR 要 15-20 分钟，而正文里每张表都有"如表X-Y所示"或"表X-Y 标题"的引用，
用它可以精准定位，只 OCR 那十几张。

输出：表号 → 候选图片（页 + 图序号 + 尺寸），并存成 JSON 供下一步 OCR 使用。

用法：
    python scripts/find_table_images.py --pdf "xxx.pdf" --out tables.json
"""
import argparse
import json
import re
import sys

# "表4-1" / "表 4-1" / "表4—1"（有的书用全角破折号）
_TABLE_REF = re.compile(r"表\s*(\d+)\s*[-–—]\s*(\d+)")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", default="table_images.json")
    parser.add_argument("--lookahead", type=int, default=1,
                        help="表题出现后，往后看几页找图（表格常跨页，默认 1）")
    args = parser.parse_args()

    import pymupdf

    doc = pymupdf.open(args.pdf)
    n_pages = len(doc)

    # 1) 收集每页的图片（页码、序号、宽高）
    page_images = {}
    for pno in range(n_pages):
        imgs = []
        idx = 0
        for block in doc[pno].get_text("dict")["blocks"]:
            if block.get("type") != 1:
                continue
            w, h = block.get("width", 0), block.get("height", 0)
            if w < 60 or h < 20:
                continue
            idx += 1
            imgs.append({"page": pno, "idx": idx, "w": w, "h": h,
                         "ratio": round(w / h, 2) if h else 0})
        page_images[pno] = imgs

    # 2) 扫描正文，找"表X-Y"出现在哪些页
    refs = {}
    for pno in range(n_pages):
        text = doc[pno].get_text()
        for m in _TABLE_REF.finditer(text):
            key = f"表{m.group(1)}-{m.group(2)}"
            refs.setdefault(key, set()).add(pno)

    # 3) 表题所在页 + 后 lookahead 页里的图片 = 候选
    result = []
    used = set()
    for key in sorted(refs, key=lambda k: [int(x) for x in re.findall(r"\d+", k)]):
        pages = sorted(refs[key])
        cand_pages = set()
        for p in pages:
            for q in range(p, min(p + args.lookahead + 1, n_pages)):
                cand_pages.add(q)
        cands = [im for p in sorted(cand_pages) for im in page_images.get(p, [])]
        for c in cands:
            used.add((c["page"], c["idx"]))
        result.append({
            "table": key,
            "ref_pages": pages,
            "images": cands,
        })

    # 4) 打印
    print(f"PDF: {args.pdf}   共 {n_pages} 页")
    print(f"正文里出现的表号: {len(result)} 个")
    print("=" * 76)
    total = 0
    for r in result:
        c = r["images"]
        total += len(c)
        pages = ",".join(str(p) for p in r["ref_pages"])
        if c:
            desc = ", ".join(f"第{i['page']}页图{i['idx']}({i['w']:.0f}x{i['h']:.0f}, {i['ratio']})"
                             for i in c)
        else:
            desc = "** 没找到候选图 **"
        print(f"  {r['table']:<8} 表题在第 {pages:<12} 页 → {desc}")

    print("=" * 76)
    print(f"候选图片合计（含重复）: {total} 张，去重后: {len(used)} 张")
    print(f"全书图片总数: {sum(len(v) for v in page_images.values())} 张")
    print(f"\n→ 只需要 OCR 这 {len(used)} 张，而不是全书 250 张")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tables": result, "unique_images": sorted(used)}, f,
                  ensure_ascii=False, indent=2)
    print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()
