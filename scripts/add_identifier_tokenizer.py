"""给 chunk_content 加一个"会拆标识符"的 sub-field：chunk_content.delim

**要解决的问题**（实测）：
    PythonCodeTextSplitter  → IK 切成 ['pythoncodetextsplitter']   ← 一个整体
    用户搜 TextSplitter      → 切成 ['textsplitter']
                              两者不相等 → 搜不到
也就是说用户必须**打全名**才能搜到，而真实场景里没人记得住完整类名。

**做法**：
  1. 新增自定义分析器 code_identifier = standard 分词 + word_delimiter_graph 拆词 + lowercase

     为什么必须是 standard 而不是 ik_max_word：IK 会先把英文转成小写，
     word_delimiter_graph 就看不到驼峰边界了 —— 实测 ik_max_word 版
     PythonCodeTextSplitter 依然是一个 token；换成 standard 才拆成
     ['python', 'code', 'text', 'splitter']。
     代价：standard 会把中文切成单字（'混合检索' → 混/合/检/索），
     所以**查询侧只把英文标识符送到这个字段**，中文查询仍走主字段，避免噪音。
  2. 挂到 chunk_content 的子字段 delim 上
  3. 让已有文档重新索引，把新字段填上（加 sub-field 不会自动回填老文档）

用法：
    python scripts/add_identifier_tokenizer.py            # 只看现状
    python scripts/add_identifier_tokenizer.py --apply
"""
import argparse
import json
import sys
import urllib.request

ES = "http://localhost:9200"
INDEX = "chunk_info"

ANALYZER = {
    "analysis": {
        "analyzer": {
            "code_identifier": {
                "type": "custom",
                "tokenizer": "standard",
                # 顺序：先按大小写/下划线拆，再转小写（顺序反了就拆不出来）
                "filter": ["word_delimiter_graph", "lowercase"],
            }
        }
    }
}

MAPPING = {
    "properties": {
        "chunk_content": {
            "type": "text",
            "analyzer": "ik_max_word",
            "fields": {
                "delim": {
                    "type": "text",
                    "analyzer": "code_identifier",
                    "search_analyzer": "code_identifier",
                }
            },
        }
    }
}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ES + path, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"__error": e.code, "body": e.read().decode()[:300]}


def analyze(field_or_analyzer, text):
    body = {"text": text}
    if "." in field_or_analyzer:
        body["field"] = field_or_analyzer
    else:
        body["analyzer"] = field_or_analyzer
    r = call("POST", f"/{INDEX}/_analyze", body)
    return [t["token"] for t in r.get("tokens", [])] if "tokens" in r else r


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    m = call("GET", f"/{INDEX}/_mapping")[INDEX]["mappings"]["properties"]["chunk_content"]
    has_sub = "delim" in m.get("fields", {})
    n_docs = call("GET", f"/{INDEX}/_count").get("count")

    print(f"索引 {INDEX}：{n_docs} 条文档")
    print(f"chunk_content 已有 delim 子字段: {has_sub}")
    print(f"当前分词效果: PythonCodeTextSplitter → {analyze('chunk_content', 'PythonCodeTextSplitter')}")

    if has_sub:
        print(f"delim 字段分词效果: → {analyze('chunk_content.delim', 'PythonCodeTextSplitter')}")
        if not args.apply:
            print("\n已经装过了，无需重复操作。")
            return

    if not args.apply:
        print("\n（预览模式。加 --apply 执行：关索引 → 加分析器 → 加子字段 → 重新索引）")
        return

    # === 1. 加分析器（analysis 是静态配置，必须先关索引）===
    print("\n[1/4] 关闭索引（analysis 设置不能在线改）")
    print("     ", call("POST", f"/{INDEX}/_close"))
    print("[2/4] 写入自定义分析器")
    print("     ", call("PUT", f"/{INDEX}/_settings", ANALYZER))
    print("[3/4] 打开索引")
    print("     ", call("POST", f"/{INDEX}/_open"))

    # === 2. 加子字段 ===
    print("[4/4] 加 chunk_content.delim 子字段")
    print("     ", call("PUT", f"/{INDEX}/_mapping", MAPPING))

    # === 3. 让已有文档重新索引，把新字段填上 ===
    # 加 sub-field 只影响新写入的文档；老文档必须重写一次才会填上新字段。
    print("\n重新索引已有文档（detect_noop=false 强制重写）...")
    # 加 sub-field 不会自动回填老文档，必须让它们重新写一遍。
    # 但 update_by_query 默认 detect_noop=true —— 脚本没产生变化就跳过，
    # 所以给所有文档打一个 schema_version 标记：既有意义（表结构版本），
    # 又能强制重写、把新的 delim 子字段填上。（detect_noop 不是合法的 URL 参数，别试）
    r = call("POST", f"/{INDEX}/_update_by_query?refresh=true&conflicts=proceed",
             {"query": {"match_all": {}},
              "script": {"source": "ctx._source.schema_version = 2"}})
    if "__error" in r:
        print("     失败:", r)
        return
    print(f"     处理 {r.get('total')} 条，更新 {r.get('updated')} 条，失败 {len(r.get('failures') or [])}")

    # === 4. 验证 ===
    print("\n验证：")
    for text in ["PythonCodeTextSplitter", "TextSplitter", "parent_splitter", "混合检索"]:
        print(f"  delim 字段 {text:<26} → {analyze('chunk_content.delim', text)}")


if __name__ == "__main__":
    main()
