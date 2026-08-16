"""
多轮对话测试脚本（交互式）
运行方式：python test/test_chat.py

注意：
1. /chat 已改为 SSE 流式（debug→token×N→done），本脚本按行解析 SSE、拼回完整回答。
2. /chat 需要 X-API-Key 头（RAG_API_KEY 环境变量优先，默认 rag-dev-key）。
3. 包在 main() 里，直接跑 pytest 收集时不会执行，避免读 stdin 报错中断套件。
"""
import json
import os

import requests

base_url = "http://localhost:6010"
knowledge_id = 1
api_key = os.environ.get("RAG_API_KEY", "rag-dev-key")
headers = {"X-API-Key": api_key}


def main():
    history = []
    print("=" * 50)
    print("多轮对话测试 - 输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            history = []
            print("对话已清空")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={
                    "knowledge_id": knowledge_id,
                    "message": history,
                },
                headers=headers,
                stream=True,
                timeout=120,
            )

            if resp.status_code != 200:
                print(f"\n错误: {resp.status_code} - {resp.text}")
                history.pop()
                continue

            # 按 SSE 事件逐行解析：token 事件的数据行用 \n 拼回，error 事件提示失败
            answer = ""
            error_msg = ""
            event = None
            data_lines = []

            for line in resp.iter_lines(decode_unicode=True):
                if line == "":
                    # 空行 = 事件结束
                    if event == "token" and data_lines:
                        answer += "\n".join(data_lines)
                    event = None
                    data_lines = []
                elif line.startswith("event:"):
                    event = line[len("event:"):].strip()
                    data_lines = []
                elif line.startswith("data:"):
                    if event == "token":
                        data_lines.append(line[len("data:"):].strip())
                    elif event == "error":
                        try:
                            error_msg = json.loads(line[len("data:"):].strip()).get("message", "生成失败")
                        except Exception:
                            error_msg = "生成失败"

            if error_msg:
                print(f"\nAI: 生成失败 - {error_msg}")
                history.pop()
                continue

            print(f"\nAI: {answer}")
            history.append({"role": "system", "content": answer})

        except requests.exceptions.Timeout:
            print("\n请求超时，请重试")
            history.pop()
        except Exception as e:
            print(f"\n异常: {e}")
            history.pop()

    print("\n对话结束")


if __name__ == "__main__":
    main()
