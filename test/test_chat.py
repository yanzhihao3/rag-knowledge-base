"""
多轮对话测试脚本
运行方式：python test_chat.py
"""
import requests
import json

base_url = "http://localhost:6010"
knowledge_id = 1

# 初始化对话历史
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

    # 添加用户消息到历史
    history.append({"role": "user", "content": user_input})

    # 发送请求
    try:
        resp = requests.post(
            f"{base_url}/chat",
            json={
                "knowledge_id": knowledge_id,
                "message": history
            },
            timeout=120
        )

        if resp.status_code == 200:
            result = resp.json()
            answer = result["message"][-1]["content"]
            print(f"\nAI: {answer}")

            # 把 AI 的回答也加入历史
            history.append({"role": "system", "content": answer})
        else:
            print(f"\n错误: {resp.status_code} - {resp.text}")
            # 移除失败的用户消息
            history.pop()

    except requests.exceptions.Timeout:
        print("\n请求超时，请重试")
        history.pop()
    except Exception as e:
        print(f"\n异常: {e}")
        history.pop()

print("\n对话结束")