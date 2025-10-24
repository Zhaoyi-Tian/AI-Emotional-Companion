import requests
import time

# 请替换为你的 deepseek 个人 API 密钥
API_KEY = "sk-262419ac92684077aca08779a67e68a4"
API_URL = "https://api.deepseek.com/v1/chat/completions"

system_prompt = "You are a helpful and friendly chatbot"
model_name = "deepseek-chat"  # 可以换成 deepseek-coder 等

def build_messages(chat_history, user_msg):
    messages = [{"role": "system", "content": system_prompt}]
    for user, ai in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": user_msg})
    return messages

def predict(message, history):
    payload = {
        "model": model_name,
        "messages": build_messages(history, message),
        "stream": True,  # 启用流式输出
        "max_tokens": 128,
        "temperature": 1.0,
        "top_p": 0.9
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    with requests.post(API_URL, headers=headers, json=payload, stream=True) as resp:
        partial_message = ""
        for line in resp.iter_lines():
            if line:
                try:
                    # DeepSeek API stream 格式与 OpenAI 类似
                    if line.startswith(b'data: '):
                        line = line[6:]
                    chunk = line.decode("utf-8")
                    if chunk.strip() == "[DONE]":
                        break
                    import json
                    data = json.loads(chunk)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        partial_message += delta
                        yield partial_message
                except Exception:
                    continue

def main():
    chat_history = []
    print("DeepSeek Chatbot已启动！输入'exit'退出对话。")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            print("对话结束。")
            break

        print("助手: ", end='', flush=True)
        start_time = time.time()
        prev_token_time = start_time
        token_times = []
        full_response = ""
        for response in predict(user_input, chat_history):
            new_content = response[len(full_response):]
            current_time = time.time()
            token_duration = current_time - prev_token_time
            token_times.append(token_duration)
            prev_token_time = current_time
            print(new_content, end='', flush=True)
            full_response = response

        total_tokens = len(token_times)
        if total_tokens > 0:
            total_duration = time.time() - start_time
            avg_duration = total_duration / total_tokens
            print(f"\n[性能统计] 共生成 {total_tokens} 个token，总耗时 {total_duration:.2f} 秒，平均每token耗时 {avg_duration:.4f} 秒")
        else:
            print("\n[性能统计] 未生成有效token")

        chat_history.append((user_input, full_response))
        print()

if __name__ == "__main__":
    main()