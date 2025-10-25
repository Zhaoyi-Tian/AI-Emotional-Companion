import gradio as gr
import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread
import time  # 导入时间模块用于计时
from mindspore._c_expression import disable_multi_thread
from mindnlp.quant.smooth_quant import quantize, w8x8
disable_multi_thread()

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat", ms_dtype=mindspore.float16)
model = AutoModelForCausalLM.from_pretrained("/home/HwHiAiUser/.mindnlp/model/Qwen/Qwen1.5-0.5B-Chat", ms_dtype=mindspore.float16)
system_prompt = "You are a helpful and friendly chatbot"



def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': msg})
    return messages

# 生成模型预测的函数
def predict(message, history):
    # 格式化模型输入
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="ms",
            tokenize=True
        )
    streamer = TextIteratorStreamer(tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=1,
        num_beams=1,
        use_cache=True
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # 在单独的线程中启动生成过程
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # 如果生成了停止token则中断循环
            break
        yield partial_message


def main():
    chat_history = []
    print("Chatbot已启动！输入'exit'退出对话。")
    
    while True:
        user_input = input("\n用户: ")
        
        if user_input.lower() == 'exit':
            print("对话结束。")
            break
            
        print("助手: ", end='', flush=True)
        
        # 初始化计时变量
        start_time = time.time()  # 记录生成开始时间
        prev_token_time = start_time  # 记录上一个token的生成时间
        token_times = []  # 存储每个token的生成时间
        
        # 收集完整回复
        full_response = ""
        for response in predict(user_input, chat_history):
            # 获取新增的内容（当前token）
            new_content = response[len(full_response):]
            # 计算当前token的生成时间（与上一个token的时间差）
            current_time = time.time()
            token_duration = current_time - prev_token_time
            token_times.append(token_duration)
            # 更新上一个token的时间
            prev_token_time = current_time
            # 打印新增内容
            print(new_content, end='', flush=True)
            full_response = response
        
        # 计算并显示统计信息
        total_tokens = len(token_times)
        if total_tokens > 0:
            total_duration = time.time() - start_time
            avg_duration = total_duration / total_tokens
            print(f"\n[性能统计] 共生成 {total_tokens} 个token，总耗时 {total_duration:.2f} 秒，平均每token耗时 {avg_duration:.4f} 秒")
        else:
            print("\n[性能统计] 未生成有效token")
        
        # 添加到历史记录
        chat_history.append((user_input, full_response))
        print()  # 换行

if __name__ == "__main__":
    main()