import gradio as gr
import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import time  # 导入时间模块用于计时
from mindspore._c_expression import disable_multi_thread
disable_multi_thread()
# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ms_dtype=mindspore.float16)
print(model.dtype)


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return mindspore.Tensor(True)
        return mindspore.Tensor(False)


# Function to generate model predictions.
def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="ms")
    streamer = TextIteratorStreamer(tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        top_k=10,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
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
