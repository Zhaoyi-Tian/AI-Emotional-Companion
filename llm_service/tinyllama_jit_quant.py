import mindspore
from mindnlp.transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache
from mindnlp.core import ops
from mindnlp.configs import set_pyboost, ON_ORANGE_PI
from mindnlp.quant.smooth_quant import quantize, w8x8
import time

if ON_ORANGE_PI:
    mindspore.set_context(
        device_target='Ascend',
        enable_graph_kernel=False,  # 禁用 Graph Kernel，避免 ascend310b 不支持导致性能异常
        mode=mindspore.GRAPH_MODE,
        jit_config={
            "jit_level": "O2",
        },
    )

NUM_TOKENS_TO_GENERATE = 40

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, ms_dtype=mindspore.float16, low_cpu_mem_usage=True)
quantize_cfg = w8x8(model.model.config)
quantize(model, cfg=quantize_cfg)
model.jit()

set_pyboost(False)

@mindspore.jit(jit_config=mindspore.JitConfig(jit_syntax_level='STRICT'))
def decode_one_token(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token

def stream_predict(prompt, model, tokenizer, num_tokens_to_generate):
    inputs = tokenizer([prompt], return_tensors="ms")
    input_ids = inputs["input_ids"]
    seq_length = input_ids.shape[1]
    batch_size = 1

    generated_ids = ops.zeros((batch_size, seq_length + num_tokens_to_generate), dtype=mindspore.int32)
    generated_ids[:, :seq_length] = input_ids.to(mindspore.int32)

    past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, dtype=model.dtype)
    cache_position = ops.arange(seq_length)

    # 第一个 token
    logits, past_key_values = model(
        input_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )
    next_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]
    cache_position = ops.arange(seq_length + 1, seq_length + 2)

    for step in range(1, num_tokens_to_generate):
        logits, past_key_values = model(
            next_token,
            position_ids=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )
        next_token = ops.argmax(logits[:, -1], dim=-1)[:, None]
        generated_ids[:, seq_length + step] = next_token[:, 0]
        cache_position = ops.arange(seq_length + step + 1, seq_length + step + 2)

        text = tokenizer.batch_decode(generated_ids[:, :seq_length + step + 1], skip_special_tokens=True)
        yield text[0]

def main():
    chat_history = []
    print("Chatbot已启动！输入'exit'退出对话。")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "exit":
            print("对话结束。")
            break

        # 构造完整 prompt
        prompt = ""
        for q, a in chat_history:
            prompt += f"用户: {q}\n助手: {a}\n"
        prompt += f"用户: {user_input}\n助手: "

        print("助手: ", end='', flush=True)
        start_time = time.time()
        prev_token_time = start_time
        token_times = []
        full_response = ""

        for response in stream_predict(prompt, model, tokenizer, NUM_TOKENS_TO_GENERATE):
            # 只取新生成内容
            new_content = response[len(prompt) + len(full_response):]
            if new_content:
                current_time = time.time()
                token_duration = current_time - prev_token_time
                token_times.append(token_duration)
                prev_token_time = current_time
                print(new_content, end='', flush=True)
                full_response += new_content

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