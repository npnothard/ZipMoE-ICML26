import os
import gc
import torch
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from entry.llm_modeling import MoE
from utils.constants import *
user_home = os.path.expanduser("~")


# 1. Model Selection
print(f"\n\033[1;32mHello, this is ZipMoE. Please select your model.\033[0m")
model_select = input("\033[1;32mSupported models: 1. DeepSeek, 2. Qwen, 3. SwitchTransformers. Please type in the model index: \033[0m")
if model_select == "1":
    target_model_type = "deepseek"
elif model_select == "2":
    target_model_type = "qwen"
elif model_select == "3":
    target_model_type = "switch"
else:
    print(f"\n\033[1;32mUsing default model DeepSeek.\033[0m")
    target_model_type = "deepseek"

# 2. Runtime Setups
checkpoint = f"ZIPMOE-PREFIX/ZipMoE/models/{target_model_type}/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote=True)
config = {
    "offload_path": f"ZIPMOE-PREFIX/ZipMoE/offload/{target_model_type}/",
    "prefetcher_topk": 0,
    "caching_algorithm": "ZipMoE",
    "device_memory_ratio": 0.5,
    "gpu_pool_ratio": 0.95,#0.55,
    "code_type": "LZ4HC",
    "hyperparam_state_margin": 0,
    "num_file_chunks": 3,
    "num_compute_threads": 10,
    "trace_path": f"ZIPMOE-PREFIX/ZipMoE/trace/{target_model_type}_trace.pt",
    "expert_topk": List_expert_topk[target_model_type],
    "num_elements_per_expert": List_num_elements_per_expert[target_model_type],
    "num_tensors_per_expert": List_num_tensors_per_expert[target_model_type],
    "num_expert_layers": List_num_expert_layers[target_model_type],
    "num_experts": List_num_experts[target_model_type],
    "first_k_dense_replace": List_first_k_dense_replace[target_model_type],
}
DISPLAY_NAME = {
    "deepseek": "DeepSeek-V2-Lite-Chat",
    "switch": "Switch-Large-128",
    "qwen": "Qwen1.5-MoE-A2.7B-Chat"
}
print("Model Loading...")
model = MoE(checkpoint, config)
# If the model does not have a chat template, manually assign a default ChatML template
if tokenizer.chat_template is None:
    print("Warning: Tokenizer had no chat template for this model, as this is not a conversational LLM.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

import time
while True:
    try:
        
        TTFT = 0
        TPOT = []
        model.engine.zipmoe_engine.reset_access_counts()
        
        # 1. Obtain user input
        print(f"\n\033[1;32mHello, this is ZipMoE inference system, supporting {DISPLAY_NAME[target_model_type]}. Let's begin our conversations!\033[0m")
        user_input = input("\033[1;33mUser: \033[0m")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting ...")
            break
        
        if not user_input.strip():
            continue

        # Reset messages
        messages = [{"role": "user", "content": user_input}]

        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.1,
            use_cache=True
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        start_time = time.time()

        print("\033[1;36mZipMoE: \033[0m", end="", flush=True)
        
        IsFirstToken = True
        token_time = time.time()
        for new_text in streamer:
            if IsFirstToken:
                TTFT = time.time() - start_time
                token_time = time.time()
                IsFirstToken = False
            else:
                TPOT.append(time.time()-token_time)
                token_time = time.time()
                 
            print(new_text, end="", flush=True)
            
        
        print(f"\n\033[1;36mTrace Analysis: \033[0m")
        print(f"\n\033[1;36mTime-To-First-Token : {TTFT} seconds \033[0m")
        print(f"\n\033[1;36mTime-Per-Output-Token : {sum(TPOT)/len(TPOT)} seconds \033[0m")
        

    except KeyboardInterrupt:
        print("\nInterruption Detected...")
        break
    except Exception as e:
        print(f"\nError: {e}")
    
    finally:

        gc.collect()
        torch.cuda.empty_cache()


# 3. Exit
del tokenizer
del model
gc.collect()
torch.cuda.empty_cache()
os._exit(0)