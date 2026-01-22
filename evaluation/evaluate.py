# Copyright (c) 2026 <ZipMoE / Anonymous Team>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.


from profile_tools import *
import os
import gc
import torch
import argparse
from transformers import AutoTokenizer
from entry.llm_modeling import MoE
from utils.constants import *
import math

parser = argparse.ArgumentParser("ZipMoE Benchmark")
parser.add_argument("--system", type=str, default="ZipMoE")
parser.add_argument(
    "--model_type", 
    type=str, 
    default="deepseek",
    choices=["deepseek", "qwen", "switch"]
)
parser.add_argument("--memory_footprint", type=int, default=25)
parser.add_argument("--prefetcher_topk", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--max_prompt_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--SSD_type", type=str, default = "Samsung970EVO",
                    choices=["Samsung970EVO","AigoP2000"])
parser.add_argument("--gpu_pool_ratio", type=float, default=0.95)
parser.add_argument("--output_dir", type=str, default="ZIPMOE-PREFIX/ZipMoE/evaluation/results/")
parser.add_argument("--dataset_path", type=str, default="ZIPMOE-PREFIX/ZipMoE/evaluation/dataset/sharegpt_gpt4.jsonl")
parser.add_argument("--num_test_prompts", type=int, default=96)
parser.add_argument("--test_random_seed", type=int, default=321)
parser.add_argument("--cache_algorithm", type=str, default="ZipMoE",
                    choices=["ZipMoE","LFU", "LRU", "Marking","FIFO"])

args = parser.parse_args()

benchmark_message = (
    f"\nBenchmarking {args.system} system on {DISPLAY_NAME[args.model_type]}: \n" + 
    f"- memory-footprint: {args.memory_footprint} GB (D-Mem-Ratio: {MEMMAPPING[args.system][args.memory_footprint]})\n" +
    f"- batch-size: {args.batch_size}\n" +
    f"- max-new-tokens: {args.max_new_tokens}\n" +
    f"- cache-algorithm: {args.cache_algorithm}\n"
)

prompts = sample_first_prompts(
    args.dataset_path,
    num_samples=args.num_test_prompts,
    seed=args.test_random_seed,
    max_candidates=500
)

target_model_type = args.model_type
checkpoint = f"ZIPMOE-PREFIX/ZipMoE/models/{target_model_type}/"
config = {
    "offload_path": f"ZIPMOE-PREFIX/ZipMoE/offload/{target_model_type}/",
    "caching_algorithm": args.cache_algorithm,#"ZipMoE",
    "prefetcher_topk": args.prefetcher_topk,
    "device_memory_ratio": MEMMAPPING[args.system][args.memory_footprint],
    "gpu_pool_ratio": args.gpu_pool_ratio,
    "batch_size": args.batch_size,
    "code_type": "LZ4HC",
    "hyperparam_state_margin": 0.1,
    "num_file_chunks": 3,
    "num_compute_threads": 6,
    "trace_path": f"ZIPMOE-PREFIX/ZipMoE/trace/{target_model_type}_trace.pt",
    "expert_topk": List_expert_topk[target_model_type],
    "num_elements_per_expert": List_num_elements_per_expert[target_model_type],
    "num_tensors_per_expert": List_num_tensors_per_expert[target_model_type],
    "num_expert_layers": List_num_expert_layers[target_model_type],
    "num_experts": List_num_experts[target_model_type],
    "first_k_dense_replace": List_first_k_dense_replace[target_model_type],
}

model = MoE(checkpoint, config)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote=True)

custom_kwargs = {}
if "switch" in args.model_type.lower():
    custom_kwargs = {"decoder_start_token_id": 0}
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
elif "qwen" in args.model_type.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
    tokenizer.padding_side = "left"
elif "deepseek" in args.model_type.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
else:
    raise ValueError(f"Model {args.model_type} not supported")
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = model.model_config.eos_token_id

warmup_prompt = "This is the first question for you. Please introduce yourself."
inputs = tokenizer(
        warmup_prompt,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    ).to("cuda:0")

metrics = {}
streamer = Profiler(metrics)
streamer._start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
            inputs.input_ids,
            streamer=streamer,
            max_new_tokens=10,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            **custom_kwargs,
        )

del outputs
del inputs
del streamer
del metrics
clear_model_cache(model, aggressive=True)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()


print(
    f"\n[Warmup complete] Benmarking starts.\n {benchmark_message}"
)

EVALUATION_DATA = {}
EVALUATION_DATA["engine"] = args.system
EVALUATION_DATA["config"] = config
EVALUATION_DATA["data"] = []

outputfile = args.output_dir + (
    f"{args.system}-{args.model_type}-" + 
    f"M{args.memory_footprint}-B{args.batch_size}-L{args.max_new_tokens}-C{args.cache_algorithm}-" +
    f"{get_current_datatime()}"
) + ".json"

os.makedirs(os.path.dirname(outputfile), exist_ok=True)

num_batches = math.ceil( len(prompts) / args.batch_size )
torch.cuda.synchronize()
sleepsecs = 30
for b in range(num_batches):
    print(f"Inference begin in {sleepsecs} seconds. System Cooling ...")
    time.sleep(sleepsecs)
    print(f"Cooling complete! Processing batch ...\n")

    batch_prompts = prompts[
        b * args.batch_size : (b + 1) * args.batch_size
    ]
    inputs = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    ).to("cuda:0")
    metrics = {}
    streamer = BatchProfiler(metrics,args.batch_size,tokenizer.eos_token_id)
    torch.cuda.synchronize()
    streamer._start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
                input_ids = inputs.input_ids,
                streamer=streamer,
                max_new_tokens=args.max_new_tokens,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                **custom_kwargs,
            )
    torch.cuda.synchronize()
    if "switch" in args.model_type.lower():
        generated_ids = outputs
    else:
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


    print(f"\nGenerated Outputs:")
    for i, output in enumerate(decoded_outputs):
        print(f" [{i}] {output}")  
    # 1. Compute TTFT
    valid_ttfts = [x for x in metrics['TTFT']]
    ttft_val = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0
    # 2. Compute TPOT
    flattened_tpot = [t for sublist in metrics["TPOT"] for t in sublist]
    avg_tpot = sum(flattened_tpot) / len(flattened_tpot) if flattened_tpot else 0.0
    # 3. Compute E2E
    valid_e2e = [x for x in metrics['E2E']]
    e2e_val = sum(valid_e2e) / len(valid_e2e) if valid_e2e else 0.0
    # 4. Compute Throughput
    max_e2e = max(valid_e2e) if valid_e2e else 0.0
    throughput = metrics["num_output_tokens"] / max_e2e if max_e2e > 0 else 0.0
    # 5. Print benchmarking results
    print(
        f"{benchmark_message}"
        f"\n[Benchmark Result]: {b+1} / {num_batches} batches processed.\n"
        f"Time-To-First-Token = {ttft_val:.4f}s \n" 
        f"Time-Per-Output-Token = {avg_tpot:.4f}s \n"
        f"End-To-End Latency (Avg) = {e2e_val:.4f}s \n"    
        f"Tokens Generated = {metrics['num_output_tokens']}\n"
        f"Throughput = {throughput:.2f} tokens/sec\n"
    )
    # Insert the metric into a List.
    EVALUATION_DATA["data"].append(metrics)

    try:
        with open(outputfile, 'w', encoding='utf-8') as f:
            json.dump(EVALUATION_DATA, f, indent=4, ensure_ascii=False)
        print(f"\n[Save Success] Benchmark results saved to: {outputfile}")
    except Exception as e:
        print(f"\n[Error] Failed to save results: {e}")
        

    model.engine.zipmoe_engine.reset_access_counts()
    gc.collect()
    torch.cuda.empty_cache()
    
print("Results saved. Forcing exit after 5 secs...")
time.sleep(5)
os._exit(0)
