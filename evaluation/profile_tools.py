# Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.


import json
import time
import datetime
import random
import torch
import gc
from transformers.generation.streamers import BaseStreamer


class Profiler(BaseStreamer):
    def __init__(self, metrics):
        self.metrics = metrics
        self._start = None
        self._first = None
        self._last = None

        self.metrics["TTFT"] = None
        self.metrics["TPOT"] = []
        self.metrics["E2E"] = None
        self.metrics["num_output_tokens"] = 0

    def put(self, value):
        now = time.perf_counter()

        if self._first is None:
            self._first = now
            self.metrics["TTFT"] = now - self._start
            self._last = now
        else:
            self.metrics["TPOT"].append(now - self._last)
            self._last = now

        self.metrics["num_output_tokens"] += 1

    def end(self):
        self.metrics["E2E"] = time.perf_counter() - self._start



def get_current_datatime():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sample_first_prompts(
    jsonl_path,
    num_samples=100,
    seed=321,
    max_candidates=500 # from 0 to 499, this is the validation set
):
    random.seed(seed)
    prompts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_candidates:
                break

            data = json.loads(line)
            convs = data.get("conversations", [])

            for turn in convs:
                if turn.get("from") in ("human", "user"):
                    text = turn.get("value", "").strip()
                    if text:
                        prompts.append(text)
                    break

    return random.sample(prompts, min(num_samples, len(prompts)))


class BatchProfiler(BaseStreamer):
    def __init__(self, metrics, batch_size, eos_token_id):
        self.metrics = metrics
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self._start = None
        self._first = [None] * batch_size  
        self._last = [None] * batch_size
        self.finished = [False] * batch_size
        self.prompt_seen = False 
        self.metrics["batch_size"] = batch_size
        self.metrics["TTFT"] = [None] * batch_size
        self.metrics["TPOT"] = [[] for _ in range(batch_size)]
        self.metrics["E2E"] = [None] * batch_size
        self.metrics["num_output_tokens"] = 0

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return

        now = time.perf_counter()

        token_ids = value.tolist()
        for req in range(self.batch_size):
            if self.finished[req]:
                continue
            if self._first[req] is None:
                self._first[req] = now
                self.metrics["TTFT"][req] = now - self._start
                self._last[req] = now
            else:
                self.metrics["TPOT"][req].append(now - self._last[req])
                self._last[req] = now
            
            self.metrics["num_output_tokens"] += 1
            if token_ids[req] == self.eos_token_id:
                self.finished[req] = True
                self.metrics["E2E"][req] = now - self._start


    def end(self):
        total_end_time = time.perf_counter() - self._start
        for req in range(self.batch_size):
            if self.metrics["E2E"][req] is None:
                 self.metrics["E2E"][req] = total_end_time
            if self.metrics["TTFT"][req] is None:
                 self.metrics["TTFT"][req] = [total_end_time]


DISPLAY_NAME = {
    "deepseek": "DeepSeek-V2-Lite-Chat",
    "switch": "Switch-Large-128",
    "qwen": "Qwen1.5-MoE-A2.7B-Chat"
}


# The following mapping is profiled using jtop

memory_map_zipmoe = {
    5: 0.03,
    10: 0.1,
    15: 0.17,
    20: 0.25,
    25: 0.34,
    30: 0.43
}


MEMMAPPING = {
    "ZipMoE": memory_map_zipmoe,
}


def system_clean(model):
    # clear past_key_values
    if hasattr(model.model, 'past_key_values'):
        model.model.past_key_values = None
    
    # Clear all the cache
    if hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.past_key_value = None
    # Clear cuda cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    time.sleep(2)


def clear_model_cache(model, aggressive=True):
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None

    if hasattr(model, 'model'):
        if hasattr(model.model, 'past_key_values'):
            model.model.past_key_values = None
    
    model_core = model.model if hasattr(model, 'model') else model

    layer_names = ['layers', 'decoder', 'encoder', 'h', 'blocks']
    
    for layer_name in layer_names:
        if hasattr(model_core, layer_name):
            layers = getattr(model_core, layer_name)
            if layers is not None:
                try:
                    for layer in layers:
                        if hasattr(layer, 'self_attn'):
                            if hasattr(layer.self_attn, 'past_key_value'):
                                layer.self_attn.past_key_value = None
                            if hasattr(layer.self_attn, 'cache'):
                                layer.self_attn.cache = None

                        if hasattr(layer, 'cross_attn'):
                            if hasattr(layer.cross_attn, 'past_key_value'):
                                layer.cross_attn.past_key_value = None

                        if hasattr(layer, 'past_key_value'):
                            layer.past_key_value = None
                except:
                    pass
    
    # 5. Clear CUDA cache
    if aggressive:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Clear IPC cache
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
    
    return True







