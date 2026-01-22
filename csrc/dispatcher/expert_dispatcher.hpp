// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// Modifications and additions to this file are licensed under the
// Academic Non-Commercial License. See the LICENSE file in the
// project root for details.
//
// -------------------------------------------------------------------
// DERIVED FROM:
// EfficientMoE (Apache License 2.0)
// Copyright (c) EfficientMoE.
//
// The original code is licensed under the Apache License, Version 2.0.
// This file contains substantial modifications.
// -------------------------------------------------------------------


#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "base/noncopyable.hpp"
#include "expert_module.hpp"
#include "base/thread.hpp"
#include "memory/cache.hpp"
#include "kernels/memory_docks.hpp"

enum MUTEX_TYPE{
    INPUT_MUTEX = 0,
    OUTPUT_MUTEX = 1,
    EXEC_MUTEX = 2,
    PENDING_MUTEX = 3
};
class ExpertDispatcher: public base::noncopyable{

public:
    typedef struct{
        int layer_idx = -1;
        int expert_idx = -1;
        int gpu_id = -1;
        bool remote = false;
    } CallArgs;

    typedef struct{
        torch::Tensor hidden_states = torch::empty({0});
        ExpertNodePtr expert_node = nullptr;
        int out_gpu_id = -1;
        torch::ScalarType out_dtype = torch::kBFloat16;
        TaskState task_state;
        bool hit = false;
    } ExecArgs;

    typedef struct{
        ExecArgs exec_args;
        torch::Tensor output;
        int gpu_id;
    } OutputArgs;

    typedef std::tuple< torch::Tensor, int, int, int > CallResult;

    explicit ExpertDispatcher(
        int num_experts,
        int num_layers,
        int expert_type,
        int dtype = DTYPE_BFLOAT16
    );

    ~ExpertDispatcher(){
        main_thread_stop_flag_.store(true);
        for (auto& thread: threads_){
            thread->join();
        }

        for(auto& stream: fetch_streams_){
            cudaStreamDestroy(stream);
        }
        for (auto& stream: exec_streams_){
            cudaStreamDestroy(stream);
        }
        for (auto& stream: out_streams_){
            cudaStreamDestroy(stream);
        }
    }

    void StartThreads(int num_threads = 1);

    void SetInputs(
        const torch::Tensor& hidden_states,
        const torch::Tensor& router_mask
    ){
        hidden_states_ = hidden_states.clone();
        router_mask_ = router_mask.clone();
    }

    void EnqueueExpert(
        int layer_idx,
        int expert_idx,
        int num_token_reqs
    );


    void EnqueuePrefetch(
        int target_layer_idx,
        int predicted_expert_idx
    );

    void SubmitPrefetch(
        int target_layer_idx,
        int predicted_expert_idx,
        double priority
    );

    torch::Tensor ObtainTensor(
        int layer_idx,
        int expert_idx,
        int tensor_id
    );

    void OperationSchedule();
    
    void RegisterExpert(
        int layer_idx, 
        int expert_idx,
        const std::vector<std::uint32_t>& tensor_ids
    );

    void ClearExpertCacheCounts();
    
    void SetExpectedQueue(int expected_pending = 0) {
        pending_.store(expected_pending);
    }

    std::vector<CallResult> WaitExpert() { return Wait(); }

    void SetNode(
        int layer_idx, 
        int expert_idx, 
        const NodePtr& node
    ){
        experts_[expert_idx][layer_idx]->node = node;
    }

private:
    std::vector<CallResult> Wait();
    void Start() { start_ = true; }
    void GPUFetchFunc(int gpu_id);
    void GPUExecFunc(int gpu_id);
    void OutputThreadFunc();
    void OutputFunc(ExecArgs args, torch::Tensor output, int gpu_id);
    std::vector<std::unique_ptr<base::Thread>> threads_;
    std::mutex mutex_;
    std::deque<CallArgs> input_queue_;
    std::deque<ExecArgs> exec_queue_;
    std::vector<CallResult> output_queue_;
    std::deque<OutputArgs> output_queue_internal_;
    std::vector<std::vector<ExpertNodePtr>> experts_;
    std::atomic<size_t> num_enqueued_;
    bool start_;
    int expert_type_;
    std::atomic<bool> main_thread_stop_flag_;
    std::atomic<size_t> pending_;
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::mutex input_mutex_;
    std::mutex exec_mutex_;
    std::condition_variable input_cv_;
    std::condition_variable exec_cv_;
    std::mutex output_mutex_;
    std::mutex output_queue_mutex_;
    std::condition_variable output_queue_cv_;
    std::vector<cudaStream_t> fetch_streams_;
    std::vector<cudaStream_t> exec_streams_;
    std::vector<cudaStream_t> out_streams_;
    torch::Tensor hidden_states_;
    torch::Tensor router_mask_;
    std::vector<int64_t> cache_sizes_;
    int cache_capacity_ = 0;
};


