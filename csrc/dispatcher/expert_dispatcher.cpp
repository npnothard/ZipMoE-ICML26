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



#include "expert_dispatcher.hpp"
#include "base/zipmoe_tensor_index.hpp"
#include "common/pytorch.hpp"
#include "common/time.hpp"
#include "prefetch/task_scheduler.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/logger.hpp"
#include "topology/model_topology.hpp"
#include "tensor_engine/tensor_engine.hpp"
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <future>

torch::Tensor ExpertDispatcher::ObtainTensor(
    int layer_id,
    int expert_id,
    int tensor_id
){
    NodePtr node = experts_[expert_id][layer_id]->node;
    CacheSlot* slot = kZipMoECacheHandle->GetCacheSlotSlot(node);
    return kTensorIndex->find(node->tensor_ids[tensor_id])->second.tensor;
}

ExpertDispatcher::ExpertDispatcher(
    int num_experts,
    int num_layers,
    int expert_type,
    int dtype
):
    pending_(0),
    num_enqueued_(0),
    start_(false),
    expert_type_(expert_type)
{
    at::InferenceMode infer_guard(0);
    for (int i = 0; i < num_experts; i++ ){
        experts_.emplace_back();
        for (int j = 0; j < num_layers; j++){
            experts_[i].emplace_back();
            experts_[i][j] = std::make_shared<ExpertNode>();
            experts_[i][j]->expert_type = expert_type;
            int expert_type = expert_type_;
            switch (expert_type) {
                case DEEPSEEK_MOE_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
                    break;
                case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
                    break;
                /*
                    Scalable: Can add more MoE support.
                */
                default:
                    DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ", expert_type);
            }
            experts_[i][j]->module->eval();
            experts_[i][j]->layer_idx = j;
            experts_[i][j]->expert_idx = i;
        }
    }
    StartThreads(1);
}

void ExpertDispatcher::StartThreads(int num_threads){
    DLOG_INFO("[Initialize] ExpertDispatcher::StartThreads, NumDevices", kNumDevices);
    main_thread_stop_flag_.store(false);
    for (int i = 0; i < kNumDevices; i++){
        cudaSetDevice(i);
        cudaStream_t fetch_stream;
        cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
        fetch_streams_.emplace_back(fetch_stream);
        cudaStream_t out_stream;
        cudaStreamCreateWithFlags(&out_stream, cudaStreamNonBlocking);
        fetch_streams_.emplace_back(out_stream);
        auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
        threads_.emplace_back( new base::Thread(thread_func) );
        threads_.back()->start();
        DLOG_INFO("[Initialize] GPUFetchFunc Started.");
        auto cache_limit = kTopologyHandle->GetSparseCacheLimit( torch::Device(torch::kCUDA, i) );
        cache_sizes_.push_back(cache_limit);
    }
    for (int i = 0; i < kNumDevices*num_threads; i++){
        cudaSetDevice( i%kNumDevices );
        cudaStream_t exec_stream;
        cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
        exec_streams_.emplace_back(exec_stream);
        auto thread_func = std::bind(&ExpertDispatcher::GPUExecFunc, this, i%kNumDevices);
        threads_.emplace_back( new base::Thread(thread_func) );
        threads_.back()->start();        
        DLOG_INFO("[Initialize] GPUExecFunc Started.");
    }
    auto thread_func = std::bind(&ExpertDispatcher::OutputThreadFunc, this);
    threads_.emplace_back(new base::Thread(thread_func));
    threads_.back()->start();
    DLOG_INFO("[Initialize] OutputThreadFunc Started.");

}


void ExpertDispatcher::RegisterExpert(
    int layer_idx, 
    int expert_idx,
    const std::vector<std::uint32_t>& tensor_ids
){
    NodePtr cached_node = nullptr;
    for (auto tensor_id : tensor_ids) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        if (cached_node == nullptr) {
            cached_node = node;
            experts_[expert_idx][layer_idx]->node = node;
        } else if (cached_node != node) {
            DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
        }
    }
}


void ExpertDispatcher::ClearExpertCacheCounts() {
    for (auto& expert : experts_) {
        for (auto& expert_node : expert) {
            if (expert_node->node == nullptr) {
                continue;
            }
            expert_node->node->incache_visit_count = 0;
        }
    }
}


void ExpertDispatcher::EnqueueExpert(
    int layer_idx,
    int expert_idx,
    int num_token_reqs
){
    auto expert_node = experts_[expert_idx][layer_idx];
    TaskState task_state = kZipMoECacheHandle->ProactiveDecision(layer_idx, expert_idx, num_token_reqs);
    num_enqueued_.fetch_add(1);

    {
        std::lock_guard<std::mutex> lock(kZipMoETensorEngine->scheduled_queue_mutex);
        for ( auto& existing_task: kZipMoETensorEngine->scheduled_task_queue ){
            if (
                existing_task->layer_idx == layer_idx &&
                existing_task->expert_idx == expert_idx
            ){
                existing_task->state = TaskState::NORMAL;
                existing_task->prediction_hit = true;
                existing_task->num_token_reqs = num_token_reqs;
                kZipMoECacheHandle->UpdateOnHit(existing_task->node, num_token_reqs);
                return;
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(kZipMoETensorEngine->ready_queue_mutex);
        for ( auto& ready_task: kZipMoETensorEngine->ready_queue ){
            if (
                ready_task->layer_idx == layer_idx &&
                ready_task->expert_idx == expert_idx
            ){
                ready_task->state = TaskState::NORMAL;
                ready_task->prediction_hit = true;
                ready_task->num_token_reqs = num_token_reqs;
                kZipMoECacheHandle->UpdateOnHit(ready_task->node, num_token_reqs);
                return; 
            }
        }
    }
    if (expert_node->node->device.is_cuda()){
        kZipMoECacheHandle->UpdateOnHit(expert_node->node, num_token_reqs);
    }
    expert_node->node->mutex.lock();
    std::lock_guard<std::mutex> lock(kZipMoETensorEngine->new_queue_mutex);
    ZipMoETaskPtr task = std::make_shared<ZipMoETask>();
    task->priority = 0.0;
    task->node = expert_node->node;
    task->layer_idx = layer_idx;
    task->expert_idx = expert_idx;
    task->num_token_reqs = num_token_reqs;
    task->state = task_state;
    task->prediction_hit = true;
    task->preempt_token = std::make_shared<PreemptToken>(); 
    int total_chunks = kZipMoETensorEngine->config_ptr->num_file_chunks * expert_node->node->tensor_ids.size();
    int total_ios = (task->node->device_sm_ptr != nullptr) ? 0 : expert_node->node->tensor_ids.size();
    task->num_pending_operations.store(total_chunks+total_ios, std::memory_order_release);
    if (task->node->device.is_cuda()){
        {
            std::lock_guard<std::mutex> lock(kZipMoETensorEngine->ready_queue_mutex);
            assert(task->node->device_sm_ptr == nullptr);
            task->state = TaskState::NORMAL;
            kZipMoETensorEngine->ready_queue.push_back(std::move(task));
        }
        kZipMoETensorEngine->ready_queue_cv.notify_all();
    } else {
        kZipMoETensorEngine->new_task_queue.insert(std::move(task));
    }
    
}


void ExpertDispatcher::EnqueuePrefetch(
    int target_layer_idx,
    int predicted_expert_idx
){
    auto expert_node = experts_[predicted_expert_idx][target_layer_idx];
    expert_node->node->mutex.lock();
    std::lock_guard<std::mutex> lock(kZipMoETensorEngine->new_queue_mutex);
    ZipMoETaskPtr task = std::make_shared<ZipMoETask>();
    task->priority = 0.0;
    task->node = expert_node->node;
    task->layer_idx = target_layer_idx;
    task->expert_idx = predicted_expert_idx;
    task->num_token_reqs = 0;
    task->state = TaskState::PREFETCH;
    task->preempt_token = std::make_shared<PreemptToken>();
    int total_chunks = kZipMoETensorEngine->config_ptr->num_file_chunks * expert_node->node->tensor_ids.size();
    int total_ios = (task->node->device_sm_ptr != nullptr) ? 0 : expert_node->node->tensor_ids.size();
    task->num_pending_operations.store(total_chunks+total_ios, std::memory_order_release);
    if (task->node->device.is_cuda()){
        {
            std::lock_guard<std::mutex> lock(kZipMoETensorEngine->ready_queue_mutex);
            kZipMoETensorEngine->ready_queue.push_back(std::move(task));
        }
        kZipMoETensorEngine->ready_queue_cv.notify_all();
    } else {
        kZipMoETensorEngine->new_task_queue.insert(std::move(task));
    }
}


void ExpertDispatcher::SubmitPrefetch(
    int target_layer_idx,
    int predicted_expert_idx,
    double priority
){
    auto expert_node = experts_[predicted_expert_idx][target_layer_idx];
    expert_node->node->mutex.lock();
    ZipMoETaskPtr task = std::make_shared<ZipMoETask>();
    task->priority = priority;
    task->node = expert_node->node;
    task->layer_idx = target_layer_idx;
    task->expert_idx = predicted_expert_idx;
    task->num_token_reqs = 0;
    task->state = TaskState::PREFETCH;
    task->preempt_token = std::make_shared<PreemptToken>();
    int total_chunks = kZipMoETensorEngine->config_ptr->num_file_chunks * expert_node->node->tensor_ids.size();
    int total_ios = (task->node->device_sm_ptr != nullptr) ? 0 : expert_node->node->tensor_ids.size();
    task->num_pending_operations.store(total_chunks+total_ios, std::memory_order_release);
    if (task->node->device.is_cuda()){
        {
            std::lock_guard<std::mutex> lock(kZipMoETensorEngine->ready_queue_mutex);
            kZipMoETensorEngine->ready_queue.push_back(std::move(task));
        }
        kZipMoETensorEngine->ready_queue_cv.notify_all();
    } else {
        kZipMoETensorEngine->ToWorkers(task);
    }
}


void ExpertDispatcher::OperationSchedule(){
    kZipMoETensorEngine->ZipMoESchedulerFunc();
}


void ExpertDispatcher::GPUFetchFunc(int gpu_id){
    DLOG_INFO("[Initialize] Thread started for GPU: ", gpu_id);
    while (!main_thread_stop_flag_.load()){
        std::unique_lock<std::mutex> lock(kZipMoETensorEngine->ready_queue_mutex);
        kZipMoETensorEngine->ready_queue_cv.wait(
            lock, 
            [&]{
                bool not_empty = !kZipMoETensorEngine->ready_queue.empty();
                return not_empty;
            }
        );
        ZipMoETaskPtr task = std::move( kZipMoETensorEngine->ready_queue.front() );
        kZipMoETensorEngine->ready_queue.pop_front();
        auto expert_node = experts_[task->expert_idx][task->layer_idx];
        task->node->SetDevice(
            CUDA_DEVICE(gpu_id),
            task->exp_ptr,
            task->sm_ptr,
            task->tensor_ptr,
            kZipMoETensorEngine->tensor_stream
        );
        expert_node->SetTensorsFromBlob(CUDA_DEVICE(gpu_id));
        if (task->state == TaskState::SMCACHE){
            task->node->SetSM(task->sm_ptr);
        }
        if (
            (task->node->device_sm_ptr!=nullptr) &&
            ( task->state == TaskState::NORMAL || task->state == TaskState::PREFETCH)
        ){
            kZipMoECacheHandle->UnregisterSMCacheForNode(task->node);
        }

        if (task->state == TaskState::PREFETCH){
            kMemoryDock->UnregisterDockForNode(task->node);
            task->node->mutex.unlock();
        } 
        
        lock.unlock();        
        
        if (task->state != TaskState::PREFETCH) {
            torch::Tensor input;
            auto token_indices = router_mask_.index({"...", task->expert_idx}).to(torch::kBool);
            switch (expert_type_){
                case DEEPSEEK_MOE_DENSE_ACT_DENSE:
                case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                    input = hidden_states_.index({token_indices}).to(task->node->device);
                    break;
                default:
                    DLOG_FATAL(
                        "ExpertDispatcher::expert_type_ is unknown: ", expert_type_
                    );

            }
            ExecArgs exec_args;
            exec_args.hidden_states = std::move(input);
            exec_args.expert_node = expert_node;
            exec_args.out_gpu_id = hidden_states_.device().index();
            exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
            exec_args.task_state = task->state;
            exec_args.hit = task->node->device.is_cuda();
            std::lock_guard<std::mutex> exec_lock(exec_mutex_);
            exec_queue_.emplace_back(std::move(exec_args));
            exec_cv_.notify_all();
        }
    }
}



void ExpertDispatcher::GPUExecFunc(int gpu_id){
    cudaSetDevice(gpu_id);
    while ( !main_thread_stop_flag_.load() ){
        std::unique_lock<std::mutex> lock(exec_mutex_);
        exec_cv_.wait( lock, [&]{ return !exec_queue_.empty(); } );
        ExecArgs args = std::move( exec_queue_.front() );
        exec_queue_.pop_front();
        lock.unlock();
        if (args.expert_node == nullptr){ continue; }
        torch::Tensor output;
        at::InferenceMode infer_guard(true);
        c10::cuda::CUDAStream stream = 
            c10::cuda::getStreamFromExternal(
                kZipMoETensorEngine->tensor_stream,
                gpu_id
            );

        {
            auto start = TIME_NOW;
            c10::cuda::CUDAStreamGuard stream_guard(stream); 
            auto* expert_module = args.expert_node->module;
            int expert_type = expert_type_;
            cudaStreamSynchronize(stream);
            try{
                nvtxRangePushA("ExpertForward");
                switch (expert_type){
                    case DEEPSEEK_MOE_DENSE_ACT_DENSE:
                        output = reinterpret_cast<DeepSeekMoEDenseActDense*>(expert_module)->forward(
                            args.hidden_states
                        );
                        break;
                    case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                        output = reinterpret_cast<SwitchTransformersDenseActDense*>(expert_module)->forward(
                            args.hidden_states
                        );
                        break;            
                    default:
                        DLOG_FATAL("ExpertDispatcher::GPUExecFunc: unknown expert type", expert_type);

                }
                nvtxRangePop();

            } catch(const std::exception& e) {
                std::stringstream ss;
                ss << "DenseActDense tensor_ids: [";
                for (auto& id : args.expert_node->node->tensor_ids) {
                    ss << id << " ";
                }
                ss << "]";
                DLOG_FATAL(
                    "ExpertDispatcher::GPUExecFunc", 
                    ss.str(), 
                    "expert_type",
                    expert_type, 
                    e.what()
                );
            }

            stream.synchronize();
            auto end = TIME_NOW;
        }
        {
            std::lock_guard<std::mutex> lock(output_queue_mutex_);
            output_queue_internal_.emplace_back(
                OutputArgs{
                    std::move(args),
                    std::move(output),
                    gpu_id
                }
            );
            output_queue_cv_.notify_all();
        }
    }
}




void ExpertDispatcher::OutputThreadFunc(){
    while (!main_thread_stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(output_queue_mutex_);
        
        output_queue_cv_.wait(lock, [&] {
            return !output_queue_internal_.empty() || 
                    main_thread_stop_flag_.load();
        });
        
        if (main_thread_stop_flag_.load() && output_queue_internal_.empty()) {
            break;
        }
        
        OutputArgs args = std::move(output_queue_internal_.front());
        output_queue_internal_.pop_front();
        lock.unlock();
        
        OutputFunc(
            std::move(args.exec_args),
            std::move(args.output),
            args.gpu_id
        );
    }
}


void ExpertDispatcher::OutputFunc(
    ExecArgs args,
    torch::Tensor output,
    int gpu_id
){
    auto output_device = 
        (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
    torch::Tensor output_tensor = output.to(output_device).to(args.out_dtype);

    // Post-process for the node.
    switch (args.task_state){
        case TaskState::NORMAL:
        case TaskState::PREFETCH:
            break;
        case TaskState::SMCACHE:
            args.expert_node->node->SetDevice(
                DISK_DEVICE, nullptr,nullptr, nullptr, nullptr
            );
            break;
        case TaskState::EVICT:
            kZipMoECacheHandle->UnregisterCacheSlotForNode(args.expert_node->node);
            kZipMoECacheHandle->UnregisterSMCacheForNode(args.expert_node->node);
            break;
    }


    kMemoryDock->UnregisterDockForNode(args.expert_node->node);
    args.expert_node->node->mutex.unlock();
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_queue_.emplace_back(
            std::move(output_tensor),
            args.expert_node->layer_idx,
            args.expert_node->expert_idx,
            args.hit
        );
    }
    pending_.fetch_sub(1);
    if (pending_.load() == 0) {
        pending_cv_.notify_all();
    }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
    int wait_count = 0;

    std::unique_lock<std::mutex> lock(pending_mutex_);
    pending_cv_.wait(lock, [&] { return pending_.load() == 0; });

    num_enqueued_.store(0);
    std::vector<CallResult> output_queue;
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_queue.swap(output_queue_);
    }

    return output_queue;
}