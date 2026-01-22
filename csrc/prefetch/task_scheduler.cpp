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


#include <sstream>
#include <list>

#include "common/time.hpp"
#include "task_scheduler.hpp"
#include "utils/logger.hpp"
#include "utils/cuda_utils.hpp"

std::unique_ptr<ZipMoETaskPool> kTaskPool = nullptr;

ZipMoETaskPool::ZipMoETaskPool(){
    unified_queue_.resize(NUM_PRIORITY);
    main_thread_stop_flag_.store(false);
    int num_gpu = GetDeviceCount();
    int num_thread_per_gpu = 1;
    for (int i = -1; i < num_gpu; ++i){
        gpu_min_priority_.push_back(
            std::vector<std::uint32_t>(num_thread_per_gpu, 1000));
    }
    for (int i = 0; i < num_gpu; ++i){
        std::list<std::thread> gpu_threads;
        for (int j = 0; j < num_thread_per_gpu; ++j ){
            auto gpu_thread = std::thread(&ZipMoETaskPool::GPUThreadFunc, this, i, j);
            gpu_threads.push_back(std::move(gpu_thread));
        }
        exec_threads_.push_back(std::move(gpu_threads));
    }
}

void ZipMoETaskPool::GPUThreadFunc(int gpu_id, int thread_id) {
    while (true) {
        TaskPtr task = nullptr;
        { 
            std::unique_lock<std::mutex> lock(unified_mutex_);
            cv_task_.wait(lock, [&]() {
                if (main_thread_stop_flag_.load()) return true;
                for (uint32_t i = 0; i < NUM_PRIORITY; ++i) {
                    for (const auto& t : unified_queue_[i]) {
                        if (t->dst_device.index() == gpu_id) {
                            return true;
                        }
                    }
                }
                return false;
            });
            if (main_thread_stop_flag_.load()) {
                break; 
            }
            uint32_t max_priority = 1000;
            for (uint32_t i = 0; i < NUM_PRIORITY; ++i) {
                for (const auto& t : unified_queue_[i]) {
                     if (t->dst_device.index() == gpu_id) {
                         max_priority = i;
                         task = t;
                         break;
                     }
                }
                if (task != nullptr) break;
            }
            if (task == nullptr) {
                continue; 
            }
            auto node = task->node;
            node->incache_visit_count += 1;
            for (uint32_t i = 0; i < NUM_PRIORITY; ++i) {
                unified_queue_[i].erase(
                    std::remove_if(
                        unified_queue_[i].begin(),
                        unified_queue_[i].end(),
                        [&, task](auto& t) {
                            return (t->node == node) && (t->dst_device == task->dst_device);
                        }
                    ),
                    unified_queue_[i].end()
                );
            }
        } 

        DLOG_TRACE(("Execute task " + task->DebugString()).c_str());

        SetNodeDevice(task);

        auto node = task->node;
        if (task->on_demand) {
            node->state = 0;
            node->cv.notify_all();
        }
    }
}


void ZipMoETaskPool::SetNodeDevice(const TaskPtr& task){
    auto node = task->node;

    DLOG_TRACE("SetNodeDevice: task: {}, node: {}", task->DebugString(),
                node->str());
    if (!task->on_demand) {
        if (!node->mutex.try_lock()) {
        DLOG_TRACE("SetNodeDevice: task: {}, mutex locked", task->DebugString());
        return;
        }
    }
    if (node->device.type() == task->dst_device.type()) {
        DLOG_TRACE("SetNodeDevice: task: {}, skip same device",
                task->DebugString());
        if (!task->on_demand) node->mutex.unlock();
        return;
    }
    node->PinDense(
        task->dst_device,
        node->dense_gpu_offset_ptr
    );
    if (!task->on_demand) node->mutex.unlock();
    node->io_state = NODE_STATE_CACHED;
}

void ZipMoETaskPool::EnqueueTask( const TaskPtr& task ){
    DLOG_TRACE("EnqueueTask: {}", task->DebugString());
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (size_t i = 1; i < NUM_PRIORITY; ++i){
            unified_queue_[i].erase(
                std::remove_if(
                    unified_queue_[i].begin(),
                    unified_queue_[i].end(),
                    [&](auto& t){
                        bool is_same_node = (t->node == task->node);
                        bool is_lower_priority = (t->priority >= task->priority);
                        bool is_outdate_layers = task->remove_layer && (
                            (t->node->corr_id & 0xFFFFFFFF) < (task->node->corr_id & 0xFFFFFFFF)
                        );
                        bool need_remove = (is_same_node&&is_lower_priority)|| is_outdate_layers;
                        if (need_remove){
                            t->node->mutex.unlock(); 
                        }
                        return need_remove;
                    }
                ),
                unified_queue_[i].end()
            );
        }
    }
    if (task->src_device == task->dst_device){
        task->node->state = 0;
        task->node->cv.notify_all();
        DLOG_TRACE("EnqueueTask: {} is on the same device", task->DebugString());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
        DLOG_TRACE("EnqueueTask: finish {}", task->DebugString());
    }
    cv_task_.notify_all();
}

void ZipMoETaskPool::FetchExec(
    const uint64_t& request_id,
    const NodePtr& node
){
    auto task = std::make_shared<Task>();
    task->on_demand = false;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_device;
    task->request_id = request_id;
    DLOG_TRACE("FetchExec: {}", task->DebugString());
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (size_t i = 1; i < NUM_PRIORITY; ++i){
            unified_queue_[i].erase(
                std::remove_if(
                    unified_queue_[i].begin(),
                    unified_queue_[i].end(),
                    [&](auto& t){
                        return (t->node == node) || 
                               ((node->corr_id & 0xFFFFFFFF)>=(t->node->corr_id & 0xFFFFFFFF));
                    }
                ),
                unified_queue_[i].end()
            );
        }
    }
    if (task->src_device == task->dst_device){
        node->state = 0;
        node->cv.notify_all();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }
    cv_task_.notify_all();
}

void ZipMoETaskPool::StartExec(
    const uint64_t& request_id,
    const NodePtr& node
){
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_device;
    task->request_id = request_id;
    DLOG_TRACE("StartExec: {}", task->DebugString());
    node->visit_count += 1;
    if (node->device.is_cuda()){
        node->incache_visit_count++;
    }
    node->last_access_time = MICROSECONDS_SINCE_EPOCH;
    node->io_state = static_cast<NodeState>(node->io_state | NODE_STATE_VISITED);
    node->last_prefetch_time = MICROSECONDS_SINCE_EPOCH;
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (size_t i = 0; i < NUM_PRIORITY; ++i){
            unified_queue_[i].erase(
                std::remove_if(
                    unified_queue_[i].begin(),
                    unified_queue_[i].end(),
                    [&](auto& t){
                        return (t->node == node) || 
                               ((node->corr_id & 0xFFFFFFFF)>(t->node->corr_id & 0xFFFFFFFF));
                    }
                ),
                unified_queue_[i].end()
            );            
        }
    }
    if (task->src_device.is_cuda()){
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.insert({node->id, task});
        node->state.store(0);
        node->cv.notify_all();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        if (exec_queue_.find(node->id) != exec_queue_.end()) {
            std::stringstream ss;
            ss << "Node " << std::hex << node->id << " is already in exec queue";
            DLOG_WARN(ss.str().c_str());
            node->state = 0;
            node->cv.notify_all();
            return;
        }        
    }
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }
    cv_task_.notify_all();
}

void ZipMoETaskPool::StopExec(
    const std::uint64_t& request_id,
    const NodePtr& node
){
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_host;
    task->request_id = request_id;

    DLOG_TRACE("StopExec: {}", task->DebugString());

    node->state = 0;
    node->cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.erase(node->id);
    }

    return;
}
