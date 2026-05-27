// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
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

#include <deque>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <condition_variable>

#include "base/noncopyable.hpp"
#include "common/pytorch.hpp"
#include "topology/model_topology.hpp"

#define SKIP_TO_NEXT_ITERATION                                \
  std::this_thread::sleep_for(std::chrono::microseconds(10)); \
  continue;

#define NUM_PRIORITY 20UL

struct Task{
    bool on_demand = false;
    NodePtr node;
    std::vector<NodePtr> remove_nodes;
    uint32_t priority;
    uint64_t request_id;
    torch::Device src_device = DISK_DEVICE;
    torch::Device dst_device = DISK_DEVICE;
    cudaStream_t stream = nullptr;

    bool remove_layer = false;

    std::string DebugString() {
        std::stringstream ss;
        ss << "Task: node: " << node->str() << ", on_demand: " << on_demand
        << ", priority: " << priority << "[" << src_device.str() << "->"
        << dst_device.str() << "]";
        return ss.str();
    }
};
typedef std::shared_ptr<Task> TaskPtr;

class ZipMoETaskPool: public base::noncopyable{

public:
    DELETE_COPY_AND_ASSIGN(ZipMoETaskPool);
    STATIC_GET_INSTANCE(ZipMoETaskPool);
    ZipMoETaskPool();
    ~ZipMoETaskPool(){
        std::cout << "ZipMoETaskPool destructor" << std::endl;
        main_thread_stop_flag_.store(true);
        cv_task_.notify_all();
        // wait for all threads to stop
        for (auto& thread_list : exec_threads_) {
            for (auto& thread : thread_list) {
                thread.join();
            }
        }
    }

    void StartExec(const uint64_t& request_id, const NodePtr& node);
    void FetchExec(const uint64_t& request_id, const NodePtr& node);
    void StopExec(const uint64_t& request_id, const NodePtr& node);
    void EnqueueTask(const TaskPtr& task);

    void ClearQueue(){
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (uint32_t priority = 1; priority < NUM_PRIORITY; priority++){
            unified_queue_[priority].clear();
        }
    }

    void ReplaceCacheCandidates(const NodePtrList& candidates){
        std::lock_guard<std::mutex> lock(unified_mutex_);
        {
            std::lock_guard<std::mutex> lock(candidates_mutex_);
            candidates_.clear();
            for (auto& node: candidates){
                candidates_.insert(node);
            }
        }
        for (uint32_t priority = 1; priority < NUM_PRIORITY; priority++){
            unified_queue_[priority].clear();
        }
    }


private:
    std::vector<std::deque<TaskPtr>> unified_queue_; 
    std::unordered_map<uint64_t, TaskPtr> exec_queue_;
    std::unordered_set<NodePtr> candidates_;
    std::mutex exec_mutex_;
    std::mutex unified_mutex_;
    std::mutex candidates_mutex_;
    std::condition_variable cv_task_;

    std::vector<std::list<std::thread>> exec_threads_;
    std::atomic<bool> main_thread_stop_flag_;

    std::vector<std::vector<uint32_t>> gpu_min_priority_;

    
    void GPUThreadFunc(int gpu_id, int thread_id);
    void SetNodeDevice(const TaskPtr& task);
    std::string DebugString(const std::vector<std::deque<TaskPtr>>& queue);
};
extern std::unique_ptr<ZipMoETaskPool> kTaskPool;




























