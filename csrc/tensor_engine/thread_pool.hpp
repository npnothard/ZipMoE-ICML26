// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once


#include <iostream>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>

#include "cpu_ctrl.hpp"
#include "tasks.hpp"

class PreemptiveThreadPool{
    private:
        std::priority_queue<
            std::shared_ptr<TaskBase>,
            std::vector<std::shared_ptr<TaskBase>>,
            TaskBase::Comparator
        > task_queue;
        std::mutex queue_mutex;
        std::mutex running_tasks_mutex;
        std::mutex map_mutex;
        std::condition_variable cond_var;
        std::vector<std::thread> workers;
        std::unordered_map< std::thread::id, std::shared_ptr<TaskBase> > running_tasks;
        std::atomic<bool> flag_stop;
        std::unordered_map<std::thread::id, int> thread_idx_map;
    public:
        int num_threads;
        bool bind_core;
        int idx_shift;
        PreemptiveThreadPool(
            int num_threads,
            int idx_shift,
            bool bind_core
        );
        ~PreemptiveThreadPool();
        void worker_loop(int idx);
        void add_task(std::shared_ptr<TaskBase> task);
        bool preempt_task(
            const std::string& task_id
        );
        int get_thread_idx();
        void stop();
};

