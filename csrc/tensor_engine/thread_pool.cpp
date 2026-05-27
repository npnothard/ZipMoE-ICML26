// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#include "thread_pool.hpp"


PreemptiveThreadPool::PreemptiveThreadPool(
    int num_threads,
    int idx_shift,
    bool bind_core
):
    flag_stop(false), idx_shift(idx_shift), bind_core(bind_core)
{
    for(int i = 0; i<num_threads; i++){
        workers.emplace_back( &PreemptiveThreadPool::worker_loop , this, i );
    }
}
PreemptiveThreadPool::~PreemptiveThreadPool(){
    stop();
}
void PreemptiveThreadPool::worker_loop(int idx){
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        thread_idx_map[std::this_thread::get_id()] = idx_shift + idx;
        std::cout<<"[Pool] Thread index "<<idx_shift+idx<<" initialized."<<std::endl;
    }
    if (bind_core){
        bind_this_thread_to_core( idx_shift + idx + 1);
    }
    while (true) {
        std::shared_ptr<TaskBase> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cond_var.wait(
                lock, 
                [&]{
                    return flag_stop.load() || !task_queue.empty();
                }
            );
            if (flag_stop.load()&&task_queue.empty()){
                return;
            }
            task = std::const_pointer_cast<TaskBase>(task_queue.top());
            task_queue.pop();
        }
        {
            std::lock_guard<std::mutex> lock(running_tasks_mutex);
            running_tasks[std::this_thread::get_id()] = task;
        }
        task->execute(idx);
        {
            std::lock_guard<std::mutex> lock(running_tasks_mutex);
            running_tasks.erase( std::this_thread::get_id() );
        }
    }
}
void PreemptiveThreadPool::add_task(std::shared_ptr<TaskBase> task){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        task_queue.push(task);
    }
    cond_var.notify_one();
}
bool PreemptiveThreadPool::preempt_task(
    const std::string& task_id
){
    std::lock_guard<std::mutex> lock(running_tasks_mutex);
    for (auto& [thread_id, task]: running_tasks){
        if (task->task_id == task_id) {
            std::cout<<"[ThreadPool] Preempting: "<<task_id<<std::endl;
            task->preempt_token->set_stop();
            return true;
        }
    }
    std::cout<<"[ThreadPool] There is no: "<<task_id<<" to preempt."<<std::endl;
    return false;
}

int PreemptiveThreadPool::get_thread_idx(){
    std::lock_guard<std::mutex> lock(map_mutex);
    auto it = thread_idx_map.find(std::this_thread::get_id());
    if (it == thread_idx_map.end()){
        throw std::runtime_error("[Threadpool] Invalid index.");
    }
    return it->second;
}

void PreemptiveThreadPool::stop()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        flag_stop.store(true);
    }
    cond_var.notify_all();
    for (auto& worker: workers){
        if (worker.joinable()){
            worker.join();
        }
    }
}



