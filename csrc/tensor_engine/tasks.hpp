// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once
#include <atomic>
#include <future>
#include <memory>

class PreemptToken{
    private:
        std::atomic<bool> flag_stop;
    public:
        PreemptToken();
        bool should_stop() const;
        void set_stop();
        void reset();
};


class TaskBase{
    public:
        double priority;
        std::string task_id;
        std::shared_ptr<PreemptToken> preempt_token;
        bool discard;
        TaskBase(
            double priority,
            const std::string& task_id,
            bool discard
        );
        virtual ~TaskBase() = default;
        virtual void execute(int thread_idx) = 0;
        struct Comparator{
            bool operator()(
                const std::shared_ptr<TaskBase>& a,
                const std::shared_ptr<TaskBase>& b
            ) const;
        }; 
};


template <typename ResultType>
class TaskImpl : public TaskBase {
    public:
        using TaskFunc = std::function<ResultType(std::shared_ptr<PreemptToken>, int)>;
        std::shared_ptr<std::promise<ResultType>> promise;
        TaskFunc func;
        TaskImpl(
            double priority,
            const std::string& task_id,
            TaskFunc func,
            bool discard
        ):
            TaskBase(priority, task_id, discard),
            promise(std::make_shared<std::promise<ResultType>>()),
            func(std::move(func)) {}
        void execute(int thread_idx) override {
            try{
                if (preempt_token->should_stop()){
                    promise->set_value(ResultType{});
                    return;
                }
                promise->set_value(func(preempt_token, thread_idx));
            }catch(...){
                promise->set_exception(std::current_exception());
            }
        }
};


template <>
class TaskImpl<void> : public TaskBase{
    public:
        using TaskFunc = std::function<void(std::shared_ptr<PreemptToken>, int)>;
        std::shared_ptr<std::promise<void>> promise;
        TaskFunc func;
        TaskImpl(
            double priority,
            const std::string& task_id,
            TaskFunc func,
            bool discard
        ):
            TaskBase(priority, task_id, discard), 
            promise(std::make_shared<std::promise<void>>()),
            func(std::move(func)) {}
        void execute(int thread_idx) override {
            try{
                if (!func) {
                    throw std::runtime_error("[Task: " + task_id + "] func is null");
                }       
                if(preempt_token->should_stop()){
                    promise->set_value();
                    return;
                }
                func(preempt_token, thread_idx);
                promise->set_value();
            }catch(...){
                promise->set_exception(std::current_exception());
            }
        }
};


template<typename ResultType>
std::pair<std::shared_ptr<TaskBase>, std::future<ResultType>>
create_task(
    double priority,
    const std::string& task_id,
    std::function<ResultType(std::shared_ptr<PreemptToken>, int)> func,
    bool discard = false
){
    auto task_impl = std::make_shared<TaskImpl<ResultType>>(
        priority,
        task_id,
        std::move(func),
        discard
    );
    std::future<ResultType> future = task_impl->promise->get_future();
    return { task_impl , std::move(future) };
}





