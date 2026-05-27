// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "tasks.hpp"

PreemptToken::PreemptToken(): flag_stop(false) {}
bool PreemptToken::should_stop() const { 
    return flag_stop.load(); 
}
void PreemptToken::set_stop() { 
    flag_stop.store(true); 
}
void PreemptToken::reset() { 
    flag_stop.store(false); 
}
TaskBase::TaskBase(
    double priority,
    const std::string& task_id,
    bool discard
): 
    priority(priority),
    task_id(task_id),
    preempt_token(std::make_shared<PreemptToken>()),
    discard(discard) {}
bool TaskBase::Comparator::operator()(
        const std::shared_ptr<TaskBase>& a,
        const std::shared_ptr<TaskBase>& b
    ) const {
        return a->priority > b->priority;
}
        





