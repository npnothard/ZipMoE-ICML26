// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once

#include <cstring>
#include <cstdlib>
#include <csignal>
#include <string>
#include <iostream>
#include <thread>
#include <unistd.h>


void bind_this_thread_to_core(int core_id);
extern std::string global_group_name;
void cleanup_existing_cpuset(const std::string& group_name);
void setup_exclusive_cpuset_core(
    const std::string& core_range
);
void cleanup_cpuset_wrapper();
void cleanup_exclusive_cpuset();
void signal_handler(int sig);
std::string num_threads_to_string(
    int num_threads
);
void exclusive_cpuset_exception_handler(
    int num_threads
);
