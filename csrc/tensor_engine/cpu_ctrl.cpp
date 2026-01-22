// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "cpu_ctrl.hpp"


std::string global_group_name = "zipmoe_pool";

void bind_this_thread_to_core(int core_id){
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t thread = pthread_self();
    int bind_result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (bind_result != 0){
        std::cerr<<"[POSIX Thread]: Failed to bind to core "<<core_id<<": "<<strerror(bind_result)<<std::endl;
    } else {
        std::cout<<"[POSIX Thread]: Thread bound to core: "<<core_id<<std::endl;
    }
}


void cleanup_existing_cpuset(const std::string& group_name) {
    std::string group_path = "/sys/fs/cgroup/cpuset/" + group_name;
    std::string check_cmd = "[ -d " + group_path + " ]";
    if (system(check_cmd.c_str()) == 0) {
        std::cout << "[Cpuset] Found existing cpuset " << group_name << ", cleaning up..." << std::endl;
        std::string cleanup_cmd = 
            "sudo bash -c '"
            "if [ -f " + group_path + "/tasks ]; then "
            "  for pid in $(cat " + group_path + "/tasks 2>/dev/null); do "
            "    echo $pid > /sys/fs/cgroup/cpuset/tasks 2>/dev/null || true; "
            "  done; "
            "fi; "
            "echo 0 > " + group_path + "/cpuset.cpu_exclusive 2>/dev/null || true; "
            "rmdir " + group_path + " 2>/dev/null || true'";
        
        int ret = system(cleanup_cmd.c_str());
    }
}


void setup_exclusive_cpuset_core(
    const std::string& core_range
){
    pid_t pid = getpid();
    std::string group = global_group_name;

    cleanup_existing_cpuset(group);

    std::string cmd = 
        "sudo mkdir -p /sys/fs/cgroup/cpuset/" + group + " && " +
        "echo " + core_range + " | sudo tee /sys/fs/cgroup/cpuset/" + group + "/cpuset.cpus && " +
        "echo 0 | sudo tee /sys/fs/cgroup/cpuset/" + group + "/cpuset.mems && " +
        "echo 1 | sudo tee /sys/fs/cgroup/cpuset/" + group + "/cpuset.cpu_exclusive && " +
        "echo " + std::to_string(pid) + " | sudo tee /sys/fs/cgroup/cpuset/" + group + "/tasks";
    
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[Cpuset] Failed to set cpuset with cores " << core_range 
                  << ". Are you root or using sudo without password?\n";
    } else {
        std::cout << "[Cpuset] PID " << pid << " successfully bound to cores " << core_range << "\n";
    }
}

void cleanup_cpuset_wrapper() {
    std::string group_path = "/sys/fs/cgroup/cpuset/" + global_group_name;
    std::string move_tasks_cmd = 
        "if [ -f " + group_path + "/tasks ]; then "
        "sudo cat " + group_path + "/tasks | while read pid; do "
        "echo $pid | sudo tee /sys/fs/cgroup/cpuset/tasks >/dev/null 2>&1; "
        "done; fi";
    std::string remove_cmd = "sudo rmdir " + group_path;

    std::string full_cmd = move_tasks_cmd + " && " + remove_cmd;
    
    int ret = system(full_cmd.c_str());
    if (ret != 0) {
        std::cerr << "[Cleanup] Warning: Failed to remove cpuset group " << global_group_name << std::endl;
        std::cerr << "[Cleanup] You may need to manually run: sudo rmdir " << group_path << std::endl;
    } else {
        std::cout << "[Cleanup] Successfully removed cpuset group " << global_group_name << std::endl;
    }
}

void cleanup_exclusive_cpuset() {

    std::string group_path = "/sys/fs/cgroup/cpuset/" + global_group_name;
    std::string move_tasks_cmd = 
        "if [ -f " + group_path + "/tasks ]; then "
        "sudo cat " + group_path + "/tasks | while read pid; do "
        "echo $pid | sudo tee /sys/fs/cgroup/cpuset/tasks >/dev/null 2>&1; "
        "done; fi";
    std::string remove_cmd = "sudo rmdir " + group_path;
    std::string full_cmd = move_tasks_cmd + " && " + remove_cmd;
    int ret = system(full_cmd.c_str());
    if (ret != 0) {
        std::cerr << "[Cleanup] Warning: Failed to remove cpuset group " << global_group_name << std::endl;
        std::cerr << "[Cleanup] You may need to manually run: sudo rmdir " << group_path << std::endl;
    } else {
        std::cout << "[Cleanup] Successfully removed cpuset group " << global_group_name << std::endl;
    }
}

void signal_handler(int sig) {
    std::cout << "\n[Signal] Caught signal " << sig << ", cleaning up..." << std::endl;
    cleanup_cpuset_wrapper();
    exit(sig);
}


std::string num_threads_to_string(
    int num_threads
){
    return "0-"+std::to_string(num_threads);
}

void exclusive_cpuset_exception_handler(
    int num_threads
){
    std::string core_range = num_threads_to_string(num_threads);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    setup_exclusive_cpuset_core(core_range);
}
