// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>
#include "topology/model_topology.hpp"

struct Dock {
    size_t tensor_offset = 0;  
    size_t io_pinned_offset = 0;  
    size_t exp_pinned_offset = 0;  
    uint16_t* tensor_base = nullptr;
    uint8_t* io_pinned_host_base = nullptr;
    uint8_t* io_pinned_device_base = nullptr;
    uint8_t* exp_pinned_host_base = nullptr;
    uint8_t* exp_pinned_device_base = nullptr;
    size_t single_buffer_size = 0;
    int num_tensors = 0;
    inline uint16_t* get_tensor() const {
        return tensor_base + tensor_offset / sizeof(uint16_t);
    }
    inline std::pair<uint8_t*, uint8_t*> get_sm_pinned(int tensor_idx) const {
        size_t offset = io_pinned_offset + tensor_idx * single_buffer_size;
        return {io_pinned_host_base + offset, io_pinned_device_base + offset};
    }
    inline std::pair<uint8_t*, uint8_t*> get_exp_pinned(int tensor_idx) const {
        size_t offset = exp_pinned_offset + tensor_idx * single_buffer_size;
        return {exp_pinned_host_base + offset, exp_pinned_device_base + offset};
    }
};

class MemoryDock {
public:
    MemoryDock(int num_nodes, int num_tensors_per_node, int num_file_chunks, size_t shared_mem_size);
    ~MemoryDock();
    int RegisterDockForNode(NodePtr node);
    Dock* GetDock(NodePtr node);
    void UnregisterDockForNode(NodePtr node);
private:
    int num_docks_;
    int num_tensors_per_node_;
    int num_file_chunks_;
    size_t shared_mem_size_;
    uint16_t* global_tensor_pool_ = nullptr;          
    uint8_t* global_io_pinned_host_ = nullptr;       
    uint8_t* global_io_pinned_device_ = nullptr;   
    uint8_t* global_exp_pinned_host_ = nullptr;  
    uint8_t* global_exp_pinned_device_ = nullptr;  
    std::vector<Dock> docks_;
    std::vector<int> free_dock_ids_;
    std::unordered_map<size_t, int> node_to_dock_id_;
    std::mutex dock_mutex_;
};

extern std::unique_ptr<MemoryDock> kMemoryDock;
