// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#include "memory_docks.hpp"

std::unique_ptr<MemoryDock> kMemoryDock = nullptr;

MemoryDock::MemoryDock(
    int num_nodes, 
    int num_tensors_per_node, 
    int num_file_chunks, 
    size_t shared_mem_size
): 
    num_docks_(num_nodes),
    num_tensors_per_node_(num_tensors_per_node),
    num_file_chunks_(num_file_chunks),
    shared_mem_size_(shared_mem_size) 
{
    size_t single_dock_tensor_size = num_tensors_per_node * shared_mem_size * sizeof(uint16_t);
    size_t single_dock_pinned_size = num_tensors_per_node * shared_mem_size;
    size_t total_tensor_size = num_nodes * single_dock_tensor_size;
    size_t total_pinned_size = num_nodes * single_dock_pinned_size;
    if (cudaMalloc(&global_tensor_pool_, total_tensor_size) != cudaSuccess) {
        throw std::runtime_error("[MemoryDock] cudaMalloc for global tensor pool failed");
    }
    if (cudaHostAlloc(&global_io_pinned_host_, total_pinned_size, cudaHostAllocMapped) != cudaSuccess) {
        cudaFree(global_tensor_pool_);
        throw std::runtime_error("[MemoryDock] cudaHostAlloc for IO pinned pool failed");
    }
    if (cudaHostGetDevicePointer(&global_io_pinned_device_, global_io_pinned_host_, 0) != cudaSuccess) {
        cudaFree(global_tensor_pool_);
        cudaFreeHost(global_io_pinned_host_);
        throw std::runtime_error("[MemoryDock] cudaHostGetDevicePointer for IO failed");
    }
    if (cudaHostAlloc(&global_exp_pinned_host_, total_pinned_size, cudaHostAllocMapped|cudaHostAllocWriteCombined) != cudaSuccess) {
        cudaFree(global_tensor_pool_);
        cudaFreeHost(global_io_pinned_host_);
        throw std::runtime_error("[MemoryDock] cudaHostAlloc for EXP pinned pool failed");
    }
    if (cudaHostGetDevicePointer(&global_exp_pinned_device_, global_exp_pinned_host_, 0) != cudaSuccess) {
        cudaFree(global_tensor_pool_);
        cudaFreeHost(global_io_pinned_host_);
        cudaFreeHost(global_exp_pinned_host_);
        throw std::runtime_error("[MemoryDock] cudaHostGetDevicePointer for EXP failed");
    }
    docks_.resize(num_nodes);
    free_dock_ids_.reserve(num_nodes);
    for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
        Dock& dock = docks_[node_idx];
        dock.tensor_offset = node_idx * single_dock_tensor_size;
        dock.io_pinned_offset = node_idx * single_dock_pinned_size;
        dock.exp_pinned_offset = node_idx * single_dock_pinned_size;
        dock.tensor_base = global_tensor_pool_;
        dock.io_pinned_host_base = global_io_pinned_host_;
        dock.io_pinned_device_base = global_io_pinned_device_;
        dock.exp_pinned_host_base = global_exp_pinned_host_;
        dock.exp_pinned_device_base = global_exp_pinned_device_;
        dock.single_buffer_size = shared_mem_size;
        dock.num_tensors = num_tensors_per_node;
        free_dock_ids_.push_back(node_idx);
    }
}

MemoryDock::~MemoryDock() {
    cudaDeviceSynchronize();

    if (global_tensor_pool_) {
        cudaFree(global_tensor_pool_);
    }
    if (global_io_pinned_host_) {
        cudaFreeHost(global_io_pinned_host_);
    }
    if (global_exp_pinned_host_) {
        cudaFreeHost(global_exp_pinned_host_);
    }
}

int MemoryDock::RegisterDockForNode(NodePtr node) {
    std::lock_guard<std::mutex> lock(dock_mutex_);

    if (free_dock_ids_.empty()) {
        throw std::runtime_error("[MemoryDock] No free docks available!");
    }
    
    if (node_to_dock_id_.find(node->id) != node_to_dock_id_.end()) {
        return node_to_dock_id_[node->id];
    }

    int dock_id = free_dock_ids_.back();
    free_dock_ids_.pop_back();

    node_to_dock_id_[node->id] = dock_id;

    return dock_id;
}

Dock* MemoryDock::GetDock(NodePtr node) {
    std::lock_guard<std::mutex> lock(dock_mutex_);
    
    auto it = node_to_dock_id_.find(node->id);
    if (it == node_to_dock_id_.end()) {
        throw std::runtime_error("[MemoryDock] Node Unregistered in Dock!");
    }
    return &docks_[it->second];
}


void MemoryDock::UnregisterDockForNode(NodePtr node) {
    std::lock_guard<std::mutex> lock(dock_mutex_);

    auto it = node_to_dock_id_.find(node->id);
    if (it == node_to_dock_id_.end()) {
        return; 
    }
    int dock_id = it->second;
    node_to_dock_id_.erase(it);
    free_dock_ids_.push_back(dock_id);
}