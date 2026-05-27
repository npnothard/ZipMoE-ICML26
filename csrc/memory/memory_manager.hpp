// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "base/noncopyable.hpp"
#include "topology/model_topology.hpp"
#include "tensor_engine/zipmoe_io_handle.hpp"
#include "utils/logger.hpp"
#include "utils/cuda_utils.hpp"
#include "prefetch/task_scheduler.hpp"



struct CacheSlot {
    uint16_t* ptr;
    size_t id;
    size_t size;
    NodePtr node;
    CacheSlot(): ptr(nullptr), id(0), size(0), node(nullptr) {}
    bool is_free() const { return node == nullptr; }
    void reset() { node = nullptr; }
};

struct SMSlot {
    std::pair<uint8_t*,uint8_t*> ptr;
    size_t id;
    size_t size;
    NodePtr node;
    SMSlot(): ptr{nullptr, nullptr}, id(0), size(0), node(nullptr) {}
    bool is_free() const { return node == nullptr; }
    void reset() { node = nullptr; }
};



class ZipMoEGPUMemoryPool {
public:
    ZipMoEGPUMemoryPool(
        size_t pool_size,
        size_t slot_size,
        int pool_id,
        uint16_t* handover_ptr
    );
    ~ZipMoEGPUMemoryPool();
    void place(NodePtr& node, size_t slot_id); 
    void evict(size_t slot_id);
    bool contains(const NodePtr& node) const;
    bool has_free_slot() const;
    size_t get_free_slot() const;
    size_t pop_free_slot();
    CacheSlot* get_slot(size_t slot_id);
    const std::vector<CacheSlot>& get_all_slots() const;
    size_t get_slot_id_for_node( size_t node_id ) const;
    size_t num_slots() const;
    size_t num_free_slots() const;
    void print_status() const;
private:

    uint16_t* base_ptr_;
    size_t pool_size_;
    size_t slot_size_;
    size_t num_slots_;
    int pool_id_;
    std::unordered_map<size_t,size_t> node_to_slot_; 
    std::vector<CacheSlot> slots_;
    std::vector<size_t> free_slots_;
    mutable std::mutex pool_mutex_;
    void init_slots();
    void destroy_pool();
};

class ZipMoEPinnedMemoryPool {
public:
    ZipMoEPinnedMemoryPool(
        size_t pool_size,
        size_t slot_size,
        int pool_id,
        uint8_t* host_handover_ptr,
        uint8_t* device_handover_ptr
    );
    ~ZipMoEPinnedMemoryPool();
    void place(NodePtr& node, size_t slot_id); 
    void evict(size_t slot_id);
    bool contains(const NodePtr& node) const;
    bool has_free_slot() const; 
    size_t get_free_slot() const;
    size_t pop_free_slot();
    SMSlot* get_slot(size_t slot_id);
    const std::vector<SMSlot>& get_all_slots() const;
    size_t get_slot_id_for_node( size_t node_id ) const;
    size_t num_slots() const;
    size_t num_free_slots() const;
    void print_status() const;

private:

    uint8_t* host_base_ptr_;
    uint8_t* device_base_ptr_;
    size_t pool_size_;
    size_t slot_size_;
    size_t num_slots_;
    int pool_id_;
    std::unordered_map<size_t,size_t> node_to_slot_; 
    std::vector<SMSlot> slots_;
    std::vector<size_t> free_slots_;
    mutable std::mutex pool_mutex_;
    void init_slots();
    void destroy_pool();

};