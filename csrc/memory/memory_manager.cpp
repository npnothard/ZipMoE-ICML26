// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "memory_manager.hpp"
#include <iostream>
#include <cassert>

ZipMoEGPUMemoryPool::ZipMoEGPUMemoryPool(
    size_t pool_size,
    size_t slot_size,
    int pool_id,
    uint16_t* handover_ptr
):
    base_ptr_(handover_ptr), pool_size_(pool_size), slot_size_(slot_size), pool_id_(pool_id)
{
    num_slots_ = (pool_size + slot_size_ - 1) / slot_size_;
    pool_size_ = slot_size_ * num_slots_;
    if (num_slots_ == 0){ DLOG_FATAL("[ZipMoEGPUMemoryPool] Pool size too small for even one slot!"); }

    if (base_ptr_ == nullptr){
        CUDA_CHECK( cudaMalloc(&base_ptr_, pool_size_) );
    }
    init_slots();
}

ZipMoEGPUMemoryPool::~ZipMoEGPUMemoryPool(){
    destroy_pool();
}

void ZipMoEGPUMemoryPool::init_slots(){
    slots_.resize(num_slots_);
    free_slots_.reserve(num_slots_);

    uint16_t* slot_ptr = base_ptr_;
    for (size_t i = 0; i < num_slots_; i++){
        slots_[i].id = i;
        slots_[i].size = slot_size_;
        slots_[i].ptr = slot_ptr;
        free_slots_.push_back(i);

        slot_ptr = reinterpret_cast<uint16_t*>(
            reinterpret_cast<uint8_t*>(slot_ptr) + slot_size_
        );
    }
}

void ZipMoEGPUMemoryPool::destroy_pool(){
    if (base_ptr_){
        cudaFree(base_ptr_);
        base_ptr_ = nullptr;
    }
}


void ZipMoEGPUMemoryPool::place(
    NodePtr& node,
    size_t slot_id
){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if(slot_id >= num_slots_){ DLOG_FATAL("[ZipMoEGPUMemoryPool] Invalid slot ID."); }
    CacheSlot& slot = slots_[slot_id];
    if(!slot.is_free()){ DLOG_FATAL("[ZipMoEGPUMemoryPool] Slot {} is not free!.", std::to_string(slot_id) ); }
    slot.node = node;
    node_to_slot_[node->id] = slot_id;
    auto it = std::find(
        free_slots_.begin(),
        free_slots_.end(),
        slot_id
    );
    if (it != free_slots_.end()){
        free_slots_.erase(it);
    }
}


void ZipMoEGPUMemoryPool::evict( size_t slot_id ){

    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (slot_id >= num_slots_){ DLOG_FATAL("[ZipMoEGPUMemoryPool] Invalid slot ID."); }

    CacheSlot& slot = slots_[slot_id];
    if(slot.is_free()){
        return;
    }
    node_to_slot_.erase(slot.node->id);
    slot.reset();
    free_slots_.push_back(slot_id);
}

bool ZipMoEGPUMemoryPool::contains( const NodePtr& node ) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return node_to_slot_.find(node->id) != node_to_slot_.end();
}

bool ZipMoEGPUMemoryPool::has_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return !free_slots_.empty();
}

size_t ZipMoEGPUMemoryPool::get_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[ZipMoEGPUMemoryPool] No free slot exists.");
    }
    return free_slots_.back();
}

size_t ZipMoEGPUMemoryPool::pop_free_slot() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[ZipMoEGPUMemoryPool] No free slot exists.");
    }
    size_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    return slot_id;
}

CacheSlot* ZipMoEGPUMemoryPool::get_slot( size_t slot_id ){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (slot_id>=num_slots_) { return nullptr; }
    return &slots_[slot_id];
}

const std::vector<CacheSlot>& ZipMoEGPUMemoryPool::get_all_slots() const {
    return slots_;
}

size_t ZipMoEGPUMemoryPool::get_slot_id_for_node(size_t node_id) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto it = node_to_slot_.find(node_id);
    if (it != node_to_slot_.end()) {
        return it->second;
    }
    return SIZE_MAX;
}

size_t ZipMoEGPUMemoryPool::num_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return num_slots_;
}

size_t ZipMoEGPUMemoryPool::num_free_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_slots_.size();
}


void ZipMoEGPUMemoryPool::print_status() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    std::cout << "[ZipMoEGPUMemoryPool] Status:\n"
                << "  Total slots: " << num_slots_ << "\n"
                << "  Free slots: " << free_slots_.size() << "\n"
                << "  Used slots: " << (num_slots_ - free_slots_.size()) << "\n"
                << "  Slot size: " << slot_size_ << " bytes\n"
                << "  Total size: " << pool_size_ << " bytes\n";
}


ZipMoEPinnedMemoryPool::ZipMoEPinnedMemoryPool(
    size_t pool_size,
    size_t slot_size,
    int pool_id,
    uint8_t* host_handover_ptr,
    uint8_t* device_handover_ptr
):
    host_base_ptr_(host_handover_ptr), device_base_ptr_(device_handover_ptr),
    pool_size_(pool_size), slot_size_(slot_size), pool_id_(pool_id)
{

    num_slots_ = (pool_size + slot_size_ - 1) / slot_size_;
    pool_size_ = slot_size_ * num_slots_;
    if (num_slots_ == 0){ DLOG_FATAL("[ZipMoEGPUMemoryPool] Pool size too small for even one slot!"); }
    if (host_base_ptr_==nullptr){
        CUDA_CHECK( 
            cudaHostAlloc(&host_base_ptr_, pool_size_, cudaHostAllocMapped)
        );
        CUDA_CHECK(
            cudaHostGetDevicePointer(&device_base_ptr_, host_base_ptr_, 0)
        );
    }
    init_slots();

}


ZipMoEPinnedMemoryPool::~ZipMoEPinnedMemoryPool(){
    destroy_pool();
}


void ZipMoEPinnedMemoryPool::init_slots(){
    slots_.resize(num_slots_);
    free_slots_.reserve(num_slots_);

    uint8_t* host_slot_ptr = host_base_ptr_;
    uint8_t* device_slot_ptr = device_base_ptr_;
    for (size_t i = 0; i < num_slots_; i++){
        slots_[i].id = i;
        slots_[i].size = slot_size_;
        slots_[i].ptr = std::make_pair(host_slot_ptr, device_slot_ptr);
        free_slots_.push_back(i);
        host_slot_ptr = host_slot_ptr + slot_size_;
        device_slot_ptr = device_slot_ptr + slot_size_;
    }
}


void ZipMoEPinnedMemoryPool::destroy_pool(){
    if (host_base_ptr_){
        cudaFreeHost(host_base_ptr_);
        host_base_ptr_ = nullptr;
    }
}

void ZipMoEPinnedMemoryPool::place(
    NodePtr& node,
    size_t slot_id
){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if(slot_id >= num_slots_){ DLOG_FATAL("[ZipMoEPinnedMemoryPool] Invalid slot ID."); }
    SMSlot& slot = slots_[slot_id];
    if(!slot.is_free()){ DLOG_FATAL("[ZipMoEPinnedMemoryPool] Slot {} is not free!.", std::to_string(slot_id) ); }
    slot.node = node;
    node_to_slot_[node->id] = slot_id;
    auto it = std::find(
        free_slots_.begin(),
        free_slots_.end(),
        slot_id
    );
    if (it != free_slots_.end()){
        free_slots_.erase(it);
    }

}


void ZipMoEPinnedMemoryPool::evict( size_t slot_id ){

    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (slot_id >= num_slots_){ DLOG_FATAL("[ZipMoEPinnedMemoryPool] Invalid slot ID."); }

    SMSlot& slot = slots_[slot_id];
    if(slot.is_free()){
        return;
    }
    node_to_slot_.erase(slot.node->id);
    slot.reset();
    free_slots_.push_back(slot_id);
}


bool ZipMoEPinnedMemoryPool::contains( const NodePtr& node ) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return node_to_slot_.find(node->id) != node_to_slot_.end();
}


bool ZipMoEPinnedMemoryPool::has_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return !free_slots_.empty();
}


size_t ZipMoEPinnedMemoryPool::get_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[ZipMoEGPUMemoryPool] No free slot exists.");
    }
    return free_slots_.back();
}


size_t ZipMoEPinnedMemoryPool::pop_free_slot() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[ZipMoEGPUMemoryPool] No free slot exists.");
    }
    size_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    return slot_id;
}


SMSlot* ZipMoEPinnedMemoryPool::get_slot( size_t slot_id ){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (slot_id>=num_slots_) { return nullptr; }
    return &slots_[slot_id];
}


const std::vector<SMSlot>& ZipMoEPinnedMemoryPool::get_all_slots() const {
    return slots_;
}



size_t ZipMoEPinnedMemoryPool::get_slot_id_for_node(size_t node_id) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto it = node_to_slot_.find(node_id);
    if (it != node_to_slot_.end()) {
        return it->second;
    }
    return SIZE_MAX;
}


size_t ZipMoEPinnedMemoryPool::num_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return num_slots_;
}


size_t ZipMoEPinnedMemoryPool::num_free_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_slots_.size();
}


void ZipMoEPinnedMemoryPool::print_status() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    std::cout << "[ZipMoEGPUMemoryPool] Status:\n"
                << "  Total slots: " << num_slots_ << "\n"
                << "  Free slots: " << free_slots_.size() << "\n"
                << "  Used slots: " << (num_slots_ - free_slots_.size()) << "\n"
                << "  Slot size: " << slot_size_ << " bytes\n"
                << "  Total size: " << pool_size_ << " bytes\n";
}

