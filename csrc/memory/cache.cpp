// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#include "cache.hpp"

std::unique_ptr<ZipMoECacheHandle> kZipMoECacheHandle = nullptr;

ZipMoECacheHandle::ZipMoECacheHandle(
    size_t total_memory_pool_size,
    size_t gpu_slot_size,
    size_t sm_slot_size,
    double gpu_memory_ratio,
    size_t num_pools,
    int num_layers,
    int num_experts,
    double hyperparam_margin,
    const std::string& caching_algorithm
):
    gpu_ptr_base_(nullptr),
    host_pinned_ptr_base_(nullptr),
    device_pinned_ptr_base_(nullptr),
    gpu_slot_size_(gpu_slot_size),
    sm_slot_size_(sm_slot_size),
    gpu_memory_ratio_(gpu_memory_ratio),
    num_gpu_pools_(num_pools),
    num_sm_pools_(num_pools),
    num_experts_(num_experts),
    num_layers_(num_layers),
    hyperparam_margin_(hyperparam_margin)
{
    if (caching_algorithm == "ZipMoE"){
        use_pool_planning = true;
        DLOG_INFO("[ZipMoECacheHandle] Applying Cache Pool Planning.");
    }
    gpu_pool_total_size_ = static_cast<size_t>(gpu_memory_ratio * static_cast<double>(total_memory_pool_size));
    sm_pool_total_size_ = total_memory_pool_size - gpu_pool_total_size_;
    size_t size_per_gpu_pool = ( gpu_pool_total_size_ + num_pools - 1 ) / num_pools;
    size_t size_per_sm_pool = ( sm_pool_total_size_ + num_pools - 1 ) / num_pools;
    num_gpu_slots_per_pool_ = ( size_per_gpu_pool + gpu_slot_size_ - 1 ) / gpu_slot_size_;
    num_sm_slots_per_pool_ = ( size_per_sm_pool + sm_slot_size_ - 1 ) / sm_slot_size_;
    size_per_gpu_pool = num_gpu_slots_per_pool_ * gpu_slot_size_;
    size_per_sm_pool = num_sm_slots_per_pool_ * sm_slot_size_;
    gpu_pool_total_size_ = size_per_gpu_pool * num_pools;
    sm_pool_total_size_ = size_per_sm_pool * num_pools;
    sparse_cache_size = gpu_pool_total_size_ + sm_pool_total_size_;
    allocate_resource(
        gpu_pool_total_size_,
        sm_pool_total_size_
    );
    size_t layer_id = 0;
    uint16_t* gpu_handover_ptr = gpu_ptr_base_;
    uint8_t* host_handover_ptr = host_pinned_ptr_base_;
    uint8_t* device_handover_ptr = device_pinned_ptr_base_;
    gpu_pools_.reserve(num_pools);
    sm_pools_.reserve(num_pools);
    gpu_cache_algorithms_.reserve(num_pools);
    sm_cache_algorithms_.reserve(num_pools);
    for (size_t pool_id = 0; pool_id < num_pools; pool_id++){
        gpu_pools_.emplace_back(
            std::make_unique<ZipMoEGPUMemoryPool>(
                size_per_gpu_pool,
                gpu_slot_size,
                static_cast<int>(pool_id),
                gpu_handover_ptr
            )
        );
        sm_pools_.emplace_back(
            std::make_unique<ZipMoEPinnedMemoryPool>(
                size_per_sm_pool,
                sm_slot_size,
                pool_id,
                host_handover_ptr,
                device_handover_ptr
            )
        );
        auto gpu_cache_policy = gpu_policy_factory_.construct(
            caching_algorithm,
            gpu_pools_[pool_id].get()
        );
        gpu_cache_policy->init(gpu_pools_[pool_id]->num_slots());
        gpu_cache_algorithms_.push_back(gpu_cache_policy);

        auto sm_cache_policy = sm_policy_factory_.construct(
            caching_algorithm,
            sm_pools_[pool_id].get()
        );
        sm_cache_policy->init(sm_pools_[pool_id]->num_slots());
        sm_cache_algorithms_.push_back(sm_cache_policy);
        if (num_pools == 1){
            std::vector<size_t> layer_ids;
            for (size_t i = layer_id; i<num_layers; i++){
                layer_ids.push_back(i);
            }
            assign_layers_to_pool(
                pool_id,
                layer_ids
            );
        } else {
            assign_layers_to_pool(
                pool_id,
                {layer_id}
            );
            gpu_handover_ptr = reinterpret_cast<uint16_t*>(
                reinterpret_cast<uint8_t*>(gpu_handover_ptr) + size_per_gpu_pool
            );
            host_handover_ptr += size_per_sm_pool;
            device_handover_ptr += size_per_sm_pool;
            layer_id += 1;
        }
        DLOG_INFO("number of full-cache slots: ", gpu_pools_[0]->num_slots());
        DLOG_INFO("number of SM-cache slots: ", sm_pools_[0]->num_slots());
    }
    set_up_cpu_zero_tensors();
    set_up_global_access_counts(
        num_layers,
        num_experts
    );
    Threshold_NORMAL = num_gpu_slots_per_pool_ + hyperparam_margin_ * num_experts_;
    Threshold_SMCACHE = Threshold_NORMAL + num_sm_slots_per_pool_ + hyperparam_margin_ * num_experts_;
}


void ZipMoECacheHandle::allocate_resource(
    size_t gpu_pool_total_size_,
    size_t sm_pool_total_size_
){
    CUDA_CHECK(
        cudaMalloc(
            &this->gpu_ptr_base_,
            gpu_pool_total_size_
        )
    );
    CUDA_CHECK(
        cudaHostAlloc(
            &this->host_pinned_ptr_base_,
            sm_pool_total_size_,
            cudaHostAllocMapped
        )
    );
    CUDA_CHECK(
        cudaHostGetDevicePointer(
            &this->device_pinned_ptr_base_,
            this->host_pinned_ptr_base_,
            0
        )
    );
}



void ZipMoECacheHandle::SetUpDensePool(
    size_t total_dense_size
){
    CUDA_CHECK(
        cudaMalloc(
            &this->dense_gpu_ptr_base,
            total_dense_size
        )
    );
}


void ZipMoECacheHandle::set_up_cpu_zero_tensors(){
    at::TensorOptions options;
    options = options.device(torch::kCPU);
    options = options.dtype(torch::kBFloat16);
    auto tensor = torch::zeros({1}, options);
    this->cpu_zero_tensor = tensor;
}


void ZipMoECacheHandle::set_up_global_access_counts(
    int num_layers,
    int num_experts
){
    global_access_counts.resize(num_layers);
    for (int lid = 0; lid < num_layers; lid++){
        global_access_counts[lid].resize(num_experts,0);
    }
}


void ZipMoECacheHandle::reset_global_access_counts(){
    for (int layer_id = 0; layer_id < num_layers_; layer_id++ ){
        for (int expert_id = 0; expert_id < num_experts_; expert_id++ ){
            global_access_counts[layer_id][expert_id] = 0;
        }
    }
}


void ZipMoECacheHandle::assign_layers_to_pool(
    size_t pool_id,
    const std::vector<size_t>& layer_ids
){

    std::lock_guard<std::mutex> lock_cache(cache_mutex_);
    std::lock_guard<std::mutex> lock_sm(sm_mutex_);

    if (pool_id >= gpu_pools_.size()||pool_id >= sm_pools_.size()){
        DLOG_FATAL("ZipMoECacheHandle: Invalid pool_id{}", pool_id);
    }
    for (size_t layer_id: layer_ids){
        layer_to_pool_[layer_id] = pool_id;
    }

}


std::vector<std::string> ZipMoECacheHandle::gpu_available_policies() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return gpu_policy_factory_.available_policies();
}


std::vector<std::string> ZipMoECacheHandle::sm_available_policies() const {
    std::lock_guard<std::mutex> lock(sm_mutex_);
    return sm_policy_factory_.available_policies();
}


TaskState ZipMoECacheHandle::ProactiveDecision(int layer_idx, int expert_idx, int num_tokens){
    if (use_pool_planning){

        global_access_counts[layer_idx][expert_idx] += num_tokens;
        uint64_t task_node_visit_count = global_access_counts[layer_idx][expert_idx];
        size_t count_up = 0;
        size_t count_down = 0;
        for (int e_idx = 0; e_idx < num_experts_; e_idx++){
            if (global_access_counts[layer_idx][e_idx]>task_node_visit_count){
                count_up++;
            }
            if (global_access_counts[layer_idx][e_idx]<task_node_visit_count){
                count_down++;
            }
        }
        if (count_up < Threshold_NORMAL){
            if (gpu_pools_[0]->has_free_slot()){
                return TaskState::NORMAL;
            }
            if (sm_pools_[0]->has_free_slot()){
                return TaskState::SMCACHE;
            }
            return TaskState::NORMAL;
        } else if (count_up < Threshold_SMCACHE){

            if (sm_pools_[0]->has_free_slot()){
                return TaskState::SMCACHE;
            }
            if (gpu_pools_[0]->has_free_slot()){
                return TaskState::NORMAL;
            }
            return TaskState::SMCACHE;
        } else {
            if (sm_pools_[0]->has_free_slot()){
                return TaskState::SMCACHE;
            }        
            if (gpu_pools_[0]->has_free_slot()){
                return TaskState::NORMAL;
            }
            return TaskState::EVICT;
        }
    } else {
        return TaskState::NORMAL;
    }
}


void ZipMoECacheHandle::UpdateOnHit(NodePtr& node, int num_token_reqs){
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& gpu_pool = gpu_pools_[pool_id];
    auto& sm_pool = sm_pools_[pool_id];
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto& algorithm = gpu_cache_algorithms_[pool_id];
        size_t slot_id = gpu_pool->get_slot_id_for_node(node->id);
        if ( slot_id != SIZE_MAX ){
            algorithm->on_hit(slot_id, num_token_reqs);
        }
    }
    {
        std::lock_guard<std::mutex> lock(sm_mutex_);
        auto& algorithm = sm_cache_algorithms_[pool_id];
        size_t slot_id = sm_pool->get_slot_id_for_node(node->id);
        if ( slot_id != SIZE_MAX ){
            algorithm->on_hit(slot_id, num_token_reqs);
        }
    }
}
size_t ZipMoECacheHandle::RegisterCacheSlotForNode(NodePtr& node){
    std::lock_guard<std::mutex> lock(cache_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& gpu_pool = gpu_pools_[pool_id];
    auto& algorithm = gpu_cache_algorithms_[pool_id];
    size_t slot_id = gpu_pool->get_slot_id_for_node(node->id);
    if ( slot_id != SIZE_MAX ){
        return slot_id;
    }
    size_t target_slot;
    if (gpu_pool->has_free_slot()){
        target_slot = gpu_pool->pop_free_slot();
    } else {
        target_slot = algorithm->select_victim_slot();
        algorithm->on_evict(target_slot);
        auto* target_slot_ptr = gpu_pool->get_slot(target_slot);
        if (target_slot_ptr&&target_slot_ptr->node){
            std::lock_guard<std::mutex> target_node_lock(target_slot_ptr->node->mutex);
            target_slot_ptr->node->SetDevice(
                DISK_DEVICE,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
        }
        gpu_pool->evict(target_slot);
    }
    gpu_pool->place(
        node, 
        target_slot
    );
    algorithm->on_miss(target_slot, node->corr_id & 0xFFFFFFFF);
    return target_slot;
}


CacheSlot* ZipMoECacheHandle::GetCacheSlotSlot(NodePtr& node){
    std::lock_guard<std::mutex> lock(cache_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& gpu_pool = gpu_pools_[pool_id];
    size_t target_slot_id = gpu_pool->get_slot_id_for_node(node->id);
    return gpu_pool->get_slot(target_slot_id);
}


void ZipMoECacheHandle::UnregisterCacheSlotForNode(NodePtr& node){
    std::lock_guard<std::mutex> lock(cache_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& gpu_pool = gpu_pools_[pool_id];
    size_t evict_slot_id = gpu_pool->get_slot_id_for_node(node->id);
    if(evict_slot_id != SIZE_MAX){
        node->SetDevice(
            DISK_DEVICE, nullptr, nullptr, nullptr, nullptr
        );
    }
    gpu_pool->evict(evict_slot_id);
}


bool ZipMoECacheHandle::IsSMCached(NodePtr& node){
    std::lock_guard<std::mutex> lock(sm_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& sm_pool = sm_pools_[pool_id];
    return sm_pool->contains(node);
}




size_t ZipMoECacheHandle::RegisterSMCacheForNode(NodePtr& node){
    std::lock_guard<std::mutex> lock(sm_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& sm_pool = sm_pools_[pool_id];
    auto& algorithm = sm_cache_algorithms_[pool_id];
    size_t slot_id = sm_pool->get_slot_id_for_node(node->id);
    if ( slot_id != SIZE_MAX ){
        return slot_id; 
    }

    size_t target_slot;
    if (sm_pool->has_free_slot()){
        target_slot = sm_pool->pop_free_slot();
    } else {
        target_slot = algorithm->select_victim_slot();
        algorithm->on_evict(target_slot);
        auto* target_slot_ptr = sm_pool->get_slot(target_slot);
        if (target_slot_ptr&&target_slot_ptr->node){
            std::lock_guard<std::mutex> target_node_lock(target_slot_ptr->node->mutex);
            target_slot_ptr->node->SetSM(
                nullptr
            );
        }
        sm_pool->evict(target_slot);
    }
    sm_pool->place(
        node, 
        target_slot
    );
    algorithm->on_miss(target_slot, node->corr_id & 0xFFFFFFFF);
    return target_slot;
}


SMSlot* ZipMoECacheHandle::GetSMCacheSlot(NodePtr& node){
    std::lock_guard<std::mutex> lock(sm_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& sm_pool = sm_pools_[pool_id];
    size_t target_slot_id = sm_pool->get_slot_id_for_node(node->id);
    return sm_pool->get_slot(target_slot_id);
}


void ZipMoECacheHandle::UnregisterSMCacheForNode(NodePtr& node){
    std::lock_guard<std::mutex> lock(sm_mutex_);
    size_t layer_id = node_to_layer(node);
    size_t pool_id = layer_to_pool(layer_id);
    auto& sm_pool = sm_pools_[pool_id];
    size_t evict_slot_id = sm_pool->get_slot_id_for_node(node->id);
    if (evict_slot_id != SIZE_MAX){
        node->SetSM(nullptr);
    }
    sm_pool->evict(evict_slot_id);
}


template class CachePolicyZipMoE<ZipMoEGPUMemoryPool>;
template class CachePolicyZipMoE<ZipMoEPinnedMemoryPool>;

template class CachePolicyLRU<ZipMoEGPUMemoryPool>;
template class CachePolicyLRU<ZipMoEPinnedMemoryPool>;

template class CachePolicyLFU<ZipMoEGPUMemoryPool>;
template class CachePolicyLFU<ZipMoEPinnedMemoryPool>;


template class CachePolicyMarking<ZipMoEGPUMemoryPool>;
template class CachePolicyMarking<ZipMoEPinnedMemoryPool>;


template class CachePolicyFIFO<ZipMoEGPUMemoryPool>;
template class CachePolicyFIFO<ZipMoEPinnedMemoryPool>;


template<typename PoolType>
size_t CachePolicyZipMoE<PoolType>::select_victim_slot(){
    const auto& slots = this->pool_->get_all_slots();
    size_t victim_slot = SIZE_MAX;
    uint64_t min_count = UINT64_MAX;

    for (size_t i = 0; i < num_slots_; i++){
        if ( !slots[i].is_free() && !this->is_protected(i) ){
            uint64_t count = access_counts_[i].load(std::memory_order_relaxed);
            if (count < min_count){
                min_count = count;
                victim_slot = i;
            }
        }
    }
    if (victim_slot == SIZE_MAX){
        DLOG_FATAL("[ZipMoE]: No victim found! All slots are either free or protected!");
    }
    return victim_slot;
}


template<typename PoolType>
void CachePolicyZipMoE<PoolType>::on_hit(size_t slot_id, size_t num_tokens){
    if (slot_id < num_slots_ ){
        access_counts_[slot_id].fetch_add(num_tokens,std::memory_order_relaxed);
    }
}


template<typename PoolType>
void CachePolicyZipMoE<PoolType>::on_miss(size_t slot_id, uint32_t layer_id){
    if (slot_id < num_slots_ ){
        access_counts_[slot_id].store(1,std::memory_order_relaxed);
    }
}


template<typename PoolType>
void CachePolicyZipMoE<PoolType>::on_evict(size_t slot_id){
    if (slot_id < num_slots_){
        access_counts_[slot_id].store(0,std::memory_order_relaxed);
    }
}



template<typename PoolType>
void CachePolicyMarking<PoolType>::start_new_phase(){
    // Clear all marks and start new marking phase
    for (auto& meta: metadata_){
        meta.marked = 0;
    }
    is_mark_phase_.store(true, std::memory_order_relaxed);
    eviction_hand_ = 0;
}


template<typename PoolType>
size_t CachePolicyMarking<PoolType>::select_victim_slot() {
    const auto& slots = this->pool_->get_all_slots();
    const size_t num_slots = slots.size();


    auto pick_random_candidate = [&](bool only_unmarked) -> size_t {
        size_t victim = SIZE_MAX;
        size_t candidates_count = 0;
        for (size_t i = 0; i < num_slots; i++) {
            if (!slots[i].is_free() && !this->is_protected(i)) {
                if (only_unmarked && metadata_[i].marked == 1) {
                    continue;
                }
                candidates_count++;
                if ((std::rand() % candidates_count) == 0) {
                    victim = i;
                }
            }
        }
        return victim;
    };


    size_t victim_slot = pick_random_candidate(true);

    if (victim_slot != SIZE_MAX) {
        return victim_slot;
    }

    start_new_phase();
    victim_slot = pick_random_candidate(false);

    if (victim_slot == SIZE_MAX) {
        DLOG_FATAL("[Marking]: No victim found! All slots are either free or protected!");
    }
    return victim_slot;
}

template<typename PoolType>
void CachePolicyMarking<PoolType>::on_hit(size_t slot_id, size_t num_tokens){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].marked = 1;
        metadata_[slot_id].access_count++;
    }
}

template<typename PoolType>
void CachePolicyMarking<PoolType>::on_miss(size_t slot_id, uint32_t layer_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].marked = 1;
        metadata_[slot_id].access_count = 1;
        metadata_[slot_id].layer_id = layer_id;
    }
}


template<typename PoolType>
void CachePolicyMarking<PoolType>::on_evict(size_t slot_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].marked = 0;
        metadata_[slot_id].access_count = 0;
        metadata_[slot_id].layer_id = SIZE_MAX;
    }
}

template<typename PoolType>
size_t CachePolicyLRU<PoolType>::select_victim_slot(){
    const auto& slots = this->pool_->get_all_slots();
    size_t victim_slot = SIZE_MAX;
    uint64_t oldest_time = UINT64_MAX;

    for (size_t i = 0; i < slots.size(); i++){
        if ( !slots[i].is_free() && !this->is_protected(i) ){
            if (metadata_[i].last_access_time < oldest_time){
                oldest_time = metadata_[i].last_access_time;
                victim_slot = i;
            }
        }
    }
    if (victim_slot == SIZE_MAX){
        DLOG_FATAL("[LRU]: No victim found! All slots are either free or protected!");
    }
    return victim_slot;
}

template<typename PoolType>
void CachePolicyLRU<PoolType>::on_hit(size_t slot_id, size_t num_tokens){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].last_access_time = MILLISECONDS_SINCE_EPOCH;
    }
}

template<typename PoolType>
void CachePolicyLRU<PoolType>::on_miss(size_t slot_id, uint32_t layer_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].last_access_time = MILLISECONDS_SINCE_EPOCH;
    }
}

template<typename PoolType>
void CachePolicyLRU<PoolType>::on_evict(size_t slot_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].last_access_time = UINT64_MAX;
    }
}

template<typename PoolType>
size_t CachePolicyLFU<PoolType>::select_victim_slot(){
    const auto& slots = this->pool_->get_all_slots();
    size_t victim_slot = SIZE_MAX;
    uint64_t min_count = UINT64_MAX;

    for (size_t i = 0; i < num_slots_; i++){
        if ( !slots[i].is_free() && !this->is_protected(i) ){
            uint64_t count = access_counts_[i].load(std::memory_order_relaxed);
            if (count < min_count){
                min_count = count;
                victim_slot = i;
            }
        }
    }
    if (victim_slot == SIZE_MAX){
        DLOG_FATAL("[LFU]: No victim found! All slots are either free or protected!");
    }
    return victim_slot;
}

template<typename PoolType>
void CachePolicyLFU<PoolType>::on_hit(size_t slot_id, size_t num_tokens){
    if (slot_id < num_slots_ ){
        access_counts_[slot_id].fetch_add(1,std::memory_order_relaxed);
    }
}

template<typename PoolType>
void CachePolicyLFU<PoolType>::on_miss(size_t slot_id, uint32_t layer_id){
    if (slot_id < num_slots_ ){
        access_counts_[slot_id].store(1,std::memory_order_relaxed);
    }
}

template<typename PoolType>
void CachePolicyLFU<PoolType>::on_evict(size_t slot_id){
    if (slot_id < num_slots_){
        access_counts_[slot_id].store(0,std::memory_order_relaxed);
    }
}

template<typename PoolType>
size_t CachePolicyFIFO<PoolType>::select_victim_slot(){
    const auto& slots = this->pool_->get_all_slots();
    size_t victim_slot = SIZE_MAX;
    uint64_t oldest_time = UINT64_MAX;

    for (size_t i = 0; i < slots.size(); i++){
        if ( !slots[i].is_free() && !this->is_protected(i) ){
            if (metadata_[i].insertion_time < oldest_time){
                oldest_time = metadata_[i].insertion_time;
                victim_slot = i;
            }
        }
    }
    if (victim_slot == SIZE_MAX){
        DLOG_FATAL("[FIFO]: No victim found! All slots are either free or protected!");
    }
    return victim_slot;
}

template<typename PoolType>
void CachePolicyFIFO<PoolType>::on_hit(size_t slot_id, size_t num_tokens){
    if (slot_id < metadata_.size()){
    }
}

template<typename PoolType>
void CachePolicyFIFO<PoolType>::on_miss(size_t slot_id, uint32_t layer_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].insertion_time = MILLISECONDS_SINCE_EPOCH;
    }
}

template<typename PoolType>
void CachePolicyFIFO<PoolType>::on_evict(size_t slot_id){
    if (slot_id < metadata_.size()){
        metadata_[slot_id].insertion_time = UINT64_MAX;
    }
}