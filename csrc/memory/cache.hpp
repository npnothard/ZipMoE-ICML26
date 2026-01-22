// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#pragma once

#include <memory>
#include <vector>
#include <cstdlib>

#include "common/time.hpp"
#include "base/noncopyable.hpp"
#include "common/pytorch.hpp"
#include "topology/model_topology.hpp"
#include "memory_manager.hpp"

enum TaskState {
    PREFETCH,
    NORMAL,
    SMCACHE,
    EVICT
};

template<typename PoolType>
class CachePolicyFactory;


template<typename PoolType>
class CachePolicy;


template<typename PoolType>
class CachePolicyZipMoE;


template<typename PoolType>
class CachePolicyLFU;

template<typename PoolType>
class CachePolicyLRU;

template<typename PoolType>
class CachePolicyMarking;

template<typename PoolType>
class CachePolicyFIFO;

template<typename PoolType>
class CachePolicy{
protected:
    PoolType* pool_;
    mutable std::shared_mutex metadata_mutex;
public:
    CachePolicy(PoolType* pool): pool_(pool) {}

    inline bool is_protected(size_t slot_id){

        auto* slot = this->pool_->get_slot(slot_id);
        NodePtr& node = slot->node;
        if (!node){
            return false;
        }
        if (node->mutex.try_lock()){
            node->mutex.unlock();
            return false;
        }
        return true;
    }


    virtual ~CachePolicy() = default;
    virtual size_t select_victim_slot() = 0;
    virtual void init(size_t num_slots) {}
    virtual void on_hit(size_t slot_id, size_t num_tokens) {}
    virtual void on_miss(size_t slot_id, uint32_t layer_id) {}
    virtual void on_evict(size_t slot_id) {}

    virtual std::string name() const = 0;
};


template<typename PoolType>
class CachePolicyZipMoE : public CachePolicy<PoolType> {
private:
    std::unique_ptr<std::atomic<uint64_t>[]> access_counts_;
    size_t num_slots_ = 0;
public:
    using CachePolicy<PoolType>::CachePolicy;
    
    void init(size_t num_slots ) override {
        num_slots_ = num_slots;
        access_counts_ = std::make_unique<std::atomic<uint64_t>[]>(num_slots);
    }
    size_t select_victim_slot() override;
    void on_hit(size_t slot_id, size_t num_tokens) override;
    void on_miss(size_t slot_id, uint32_t layer_id) override;
    void on_evict(size_t slot_id) override;
    std::string name() const override { return "ZipMoE"; }
};


template<typename PoolType>
class CachePolicyLRU : public CachePolicy<PoolType> {
private:
    struct SlotMetadata{
        uint64_t last_access_time;
        SlotMetadata(): last_access_time(0) {}
    };
    std::vector<SlotMetadata> metadata_;

public:
    using CachePolicy<PoolType>::CachePolicy;

    void init(size_t num_slots ) override {
        metadata_.resize(num_slots);
    }
    size_t select_victim_slot() override;
    void on_hit(size_t slot_id, size_t num_tokens) override;
    void on_miss(size_t slot_id, uint32_t layer_id) override;
    void on_evict(size_t slot_id) override;
    std::string name() const override { return "LRU"; }
};


template<typename PoolType>
class CachePolicyMarking : public CachePolicy<PoolType> {
private:
    struct SlotMetadata{
        uint8_t marked;
        uint32_t layer_id;
        uint64_t access_count;
        SlotMetadata(): marked(0), layer_id(SIZE_MAX), access_count(0) {}
    };
    std::vector<SlotMetadata> metadata_;
    std::atomic<bool> is_mark_phase_;
    size_t eviction_hand_;
    void start_new_phase();

public:
    using CachePolicy<PoolType>::CachePolicy;
    
    void init(size_t num_slots ) override {
        metadata_.resize(num_slots);
        is_mark_phase_.store(true, std::memory_order_relaxed);
        eviction_hand_ = 0;
    }

    size_t select_victim_slot() override;
    void on_hit(size_t slot_id, size_t num_tokens) override;
    void on_miss(size_t slot_id, uint32_t layer_id) override;
    void on_evict(size_t slot_id) override;
    std::string name() const override { return "Marking"; }
};

template<typename PoolType>
class CachePolicyLFU : public CachePolicy<PoolType> {
private:
    std::unique_ptr<std::atomic<uint64_t>[]> access_counts_;
    size_t num_slots_ = 0;
public:
    using CachePolicy<PoolType>::CachePolicy;
    void init(size_t num_slots ) override {
        num_slots_ = num_slots;
        access_counts_ = std::make_unique<std::atomic<uint64_t>[]>(num_slots);
    }
    size_t select_victim_slot() override;
    void on_hit(size_t slot_id, size_t num_tokens) override;
    void on_miss(size_t slot_id, uint32_t layer_id) override;
    void on_evict(size_t slot_id) override;
    std::string name() const override { return "LFU"; }
};


template<typename PoolType>
class CachePolicyFIFO : public CachePolicy<PoolType> {
private:
    struct SlotMetadata{
        uint64_t insertion_time;
        SlotMetadata(): insertion_time(0) {}
    };
    std::vector<SlotMetadata> metadata_;

public:
    using CachePolicy<PoolType>::CachePolicy;
    
    void init(size_t num_slots ) override {
        metadata_.resize(num_slots);
    }

    size_t select_victim_slot() override;

    void on_hit(size_t slot_id, size_t num_tokens) override;

    void on_miss(size_t slot_id, uint32_t layer_id) override;

    void on_evict(size_t slot_id) override;

    std::string name() const override { return "FIFO"; }
};


template<typename PoolType>
class CachePolicyFactory{
private:
    using PolicyConstructor = std::function<
        std::shared_ptr<CachePolicy<PoolType>>(PoolType*)
    >;
    std::map<std::string, PolicyConstructor> registry_;

    void register_builtin_policies(){

        register_policy(
            "ZipMoE",
            [](PoolType* pool){
                return std::make_shared<CachePolicyZipMoE<PoolType>>(pool);
            }
        );
        register_policy(
            "LRU",
            [](PoolType* pool){
                return std::make_shared<CachePolicyLRU<PoolType>>(pool);
            }
        );
        register_policy(
            "LFU",
            [](PoolType* pool){
                return std::make_shared<CachePolicyLFU<PoolType>>(pool);
            }
        );
        register_policy(
            "Marking",
            [](PoolType* pool){
                return std::make_shared<CachePolicyMarking<PoolType>>(pool);
            }
        );
        register_policy(
            "FIFO",
            [](PoolType* pool){
                return std::make_shared<CachePolicyFIFO<PoolType>>(pool);
            }
        );
    };


public:
    CachePolicyFactory(){
        register_builtin_policies();
    }

    void register_policy(
        std::string name,
        PolicyConstructor constructor
    ){
        if (registry_.find(name) != registry_.end()){
            DLOG_WARN("[CachePolicyFactory] Overwriting existing policy: {}", name);
        }
        registry_[name] = constructor;
    }

    bool unregister_policy( const std::string& name ){
        return registry_.erase(name) > 0;
    }
    std::shared_ptr<CachePolicy<PoolType>> construct(
        const std::string& name,
        PoolType* pool
    ){
        auto it = registry_.find(name);
        if (it == registry_.end()){
            DLOG_FATAL("PolicyFactory: Uknown cache policy: {}.", name.c_str());

        }
        return it->second(pool);
    }

    bool has_policy( const std::string& name ) const {
        return registry_.find(name) != registry_.end();
    }

    std::vector<std::string> available_policies() const {
        std::vector<std::string> names;
        for (const auto& pair: registry_){ names.push_back(pair.first); }
        return names;
    }
};


class ZipMoECacheHandle : public base::noncopyable {

private:
    std::vector<std::unique_ptr<ZipMoEGPUMemoryPool>>  gpu_pools_;
    std::vector<std::unique_ptr<ZipMoEPinnedMemoryPool>>  sm_pools_;
    size_t gpu_slot_size_;
    size_t sm_slot_size_;
    std::vector<std::shared_ptr<CachePolicy<ZipMoEGPUMemoryPool>>> gpu_cache_algorithms_;
    std::vector<std::shared_ptr<CachePolicy<ZipMoEPinnedMemoryPool>>> sm_cache_algorithms_;
    CachePolicyFactory<ZipMoEGPUMemoryPool> gpu_policy_factory_;
    CachePolicyFactory<ZipMoEPinnedMemoryPool> sm_policy_factory_;
    std::unordered_map<size_t, size_t> layer_to_pool_;
    mutable std::mutex cache_mutex_;
    mutable std::mutex sm_mutex_;
    double gpu_memory_ratio_ = 0;
    size_t gpu_pool_total_size_ = 0;
    size_t num_gpu_pools_ = 0;
    size_t num_gpu_slots_per_pool_ = 0;
    size_t sm_pool_total_size_ = 0;
    size_t num_sm_pools_ = 0;
    size_t num_sm_slots_per_pool_ = 0;
    uint16_t* gpu_ptr_base_;
    uint8_t* host_pinned_ptr_base_;
    uint8_t* device_pinned_ptr_base_;
    size_t num_experts_;
    size_t num_layers_;
    double hyperparam_margin_;
    size_t Threshold_NORMAL;
    size_t Threshold_SMCACHE;
    bool use_pool_planning = false;
    void allocate_resource(
        size_t gpu_pool_total_size_,
        size_t sm_pool_total_size_
    );

    void set_up_cpu_zero_tensors();
    inline size_t layer_to_pool( size_t layer_id ){
        return 0;
    }

    inline size_t node_to_layer( const NodePtr& node ){
        return node->corr_id & 0xFFFFFFFF;
    }

public:
    at::Tensor cpu_zero_tensor;
    uint8_t* dense_gpu_ptr_base = nullptr;
    std::vector<std::vector<uint64_t>> global_access_counts;
    int64_t sparse_cache_size;
    DELETE_COPY_AND_ASSIGN(ZipMoECacheHandle);
    ZipMoECacheHandle(
        size_t total_memory_pool_size,
        size_t gpu_slot_size,
        size_t sm_slot_size,
        double gpu_memory_ratio,
        size_t num_pools,
        int num_layers,
        int num_experts,

        double hyperparam_margin,
        const std::string& caching_algorithm
    );
    void assign_layers_to_pool(size_t pool_id, const std::vector<size_t>& layer_ids);
    void set_up_global_access_counts(int num_layers, int num_experts);
    void reset_global_access_counts();
    size_t num_pools() const { return gpu_pools_.size(); }
    std::vector<std::string> gpu_available_policies() const;
    std::vector<std::string> sm_available_policies() const;
    void SetUpDensePool(size_t total_dense_size);
    TaskState ProactiveDecision(int layer_idx, int expert_idx, int num_tokens);
    void UpdateOnHit(NodePtr& node, int num_token_reqs);
    size_t RegisterCacheSlotForNode(NodePtr& node);
    CacheSlot* GetCacheSlotSlot(NodePtr& node);
    void UnregisterCacheSlotForNode(NodePtr& node);
    bool IsSMCached(NodePtr& node);
    size_t RegisterSMCacheForNode(NodePtr& node);
    SMSlot* GetSMCacheSlot(NodePtr& node);
    void UnregisterSMCacheForNode(NodePtr& node);
};
extern std::unique_ptr<ZipMoECacheHandle> kZipMoECacheHandle;





