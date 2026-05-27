// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// Modifications and additions to this file are licensed under the
// Academic Non-Commercial License. See the LICENSE file in the
// project root for details.
//
// -------------------------------------------------------------------
// DERIVED FROM:
// EfficientMoE (Apache License 2.0)
// Copyright (c) EfficientMoE.
//
// The original code is licensed under the Apache License, Version 2.0.
// This file contains substantial modifications.
// -------------------------------------------------------------------



#pragma once

#include <cuda_runtime.h>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>


#include "../common/pytorch.hpp"
#include "../common/types.hpp"
#include "kernels/tensor_recover.hpp"
enum NodeState {
  NODE_STATE_NONE = 0x0,
  NODE_STATE_CACHED = 0x1,
  NODE_STATE_PREFETCHED = 0x2,
  NODE_STATE_VISITED = 0x4,
};

struct Node{
    std::vector<TensorID> tensor_ids;
    size_t byte_size;
    std::size_t last_access_time;
    std:: size_t last_prefetch_time = 0;
    std::size_t id;
    std::size_t corr_id;
    torch::Device device = DISK_DEVICE;
    torch::Device default_device = DEFAULT_CUDA_DEVICE;
    torch::Device default_host = DISK_DEVICE;
    torch::Device initial_host = DISK_DEVICE;
    std::atomic_uint8_t state{0};
    std::mutex mutex;
    std::condition_variable cv;
    uint8_t* dense_gpu_offset_ptr = nullptr;
    std::uint64_t visit_count = 0;
    std::uint64_t incache_visit_count = 0;
    std::uint64_t unused_count = 0;
    bool is_sparse = false;
    NodeState io_state = NODE_STATE_NONE;
    size_t num_elements;
    uint8_t* device_sm_ptr = nullptr;
    uint16_t* device_memory_ptr = nullptr;
public:
    explicit Node();
    const std::string str();
    void SetDevice(
        const torch::Device& target_device,
        uint8_t* gpu_exp_ptr,
        uint8_t* gpu_sm_ptr,
        uint16_t* pending_recover_gpu_ptr,
        cudaStream_t stream
    );
    void SetSM(
        uint8_t* device_sm_ptr
    );
    void PinDense(
        const torch::Device& target_device,
        uint8_t* dense_offset_gpu_ptr
    );

};

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodePtrList;
typedef std::tuple<std::int64_t, NodePtrList> FilterResult;

struct Stage {
    bool is_sparse;
    std::vector<NodePtr> nodes;
    size_t visit_cnt;
    int64_t byte_size;
    std::deque<size_t> visit_time;
    std::unordered_set<size_t> activate_request;

    Stage(): is_sparse(false), visit_cnt(0), byte_size(0) {}
    Stage(bool is_sparse): is_sparse(is_sparse), visit_cnt(0), byte_size(0) {}

    std::string str() const noexcept {
        char buffer[1024];
        memset(buffer, 0, 1024);
        sprintf(buffer, "Stage[%ld,%ld,%d]", nodes.size(), visit_cnt, is_sparse);
        return std::string(buffer);
    }
};
typedef std::shared_ptr<Stage> StagePtr;

struct Pipeline{
    std::vector<StagePtr> stages;
    size_t visit_cnt = 0;

    std::string str() const noexcept {
    std::stringstream ss;
    ss << "Pipeline: " << stages.size() << " stages; visit_cnt " << visit_cnt
        << std::endl;
    return ss.str();
    }
};
typedef std::shared_ptr<Pipeline> PipelinePtr;

class ZipMoETopologyHandle: public base::noncopyable{
public:
    DELETE_COPY_AND_ASSIGN(ZipMoETopologyHandle);
    ZipMoETopologyHandle();
    ~ZipMoETopologyHandle() = default;
    bool IsLastNode(const NodePtr& node);
    bool IsFirstNode(const NodePtr& node);
    NodePtrList GetDenseNodes(const NodePtr& node, const std::size_t& k);
    NodePtrList GetSparseNodes(const NodePtr& node, const std::size_t& k);
    NodePtrList GetDenseNodes();
    NodePtrList GetSparseNodes();
    std::uint64_t GetLastActivateStage(const HashID& hash_id);
    void ExamineCompreesionRatio();
    void InitializeTopology(
        const std::vector<
            std::tuple<
                std::string,
                std::vector<
                    std::vector<
                        TensorID
                    >
                >
            >    
        >& topology
    );
    void EnableTrace() noexcept { trace_enabled_ = true; }
    void DisableTrace() noexcept { trace_enabled_ = false; }
    std::vector<std::vector<std::size_t>> GetNodeVisitCounts();
    NodePtr GetNodeFromTensorID(const TensorID& tensor_id);
    std::tuple<std::size_t, std::size_t> GetNumLayersAndExperts();
    std::int64_t GetSparseCacheLimit(const torch::Device& device);
    size_t GetDenseNodesTotalSize();
    std::size_t GetNumberOfStages() const noexcept {
        return pipeline_.stages.size();
    }
private:
    Pipeline pipeline_;
    std::unordered_set<HashID> visited_;
    std::unordered_map<HashID, std::uint64_t> last_active_stage_;
    std::unordered_map<std::size_t, std::size_t> request_time_;
    std::unordered_map<std::size_t, StagePtr> request_trace_;
    std::int64_t visit_count_ = 0;
    std::mutex mutex_;
    bool trace_enabled_ = true;
    std::unordered_map<TensorID, NodePtr> tensor_id_to_node_;
};

extern std::unique_ptr<ZipMoETopologyHandle> kTopologyHandle;
extern std::mutex kReadMutex;

#define CONTINUE_IF_NULL(node) \
  if (node == nullptr) continue;
#define BREAK_IF_NULL(node) \
  if (node == nullptr) break;


void SetModuleDisk(std::vector<TensorID>& tensor_ids);
void SetDenseModuleFromDisk(
    std::vector<TensorID>& tensor_ids,
    uint8_t* dense_offset_gpu_ptr,
    const torch::Device& device
);
void SetModuleMemoryFromDisk(
    std::vector<TensorID>& tensor_ids,
    uint8_t* gpu_exp_ptr,
    uint8_t* gpu_sm_ptr,
    uint16_t* pending_recover_gpu_ptr,
    cudaStream_t stream,
    const torch::Device& device
);
void SetModuleMemoryFromDiskGrouped(
    std::vector<TensorID>& tensor_ids,
    size_t node_num_elements,
    uint8_t* gpu_exp_ptr,
    uint8_t* gpu_sm_ptr,
    uint16_t* pending_recover_gpu_ptr,
    cudaStream_t stream,
    const torch::Device& device
);

