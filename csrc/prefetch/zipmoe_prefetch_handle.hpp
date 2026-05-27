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



# pragma once

#include "base/zipmoe_tensor_handle.hpp"
#include "../topology/model_topology.hpp"
#include "dispatcher/expert_dispatcher.hpp"
#include "../tensor_engine/tensor_engine.hpp"

class ZipMoEPrefetchHandle{

public:

    ZipMoEPrefetchHandle(
        const std::string& prefix,
        const std::string& offload_file_name,
        const std::string& CODE_TYPE,
        const std::string& caching_algorithm,
        const double& device_memory_ratio,
        const double& gpu_pool_ratio,
        double decompression_delay,
        double sm_io_delay,
        int num_compute_threads,
        int num_file_chunks,
        int prefetcher_topk,
        int expert_topk,
        size_t num_elements_per_expert,
        size_t num_tensors_per_expert,
        int num_expert_layers,
        //int num_layers,
        int num_experts,
        //int first_k_dense_replace,
        int LZ4_accelerationLevel,
        int LZ4HC_compressionLevel,
        int ZSTD_compressionLevel,
        double hyperparam_state_margin,
        bool bind_core
    );
    ~ZipMoEPrefetchHandle();
    bool IsTensorOffloaded(const uint32_t tensor_id);
    void SetUpDensePool();
    void OpenOffloadFile();
    uint32_t AcquireTensor(uint64_t& request_id, torch::Tensor& buffer);
    void ReleaseTensor(uint64_t& request_id, torch::Tensor& buffer);
    void FetchTensors(
        uint64_t& request_id,
        const std::vector<uint32_t>& buffer
    );
    void OffloadTensor(
        const std::uint32_t tensor_id,
        torch::Tensor& tensor,
        const std::vector<py::array_t<uint8_t>>& exponents_chunks,
        const py::array_t<uint8_t>& sign_mantissa,
        bool is_sparse
    );

    void BatchOffloadTensor(
        const std::vector<std::uint32_t> tensor_ids,
        std::vector<torch::Tensor>& tensors,
        const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
        const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
    );

    void RegisterTensor(
        torch::Tensor& tensor,
        const uint32_t tensor_id
    );

    int GetNodeDefaultDevice(std::vector<uint32_t> tensor_ids) const;
    void ResetAccessCounts();
    void SetTrace(const torch::Tensor& trace);
    void TraceRequest(const uint64_t request_id, const TensorID tensor_id);
    void SetTopology(
        const std::vector<
            std::tuple<
                std::string, 
                std::vector<std::vector<TensorID>>
            >
        >&  topology
    );
    void UpdateTensorMap(std::uint64_t old_ptr, std::uint64_t new_ptr);
    bool IsTensorIndexInitialized() const;
    void CleanUpResources();

private:

    std::string prefix_;
    std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>> node_id_to_tensor_ids_;
    std::unordered_set<std::uint32_t> tensors_to_delete_;
    uint64_t last_layer_id_;
    NodePtr last_node_;
    bool has_cleaned_up_resources_;
    std::unordered_map<std::uint64_t, std::unordered_set<NodePtr>> request_id_to_nodes_;
    std::mutex mutex_;

};