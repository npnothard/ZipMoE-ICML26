// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
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



#include "model_topology.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <sstream>
#include "utils/logger.hpp"
#include "utils/tqdm.hpp"
#include "common/time.hpp"
#include "common/types.hpp"
#include "memory/cache.hpp"
#include "base/zipmoe_tensor_handle.hpp"
#include "base/zipmoe_tensor_index.hpp"

std::unique_ptr<ZipMoETopologyHandle> kTopologyHandle = nullptr;

const std::string Node::str() {
    std::stringstream ss;
    for (auto& tensor_id : tensor_ids) {
        ss << tensor_id << ",";
    }
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(
        buffer, 
        "ID[%ld,%lx] (%ldMB) STATE(%d) TENSOR[%s] DEVICE[%s;%s;%s];",
        id, 
        corr_id, 
        byte_size / MB, 
        state.load(), 
        ss.str().c_str(),
        device.str().c_str(), 
        default_device.str().c_str(),
        default_host.str().c_str()
    );

    return std::string(buffer);
}

Node::Node()
    : corr_id(0),
      byte_size(0),
      last_access_time(MICROSECONDS_SINCE_EPOCH),
      device(DISK_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE) {}
void Node::SetDevice(
    const torch::Device& target_device,
    uint8_t* gpu_exp_ptr,
    uint8_t* gpu_sm_ptr,
    uint16_t* pending_recover_gpu_ptr,
    cudaStream_t stream
){
    DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str());
    if (device == target_device) {
        DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str() +
               " but device is the same");
        return;
    }
    if (target_device == DISK_DEVICE){
        SetModuleDisk(tensor_ids);
        if (device_memory_ptr != nullptr){
            device_memory_ptr = nullptr;
        }
    } else {
        if (device_memory_ptr==nullptr){
            SetModuleMemoryFromDiskGrouped(
                tensor_ids,
                num_elements,
                gpu_exp_ptr,
                gpu_sm_ptr,
                pending_recover_gpu_ptr,
                stream,
                target_device
            );
            device_memory_ptr = pending_recover_gpu_ptr;
        }
    }
    device = target_device;
}


void Node::SetSM(
    uint8_t* gpu_sm_ptr
){
    device_sm_ptr = gpu_sm_ptr;
}

void Node::PinDense(
    const torch::Device& target_device,
    uint8_t* dense_offset_gpu_ptr
){
    DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str());
    if (device == target_device) {
        DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str() +
               " but device is the same");
        return;
    }
    if (target_device == DISK_DEVICE){
        SetModuleDisk(tensor_ids);
        if (dense_gpu_offset_ptr != nullptr){
            dense_gpu_offset_ptr = nullptr;
        }
    } else {
        if (dense_gpu_offset_ptr == nullptr){
            SetDenseModuleFromDisk(
                tensor_ids,
                dense_offset_gpu_ptr,
                target_device
            );
            dense_gpu_offset_ptr = dense_offset_gpu_ptr;
        }
    }
    device = target_device;
}

ZipMoETopologyHandle::ZipMoETopologyHandle() {}


NodePtrList ZipMoETopologyHandle::GetDenseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (stage->is_sparse) {
      continue;
    }
    for (auto node : stage->nodes) {
      nodes.push_back(node);
    }
  }
  return nodes;
}


NodePtrList ZipMoETopologyHandle::GetSparseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (!stage->is_sparse) {
      continue;
    }
    for (auto node : stage->nodes) {
      nodes.push_back(node);
    }
  }
  return nodes;
}




void ZipMoETopologyHandle::ExamineCompreesionRatio(){
    size_t total_bytes_compressed = 0;
    size_t total_bytes_original = 0;
    for (auto stage : pipeline_.stages ){
        if (!stage->is_sparse) {
            continue;
        }
        for (auto node: stage->nodes){
            total_bytes_original += node->byte_size;
            for (auto tensor_id: node->tensor_ids){
                auto it = kTensorIndex->find(tensor_id);
                if (it != kTensorIndex->end()){
                    // The tensor id is found
                    //int64_t size_aligned = (it->second.size + OS_PAGE_SIZE - 1)&~(OS_PAGE_SIZE - 1);
                    
                    // Compute the compressed size
                    for (auto compressed_size: it->second.compressed_sizes){
                        total_bytes_compressed += compressed_size;
                    }
                    total_bytes_compressed += it->second.sm_size;

                } else {
                    DLOG_ERROR("Tensor {} not found in tensor index", tensor_id);
                }
            }
        }
    }


    double compression_ratio = double(total_bytes_compressed)/double(total_bytes_original);
    DLOG_INFO("[ZIPMOE PROFILE] Compression Ratio is: ", compression_ratio);


}



NodePtrList ZipMoETopologyHandle::GetDenseNodes(
    const NodePtr& node,
    const size_t& k
){
    NodePtrList nodes;
    size_t low_corr_id = node->corr_id & 0xFFFFFFFF;
    size_t high_corr_id = node->corr_id >> 32;
    bool is_last_node = (0xFFFFFFFF == high_corr_id);
    if (is_last_node) {
        high_corr_id = 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    low_corr_id++;
    size_t count = 0;
    while ( (low_corr_id<pipeline_.stages.size())&&(count<k) ){
        auto stage = pipeline_.stages[low_corr_id];
        low_corr_id++;
        if (stage->is_sparse){
            continue;
        }
        nodes.push_back(stage->nodes[0]);
        count++;
    }
    return nodes;
}


NodePtrList ZipMoETopologyHandle::GetSparseNodes(
    const NodePtr& node,
    const size_t& k
){
    NodePtrList nodes;
    size_t low_corr_id = node->corr_id & 0xFFFFFFFF;
    size_t high_corr_id = node->corr_id >> 32;
    bool is_last_node = (0xFFFFFFFF == high_corr_id);
    if (is_last_node) {
        high_corr_id = 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    low_corr_id++;
    size_t count = 0;
    while ( (low_corr_id<pipeline_.stages.size())&&(count<k) ){
        auto stage = pipeline_.stages[low_corr_id];
        low_corr_id++;
        if (!stage->is_sparse) {
            continue;
        }

        nodes.push_back(stage->nodes[0]);
        count++;
    }
    return nodes;      
}


std::uint64_t ZipMoETopologyHandle::GetLastActivateStage(const HashID& hash_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = last_active_stage_.find(hash_id);
  if (it == last_active_stage_.end()) {
    return 0;
  }
  return it->second;
}

bool ZipMoETopologyHandle::IsLastNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto last_stage_ptr = pipeline_.stages.back();
  auto& nodes = last_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n == node) {
      return true;
    }
  }
  return false;
}


bool ZipMoETopologyHandle::IsFirstNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto first_stage_ptr = pipeline_.stages.front();
  auto& nodes = first_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n == node) {
      return true;
    }
  }
  return false;
}

void ZipMoETopologyHandle::InitializeTopology(
    const std::vector<
        std::tuple<
            std::string,
            std::vector<
                std::vector<TensorID>
            >
        >
    >& topology
){
    std::lock_guard<std::mutex> lock(mutex_);
    pipeline_.stages.clear();
    size_t node_id = 0;
    size_t layer_id = 0;
    size_t last_sparse_layer_id = UINT64_MAX;

    size_t num_sparse_layers = 0;
    size_t num_experts = 0;

    std::vector<NodePtr> all_nodes;

    for (auto& stage: topology) {
        auto& stage_tensors = std::get<1>(stage);
        auto stage_ptr = std::make_shared<Stage>(stage_tensors.size() > 1);

        size_t expert_id = 0;
        for (auto& tensor_ids: stage_tensors){
            // Setup node
            auto node_ptr = std::make_shared<Node>();
            node_ptr->tensor_ids = tensor_ids;
            int64_t byte_size = 0;
            int64_t num_elements = 0;
            for (auto& tensor_id: tensor_ids){
                auto it = kTensorIndex->find(tensor_id);
                if (it != kTensorIndex->end()){
                    byte_size += it->second.size;
                    num_elements += it->second.num_elements;
                } else {
                    DLOG_ERROR("Tensor {} not found in tensor index", tensor_id);
                }
            }
            node_ptr->byte_size = byte_size;
            node_ptr->num_elements = num_elements;
            node_ptr->id = node_id;
            node_ptr->corr_id = (layer_id & 0xFFFFFFFF)|((expert_id & 0xFFFFFFFF)<<32);
            node_ptr->is_sparse = stage_ptr->is_sparse;
            all_nodes.push_back(node_ptr);
            stage_ptr->nodes.push_back(node_ptr);
            node_id++;
            expert_id++;
        }
        pipeline_.stages.push_back(stage_ptr);
        auto current_layer_id = layer_id;
        layer_id++;
    }

    auto last_stage_ptr = pipeline_.stages.back();
    for (auto& node : last_stage_ptr->nodes) {
        node->corr_id =
            (node->corr_id & 0xFFFFFFFF) | (UINT64_MAX << 32);
    }

    for (auto& stage : pipeline_.stages) {
        for (auto& node : stage->nodes) {
            std::stringstream ss;
            for (auto& tensor_id : node->tensor_ids) {
                ss << tensor_id << " ";
            }
        }
    }
    DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}",
                pipeline_.stages.size());
    auto num_gpu = GetDeviceCount();
    auto sparse_nodes = GetSparseNodes();
    auto dense_nodes = GetDenseNodes();
    DLOG_TRACE(
        "InitializeTopology num_gpu {} sparse_nodes.size() {} dense_nodes.size() {}",
        num_gpu, sparse_nodes.size(), dense_nodes.size()
    );

    DLOG_INFO("Moving dense parameters to GPU");
    int target_device_id = 0;
    torch::Device gpu0_device = torch::Device(torch::kCUDA, target_device_id);
    size_t dense_offset = 0;

    size_t total_dense_size = GetDenseNodesTotalSize();
    kZipMoECacheHandle->SetUpDensePool(total_dense_size);
    DLOG_INFO("Total dense parameter size: ", total_dense_size);

    if (kZipMoECacheHandle->dense_gpu_ptr_base == nullptr){
        DLOG_FATAL("Dense Node Pool not Set Up !");
    }
    for (auto& node_ptr: tqdm::tqdm(dense_nodes)){
        node_ptr->default_device = gpu0_device;
        uint8_t* dense_offset_gpu_ptr = kZipMoECacheHandle->dense_gpu_ptr_base + dense_offset;
        node_ptr->PinDense(
            node_ptr->default_device,
            dense_offset_gpu_ptr
        );
        dense_offset += node_ptr->byte_size;
    }

    DLOG_INFO("Moving sparse parameters to SSD");
    for (auto& node_ptr: tqdm::tqdm(sparse_nodes)){
        node_ptr->default_device = gpu0_device;
        node_ptr->SetDevice(
            DISK_DEVICE, 
            nullptr,
            nullptr,
            nullptr,
            nullptr
        );
    }

    DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}", pipeline_.stages.size());
    for (auto& node_ptr : all_nodes) {
        DLOG_TRACE("Node {} {} device {}", node_ptr->id, node_ptr->is_sparse, node_ptr->default_device.str());
    }
    ExamineCompreesionRatio();
    EnableTrace();
}

size_t ZipMoETopologyHandle::GetDenseNodesTotalSize(){
    size_t total_size = 0;  
    auto dense_nodes = kTopologyHandle->GetDenseNodes();  
    for (auto& node : dense_nodes) {  
        total_size += node->byte_size;  
    }  
    return total_size;  
}


NodePtr ZipMoETopologyHandle::GetNodeFromTensorID( const TensorID& tensor_id ){
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tensor_id_to_node_.find(tensor_id);
    if (it != tensor_id_to_node_.end()){
        return it->second;
    } else {
        for (auto& stage : pipeline_.stages) {
            for (auto& node : stage->nodes) {
                for (auto& id : node->tensor_ids) {
                    if (id == tensor_id) {
                        tensor_id_to_node_[tensor_id] = node;
                        return node;
                    }
                }
            }
        }        
    }
    DLOG_ERROR("Tensor {} not found in tensor id to node map", tensor_id);
    return nullptr;
}


std::int64_t ZipMoETopologyHandle::GetSparseCacheLimit(
    const torch::Device& device
){
    std::int64_t dense_cache_size = 0;
    for (auto& stage : pipeline_.stages) {
        for (auto& node : stage->nodes) {
            if (stage->is_sparse) continue;
            if (node->device == device) {
                dense_cache_size += node->byte_size;
            }
        }
    }
    std::int64_t sparse_cache_size = kZipMoECacheHandle->sparse_cache_size;
    return sparse_cache_size;
}




std::tuple<std::size_t, std::size_t>
ZipMoETopologyHandle::GetNumLayersAndExperts() {
    std::lock_guard<std::mutex> lock(mutex_);
    int num_layers = 0;
    int num_experts = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            num_layers += 1;
            num_experts = stage->nodes.size();
        }
    }
    return std::make_tuple(num_layers, num_experts);
}


void SetModuleDisk(std::vector<TensorID>& tensor_ids) {
    for (const auto& tensor_id : tensor_ids) {
        auto it = kTensorIndex->find(tensor_id);
        it->second.tensor.set_data(
            kZipMoECacheHandle->cpu_zero_tensor
        );
    }
}

void SetDenseModuleFromDisk(
    std::vector<TensorID>& tensor_ids,
    uint8_t* dense_offset_gpu_ptr,
    const torch::Device& device
){
    size_t offset = 0;
    for (const auto& tensor_id : tensor_ids){
        auto it = kTensorIndex->find(tensor_id);
        uint8_t* target_gpu_ptr = dense_offset_gpu_ptr + offset;
        kZipMoETensorEngine->ReadDense(
            target_gpu_ptr,
            it->second.sm_file_offset,
            it->second.size
        );
        auto tensor_options = torch::TensorOptions()
                                    .dtype(it->second.options.dtype())
                                    .layout(it->second.options.layout())
                                    .device(device)
                                    .requires_grad(it->second.options.requires_grad())
                                    .pinned_memory(false);
        it->second.tensor.set_data(
            torch::from_blob(
                target_gpu_ptr,
                it->second.shape,
                DoNothingDeleter<void>{},
                tensor_options
            )
        );
        offset += it->second.size;
    }

}


void SetModuleMemoryFromDisk(
    std::vector<TensorID>& tensor_ids,
    uint8_t* gpu_exp_ptr,
    uint8_t* gpu_sm_ptr,
    uint16_t* pending_recover_gpu_ptr,
    cudaStream_t stream,
    const torch::Device& device
){

    size_t offset = 0;
    for (const auto& tensor_id : tensor_ids){
        auto it = kTensorIndex->find(tensor_id);
        uint16_t* target_gpu_ptr = pending_recover_gpu_ptr + offset;
        uint8_t* target_pinned_exp = gpu_exp_ptr + offset;
        uint8_t* target_pinned_sm = gpu_sm_ptr + offset;
        zipmoe_launch_tensor_recover(
            target_gpu_ptr,
            target_pinned_exp,
            target_pinned_sm,
            it->second.num_elements,
            stream
        );
        auto tensor_options = torch::TensorOptions()
                                    .dtype(it->second.options.dtype())
                                    .layout(it->second.options.layout())
                                    .device(device)
                                    .requires_grad(it->second.options.requires_grad())
                                    .pinned_memory(false);
        it->second.tensor.set_data(
            torch::from_blob(
                target_gpu_ptr,
                it->second.shape,
                DoNothingDeleter<void>{},
                tensor_options
            )
        );
        offset += it->second.num_elements;
    }
    cudaStreamSynchronize(stream);
}


void SetModuleMemoryFromDiskGrouped(
    std::vector<TensorID>& tensor_ids,
    size_t node_num_elements,
    uint8_t* gpu_exp_ptr,
    uint8_t* gpu_sm_ptr,
    uint16_t* pending_recover_gpu_ptr,
    cudaStream_t stream,
    const torch::Device& device
){
    zipmoe_launch_tensor_recover(
        pending_recover_gpu_ptr,
        gpu_exp_ptr,
        gpu_sm_ptr,
        node_num_elements,
        stream
    );
    cudaStreamSynchronize(stream);
    size_t offset = 0;
    for (const auto& tensor_id : tensor_ids){
        auto it = kTensorIndex->find(tensor_id);
        uint16_t* target_gpu_ptr = pending_recover_gpu_ptr + offset;
        auto tensor_options = torch::TensorOptions()
                                    .dtype(it->second.options.dtype())
                                    .layout(it->second.options.layout())
                                    .device(device)
                                    .requires_grad(it->second.options.requires_grad())
                                    .pinned_memory(false);
        it->second.tensor.set_data(
            torch::from_blob(
                target_gpu_ptr,
                it->second.shape,
                DoNothingDeleter<void>{},
                tensor_options
            )
        );
        offset += it->second.num_elements;
    }

}