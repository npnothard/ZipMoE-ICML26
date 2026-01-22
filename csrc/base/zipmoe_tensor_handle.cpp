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



#include "zipmoe_tensor_handle.hpp"
#include "common/pytorch.hpp"
#include "prefetch/task_scheduler.hpp"
#include "utils/logger.hpp"



const char* ZIPMOE_PARAM_NAME = "zipmoe_param";
const char* ZIPMOE_INDEX_NAME = "zipmoe_index";

std::unique_ptr<ZipMoETensorHandle> kZipMoETensorHandle(nullptr);

ZipMoETensorHandle::ZipMoETensorHandle(const std::string& prefix)
:prefix_(prefix), file_id_(0), file_offset_(0) {
    if (prefix_.back() != '/'){ prefix_ += '/';}
    struct stat st;
    if (stat(prefix_.c_str(), &st) != -1 && !S_ISDIR(st.st_mode)) {
        DLOG_FATAL("Invalid prefix: ", prefix_, " is not a directory");
    }
    if (stat(prefix_.c_str(), &st) == -1) {
        DLOG_WARN("Invalid prefix: ", prefix_, " does not exist, creating");
        mkdir(prefix_.c_str(), 0777);
    }
    DLOG_TRACE("Aio alignment size ", st.st_blksize);
    auto ckpt_index_path = prefix_ + std::string(ZIPMOE_INDEX_NAME);
    if (access(ckpt_index_path.c_str(), F_OK) != -1){
        DLOG_INFO("Loading index file from ", ckpt_index_path);
        kTensorIndex->Deserialize(ckpt_index_path.c_str());
        is_serialized_ = true;
    } else {
         DLOG_INFO("Index file", ckpt_index_path, " does not exist, creating");
    }
    DLOG_INFO("Index file size ", kTensorIndex->size());
}


void ZipMoETensorHandle::StoreTensor(
    const std::uint32_t tensor_id,
    torch::Tensor& buffer,
    const std::vector<py::array_t<uint8_t>>& exponents_chunks,
    const py::array_t<uint8_t>& sign_mantissa,
    bool is_sparse
){
    auto it = kTensorIndex->find(tensor_id);
    bool tensor_exists = (it != kTensorIndex->end());
    TensorStorageMeta tensor_meta;
    std::unique_lock<std::mutex> lock(mutex_);
    if (tensor_exists) {
        DLOG_FATAL("Tensor {} is offloaded twice ! ", tensor_id);
    }
    tensor_meta.id = tensor_id;
    tensor_meta.shape = buffer.sizes().vec();
    tensor_meta.size = buffer.nbytes();
    size_t product = 1;
    for (uint64_t v : tensor_meta.shape) {
        product *= static_cast<size_t>(v);
    }
    tensor_meta.num_elements = product;
    tensor_meta.options = buffer.options();
    tensor_meta.offload_file_id = file_id_;
    tensor_meta.num_file_chunks = kZipMoETensorEngine->config_ptr->num_file_chunks;
    tensor_meta.exp_file_offsets = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
    tensor_meta.compressed_sizes = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
    tensor_meta.original_sizes = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
    tensor_meta.sm_file_offset = 0;
    tensor_meta.sm_size = 0;
    kTensorIndex->insert({tensor_id, tensor_meta});
    lock.unlock();
    if (is_sparse){
        kZipMoETensorEngine->PipelineCompress(
            exponents_chunks,
            sign_mantissa,
            tensor_id
        );
    } else {
        kZipMoETensorEngine->OffloadDense(
            buffer,
            tensor_id
        );
    }
}


void ZipMoETensorHandle::BatchStoreTensor(
    const std::vector<std::uint32_t> tensor_ids,
    std::vector<torch::Tensor>& buffers,
    const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
    const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
){

    size_t idx = 0;

    for (auto tensor_id: tensor_ids){
        auto it = kTensorIndex->find(tensor_id);
        bool tensor_exists = (it != kTensorIndex->end());
        TensorStorageMeta tensor_meta;
        std::unique_lock<std::mutex> lock(mutex_);
        if (tensor_exists) {
            DLOG_FATAL("Tensor {} is offloaded twice ! ", tensor_id);
        }
        tensor_meta.id = tensor_id;
        tensor_meta.shape = buffers[idx].sizes().vec();
        tensor_meta.size = buffers[idx].nbytes();
        size_t product = 1;
        for (uint64_t v : tensor_meta.shape) {
            product *= static_cast<size_t>(v);
        }
        tensor_meta.num_elements = product;
        tensor_meta.options = buffers[idx].options();
        tensor_meta.offload_file_id = file_id_;
        tensor_meta.num_file_chunks = kZipMoETensorEngine->config_ptr->num_file_chunks;
        tensor_meta.exp_file_offsets = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
        tensor_meta.compressed_sizes = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
        tensor_meta.original_sizes = std::vector<size_t>(kZipMoETensorEngine->config_ptr->num_file_chunks);
        tensor_meta.sm_file_offset = 0;
        tensor_meta.sm_size = 0;
        kTensorIndex->insert({tensor_id, tensor_meta});
        lock.unlock();
        idx+=1;
    }
    kZipMoETensorEngine->BatchPipelineCompress(
        tensor_ids,
        batch_exponents_chunks,
        batch_sign_mantissa
    );

}


int64_t ZipMoETensorHandle::GetTensorSizeAligned(
    const uint32_t tensor_id
) const {
    // Search for tensor in the map
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found", tensor_id);
    }
    auto num_bytes = it->second.size;
    int64_t num_bytes_aligned = (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);
    return num_bytes_aligned;
}


torch::TensorOptions ZipMoETensorHandle::GetTensorOptions(
    const uint32_t tensor_id
) const {
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found", tensor_id);
    }
    return it->second.options;
}

void ZipMoETensorHandle::SetTensor(
    uint32_t tensor_id,
    torch::Tensor& buffer,
    const torch::Device& device
){
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found", tensor_id);
    }
    buffer.set_data(it->second.tensor.to(device).to(buffer.dtype()));
}


void ZipMoETensorHandle::SetTensor(
    uint32_t tensor_id,
    torch::Tensor& buffer
){
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found", tensor_id);
    }
    if (buffer.dtype() != it->second.tensor.dtype()) {
        std::ostringstream oss;
        oss << buffer.dtype() << " -> " << it->second.tensor.dtype();
        DLOG_TRACE("Tensor dtype mismatch", tensor_id, oss.str());
        buffer.set_data(it->second.tensor.to(buffer.dtype()));
    } else {
        buffer.set_data(it->second.tensor);
    }
    DLOG_TRACE("Set tensor to device", tensor_id, buffer.device().str());
}


void ZipMoETensorHandle::RegisterTensor(
    const uint32_t tensor_id,
    torch::Tensor& buffer
){
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found", tensor_id);
    }

    tensor_to_id_.insert(std::make_pair((void*)buffer.data_ptr(), tensor_id));

    kTensorIndex->find(tensor_id)->second.tensor = buffer;
}


std::string ZipMoETensorHandle::GetIndexFileName(
        const uint32_t file_id
) const {
    return prefix_ + std::string(ZIPMOE_PARAM_NAME) + "_" + std::to_string(file_id);
}


uint32_t ZipMoETensorHandle::GetTensorId(void* tensor) const {
    auto it = tensor_to_id_.find(tensor);
    if (it == tensor_to_id_.end()) {
        DLOG_FATAL("Tensor not found", (void*)tensor);
        return UINT32_MAX;
    }
    return it->second;
}



void ZipMoETensorHandle::UpdateTensorMap(void* old_data_ptr,
                                         void* new_data_ptr) {
    auto it = tensor_to_id_.find(old_data_ptr);
    if (it == tensor_to_id_.end()) {
        DLOG_FATAL("Tensor ", (void*)old_data_ptr, " not found in tensor_to_id_");
        return;
    }
    auto tensor_id = it->second;
    tensor_to_id_.erase(it);
    auto it2 = kTensorIndex->find(tensor_id);
    if (it2 == kTensorIndex->end()) {
        DLOG_FATAL("Tensor not found in tensor_index_", tensor_id);
        return;
    }
    tensor_to_id_.insert(std::make_pair(new_data_ptr, tensor_id));
}


