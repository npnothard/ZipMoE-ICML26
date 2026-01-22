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


#include "zipmoe_tensor_index.hpp"
#include <stdio.h>
using namespace std;

void write_options(
    ostream& os,
    const torch::TensorOptions& obj
){
    bool pinned_memory = obj.pinned_memory();
    bool requires_grad = obj.requires_grad();
    int8_t dtype = static_cast<int8_t>(obj.dtype().toScalarType());
    int8_t device_index = static_cast<int8_t>(obj.device().index());
    int8_t device_type = static_cast<int8_t>(obj.device().type());
    int8_t layout = static_cast<int8_t>(obj.layout());

    os.write(reinterpret_cast<char*>(&pinned_memory), sizeof(pinned_memory));
    os.write(reinterpret_cast<char*>(&requires_grad), sizeof(requires_grad));
    os.write(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    os.write(reinterpret_cast<char*>(&device_index), sizeof(device_index));
    os.write(reinterpret_cast<char*>(&device_type), sizeof(device_type));
    os.write(reinterpret_cast<char*>(&layout), sizeof(layout));
}

void read_options(istream& is, torch::TensorOptions& obj) {
  bool pinned_memory = obj.pinned_memory();
  bool requires_grad = obj.requires_grad();
  int8_t dtype = static_cast<int8_t>(obj.dtype().toScalarType());
  int8_t device_index = static_cast<int8_t>(obj.device().index());
  int8_t device_type = static_cast<int8_t>(obj.device().type());
  int8_t layout = static_cast<int8_t>(obj.layout());

  is.read(reinterpret_cast<char*>(&pinned_memory), sizeof(pinned_memory));
  is.read(reinterpret_cast<char*>(&requires_grad), sizeof(requires_grad));
  is.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
  is.read(reinterpret_cast<char*>(&device_index), sizeof(device_index));
  is.read(reinterpret_cast<char*>(&device_type), sizeof(device_type));
  is.read(reinterpret_cast<char*>(&layout), sizeof(layout));

  obj =
      obj.dtype(static_cast<c10::ScalarType>(dtype))
          .device(torch::Device(static_cast<torch::DeviceType>(device_type),
                                static_cast<torch::DeviceIndex>(device_index)))
          .layout(static_cast<c10::Layout>(layout))
          .requires_grad(requires_grad)
          .pinned_memory(pinned_memory);
}

ostream& operator<<(ostream& os, const TensorStorageMeta& obj) {
    os.write(reinterpret_cast<const char*>(&obj.id), sizeof(obj.id));
    int64_t shape_size = obj.shape.size();
    os.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    for (auto dim : obj.shape) {
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    os.write(reinterpret_cast<const char*>(&obj.size), sizeof(obj.size));
    os.write(reinterpret_cast<const char*>(&obj.num_elements), sizeof(obj.num_elements));
    write_options(os, obj.options);
    os.write(reinterpret_cast<const char*>(&obj.offload_file_id),
             sizeof(obj.offload_file_id));
    os.write(reinterpret_cast<const char*>(&obj.num_file_chunks),
             sizeof(obj.num_file_chunks));
    uint32_t exp_size = obj.exp_file_offsets.size();
    os.write(reinterpret_cast<const char*>(&exp_size), sizeof(exp_size));
    for (auto x : obj.exp_file_offsets) {
        os.write(reinterpret_cast<const char*>(&x), sizeof(x));
    }
    uint32_t comp_size = obj.compressed_sizes.size();
    os.write(reinterpret_cast<const char*>(&comp_size), sizeof(comp_size));
    for (auto x : obj.compressed_sizes) {
        os.write(reinterpret_cast<const char*>(&x), sizeof(x));
    }
    uint32_t orig_size = obj.original_sizes.size();
    os.write(reinterpret_cast<const char*>(&orig_size), sizeof(orig_size));
    for (auto x : obj.original_sizes) {
        os.write(reinterpret_cast<const char*>(&x), sizeof(x));
    }
    os.write(reinterpret_cast<const char*>(&obj.sm_file_offset),
             sizeof(obj.sm_file_offset));
    os.write(reinterpret_cast<const char*>(&obj.sm_size),
             sizeof(obj.sm_size));
    return os;
}


istream& operator>>(istream& is, TensorStorageMeta& obj) {
    is.read(reinterpret_cast<char*>(&obj.id), sizeof(obj.id));
    int64_t shape_size;
    is.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    obj.shape.resize(shape_size);
    for (auto& dim : obj.shape) {
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }
    is.read(reinterpret_cast<char*>(&obj.size), sizeof(obj.size));
    is.read(reinterpret_cast<char*>(&obj.num_elements), sizeof(obj.num_elements));
    read_options(is, obj.options);
    is.read(reinterpret_cast<char*>(&obj.offload_file_id),
            sizeof(obj.offload_file_id));
    is.read(reinterpret_cast<char*>(&obj.num_file_chunks),
            sizeof(obj.num_file_chunks));
    uint32_t exp_size;
    is.read(reinterpret_cast<char*>(&exp_size), sizeof(exp_size));
    obj.exp_file_offsets.resize(exp_size);
    for (auto& x : obj.exp_file_offsets) {
        is.read(reinterpret_cast<char*>(&x), sizeof(x));
    }
    uint32_t comp_size;
    is.read(reinterpret_cast<char*>(&comp_size), sizeof(comp_size));
    obj.compressed_sizes.resize(comp_size);
    for (auto& x : obj.compressed_sizes) {
        is.read(reinterpret_cast<char*>(&x), sizeof(x));
    }
    uint32_t orig_size;
    is.read(reinterpret_cast<char*>(&orig_size), sizeof(orig_size));
    obj.original_sizes.resize(orig_size);
    for (auto& x : obj.original_sizes) {
        is.read(reinterpret_cast<char*>(&x), sizeof(x));
    }
    is.read(reinterpret_cast<char*>(&obj.sm_file_offset),
            sizeof(obj.sm_file_offset));
    is.read(reinterpret_cast<char*>(&obj.sm_size),
            sizeof(obj.sm_size));
    return is;
}


string TensorStorageMeta::DebugString() const {
    stringstream ss;
    ss << "file_id: " << offload_file_id << ", num_file_chunks: " << num_file_chunks << ", shape: [";
    for (auto& dim : shape) {
        ss << dim << ", ";
    }
    ss << "], id: " << id;
    return ss.str();
}


std::unique_ptr<ZipMoETensorIndex> kTensorIndex(nullptr);



void ZipMoETensorIndex::Serialize(const char* path) {
    std::uint32_t size = this->size();
    std::ofstream ofs(path, std::ios::binary | std::ios::out | std::ios::trunc);
    ofs.write(reinterpret_cast<char*>(&size), sizeof(size));
    for (auto& item : *this) {
        ofs.write(reinterpret_cast<const char*>(&item.first), sizeof(item.first));
        ofs << item.second;
    }
}


void ZipMoETensorIndex::Deserialize(const char* path) {
    
    this->clear();

    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    std::uint32_t size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));

    for (std::uint32_t i = 0; i < size; ++i) {
        std::uint32_t key;
        ifs.read(reinterpret_cast<char*>(&key), sizeof(key));
        TensorStorageMeta value;
        ifs >> value;
        this->insert({key, value});
    }
}
