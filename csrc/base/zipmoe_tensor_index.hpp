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
#include <torch/extension.h>
#include <torch/script.h>
#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include "noncopyable.hpp"
#include "common/pytorch.hpp"
#include "common/types.hpp"

static const std::uint32_t kTensorIndexVersion = 1;


struct TensorStorageMeta {
    TensorID id;
    std::vector<std::int64_t> shape;
    size_t size;
    size_t num_elements;
    torch::TensorOptions options;
    uint32_t offload_file_id;
    int num_file_chunks;
    std::vector<size_t> exp_file_offsets;
    std::vector<size_t> compressed_sizes;
    std::vector<size_t> original_sizes;
    size_t sm_file_offset;
    size_t sm_size; 
    torch::Tensor tensor;
    torch::Device device = DISK_DEVICE;
    std::string DebugString() const;
};

std::ostream& operator<<(std::ostream& os, const TensorStorageMeta& obj);
std::istream& operator>>(std::istream& is, TensorStorageMeta& obj);
void write_options(std::ostream& os, const torch::TensorOptions& obj);
void read_options(std::istream& is, torch::TensorOptions& obj);

class ZipMoETensorIndex
    : public std::unordered_map<uint32_t, TensorStorageMeta>,
      public base::noncopyable {
 public:
    void Serialize(const char* path);
    void Deserialize(const char* path);

    ZipMoETensorIndex() = default;
    ~ZipMoETensorIndex() = default;
    std::mutex index_mutex;
 private:
};

extern std::unique_ptr<ZipMoETensorIndex> kTensorIndex;
