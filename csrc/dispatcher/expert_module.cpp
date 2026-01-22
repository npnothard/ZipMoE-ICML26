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


#include "expert_module.hpp"
#include "base/zipmoe_tensor_handle.hpp"
#include "utils/logger.hpp"

DeepSeekMoEDenseActDense::DeepSeekMoEDenseActDense(int dtype) {
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
    gate_proj = register_parameter("gate_proj", torch::zeros({1}, options));
    up_proj = register_parameter("up_proj", torch::zeros({1}, options));
    down_proj = register_parameter("down_proj", torch::zeros({1}, options));
}

void DeepSeekMoEDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device
){
    gate_proj = kTensorIndex->find(tensor_ids[0])->second.tensor;
    up_proj = kTensorIndex->find(tensor_ids[1])->second.tensor;
    down_proj = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

torch::Tensor DeepSeekMoEDenseActDense::forward(torch::Tensor hidden_states) {
    return torch::matmul(
        torch::silu(
            torch::matmul(
                hidden_states, gate_proj.transpose(0, 1)
            )
        ) * torch::matmul(
            hidden_states, up_proj.transpose(0, 1)
        ),
        down_proj.transpose(0, 1)
    );
}


SwitchTransformersDenseActDense::SwitchTransformersDenseActDense(int dtype) {
  auto options = torch::TensorOptions().device(torch::kCPU);
  wi = register_parameter("wi", torch::zeros({1}, options));
  wo = register_parameter("wo", torch::zeros({1}, options));
}

void SwitchTransformersDenseActDense::SetTensorsFromBlob(
    void* ptr, const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) {
  wi = kTensorIndex->find(tensor_ids[0])->second.tensor;
  wo = kTensorIndex->find(tensor_ids[1])->second.tensor;
}


torch::Tensor SwitchTransformersDenseActDense::forward(
    torch::Tensor hidden_states) {
  return torch::matmul(
      torch::relu(
        torch::matmul(
          hidden_states, 
          wi.transpose(0, 1).to(hidden_states.dtype())
        )
      ),
      wo.transpose(0, 1).to(hidden_states.dtype())
  );
}


void ExpertNode::SetTensorsFromBlob(const torch::Device& device) {
    int expert_type = this->expert_type;
    switch (expert_type) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
            reinterpret_cast<SwitchTransformersDenseActDense*>(module)->SetTensorsFromBlob(
                node->device_memory_ptr, 
                node->tensor_ids,
                device
            );
        break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
            reinterpret_cast<DeepSeekMoEDenseActDense*>(module)->SetTensorsFromBlob(
                node->device_memory_ptr, 
                node->tensor_ids, 
                device
            );
            break;
        default:
        assert(false);
    }
}
