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


#include <torch/torch.h>
#include "topology/model_topology.hpp"

#ifndef EXPERT_TYPE
  #define EXPERT_TYPE 0
#endif
#define DEEPSEEK_MOE_DENSE_ACT_DENSE 0
#define SWITCH_TRANSFORMERS_DENSE_ACT_DENSE 1

#define DTYPE_BFLOAT16 0

struct ModuleUtils {
    virtual void SetTensorsFromBlob(
    void* ptr,
    const std::vector<std::uint32_t>& tensor_ids,
    const torch::Device& device) = 0;
};

struct DeepSeekMoEDenseActDense : public torch::nn::Module, public ModuleUtils {
	DeepSeekMoEDenseActDense(int dtype);
	torch::Tensor forward(torch::Tensor hidden_states);
	torch::Tensor gate_proj, up_proj, down_proj;

	void SetTensorsFromBlob(
		void* ptr,
		const std::vector<std::uint32_t>& tensor_ids,
		const torch::Device& device) override;
};

struct SwitchTransformersDenseActDense : public torch::nn::Module,
                                         public ModuleUtils {
  SwitchTransformersDenseActDense(int dtype);
  torch::Tensor forward(torch::Tensor hidden_states);
  torch::Tensor wi, wo;

  void SetTensorsFromBlob(void* ptr,
                          const std::vector<std::uint32_t>& tensor_ids,
                          const torch::Device& device) override;
};

struct ExpertNode {
	NodePtr node;
	torch::nn::Module* module;
	void SetTensorsFromBlob(const torch::Device& device);
	int layer_idx;
	int expert_idx;
	int expert_type;
};

typedef std::shared_ptr<ExpertNode> ExpertNodePtr;

inline torch::ScalarType dtype_to_torch(int dtype) {
	auto tensor_dtype = torch::kFloat32;
	switch (dtype) {
		case DTYPE_BFLOAT16:
		tensor_dtype = torch::kBFloat16;
		break;
		default:
		assert(false);
	}
	return tensor_dtype;
}

inline int torch_dtype_to_int(torch::ScalarType dtype) {
	auto tensor_dtype = DTYPE_BFLOAT16;
	switch (dtype) {
		case torch::kBFloat16:
		tensor_dtype = DTYPE_BFLOAT16;
		break;
		default:
		assert(false);
	}
	return tensor_dtype;
}
