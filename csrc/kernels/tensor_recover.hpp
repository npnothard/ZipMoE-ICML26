// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>


void zipmoe_launch_tensor_recover(
    uint16_t* gpu_output_tensor, 
    const uint8_t* gpu_sm_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
);
