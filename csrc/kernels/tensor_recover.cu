// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "tensor_recover.hpp"


__global__ void cuda_recover_uint16_to_bf16(
    uint16_t* __restrict__ output,  
    const uint8_t* __restrict__ exp_bits, 
    const uint8_t* __restrict__ sign_mantissa, 
    const int n  
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < n) {
        uint64_t sm_vec = *reinterpret_cast<const uint64_t*>(&sign_mantissa[idx]);
        uint64_t exp_vec = *reinterpret_cast<const uint64_t*>(&exp_bits[idx]);
        uint16_t results[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t sm = (sm_vec >> (i * 8)) & 0xFF;
            uint8_t s  = (sm >> 7) & 0x1;
            uint8_t m  = sm & 0x7F;
            uint8_t e  = (exp_vec >> (i * 8)) & 0xFF;

            results[i] = ((uint16_t)s << 15) | ((uint16_t)e << 7) | ((uint16_t)m);
        }
        *reinterpret_cast<uint64_t*>(&output[idx])      = *reinterpret_cast<uint64_t*>(&results[0]);
        *reinterpret_cast<uint64_t*>(&output[idx + 4])  = *reinterpret_cast<uint64_t*>(&results[4]);
    } else {
        for (int i = idx; i < n; ++i) {
            uint8_t sm = sign_mantissa[i];
            uint8_t s = (sm >> 7) & 0x1;
            uint8_t m = sm & 0x7F;
            uint8_t e = exp_bits[i];
            output[i] = ((uint16_t)s << 15) | ((uint16_t)e << 7) | ((uint16_t)m);
        }
    }
}



#define threads_per_block 512
#define elements_per_thread 8
void zipmoe_launch_tensor_recover(
    uint16_t* gpu_output_tensor, 
    const uint8_t* gpu_exp_ptr,
    const uint8_t* gpu_sm_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
){
    int blocks_per_grid = (num_elements + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);
    cuda_recover_uint16_to_bf16<<<blocks_per_grid, threads_per_block, 0, tensor_stream>>>(
        gpu_output_tensor, gpu_exp_ptr, gpu_sm_ptr, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}