// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once


#include <iostream>
#include <vector>
#include <memory>
#include <lz4.h>
#include <lz4hc.h>
#include <zstd.h>
#include "thread_pool.hpp"

#include <nvtx3/nvToolsExt.h>

class ZSTDcompressor{

    private:
        std::vector<ZSTD_CCtx*> cctx_vec;
        std::vector<ZSTD_DCtx*> dctx_vec;
    public:
        int num_threads;
        int compressionLevel;
        PreemptiveThreadPool& thread_pool;
        ZSTDcompressor(
            int num_threads,
            int compressionLevel,
            PreemptiveThreadPool& thread_pool
        );

        ZSTDcompressor(const ZSTDcompressor&) = delete;
        ZSTDcompressor& operator=(const ZSTDcompressor&) = delete;
        ZSTDcompressor(ZSTDcompressor&&) = default;
        ZSTDcompressor& operator=(ZSTDcompressor&&) = default;


        std::unique_ptr<uint8_t[]> get_compressed_buffer(
            const size_t& input_size,
            size_t& compress_bound
        );

        void compress(
            const void* uncompressed_ptr,
            size_t& uncompressed_size,
            size_t& compress_bound,
            void* compressed_ptr,
            size_t& compressed_size
        );

        std::unique_ptr<uint8_t[]> get_decompressed_buffer(
            const size_t& original_size
        );

        void decompress(
            uint8_t* compressed_ptr,
            size_t& compressed_size,
            size_t& original_size,
            uint8_t* decompressed_ptr
        );
};


class LZ4HCcompressor{

    public:
        int compressionLevel;
        LZ4HCcompressor(
            int compressionLevel
        );
        LZ4HCcompressor(const LZ4HCcompressor&) = delete;
        LZ4HCcompressor& operator=(const LZ4HCcompressor&) = delete;
        LZ4HCcompressor(LZ4HCcompressor&&) = default;
        LZ4HCcompressor& operator=(LZ4HCcompressor&&) = default;
        std::unique_ptr<uint8_t[]> get_compressed_buffer(
            const size_t& input_size,
            size_t& compress_bound
        );
        void compress(
            void* uncompressed_ptr,
            size_t& uncompressed_size,
            size_t& compress_bound,
            void* compressed_ptr,
            size_t& compressed_size
        );
        std::unique_ptr<uint8_t[]> get_decompressed_buffer(
            const size_t& original_size
        );
        void decompress(
            uint8_t* compressed_ptr,
            size_t& compressed_size,
            size_t& original_size,
            uint8_t* decompressed_ptr
        );
};


class LZ4compressor{
    public:
        int accelerationLevel;
        LZ4compressor(
            int accelerationLevel
        );
        LZ4compressor(const LZ4compressor&) = delete;
        LZ4compressor& operator=(const LZ4compressor&) = delete;
        LZ4compressor(LZ4compressor&&) = default;
        LZ4compressor& operator=(LZ4compressor&&) = default;
        std::unique_ptr<uint8_t[]> get_compressed_buffer(
            const size_t& input_size,
            size_t& compress_bound
        );
        void compress(
            void* uncompressed_ptr,
            size_t& uncompressed_size,
            size_t& compress_bound,
            void* compressed_ptr,
            size_t& compressed_size
        );
        std::unique_ptr<uint8_t[]> get_decompressed_buffer(
            const size_t& original_size
        );
        void decompress(
            uint8_t* compressed_ptr,
            size_t& compressed_size,
            size_t& original_size,
            uint8_t* decompressed_ptr
        );

};


