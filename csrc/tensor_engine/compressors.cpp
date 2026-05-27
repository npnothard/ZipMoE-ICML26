// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "compressors.hpp"


ZSTDcompressor::ZSTDcompressor(
    int num_threads,
    int compressionLevel,
    PreemptiveThreadPool& thread_pool
):
    num_threads(num_threads),
    compressionLevel(compressionLevel),
    thread_pool(thread_pool)
{
    for (int i=0; i<num_threads; i++){
        ZSTD_CCtx* cctx =  ZSTD_createCCtx();
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        if (!cctx || !dctx){
            throw std::runtime_error("[ZSTD] Failed create context!");
        }
        cctx_vec.emplace_back( cctx );
        dctx_vec.emplace_back( dctx );
    }

}

std::unique_ptr<uint8_t[]> ZSTDcompressor::get_compressed_buffer(
    const size_t& input_size,
    size_t& compress_bound
){
    compress_bound = ZSTD_compressBound(input_size);
    std::unique_ptr<uint8_t[]> compressed_ptr(new uint8_t[compress_bound]);
    return compressed_ptr;
}

void ZSTDcompressor::compress(
    const void* uncompressed_ptr,
    size_t& uncompressed_size,
    size_t& compress_bound,
    void* compressed_ptr,
    size_t& compressed_size
){
    int thread_idx = thread_pool.get_thread_idx();
    compressed_size = ZSTD_compressCCtx(
        cctx_vec[thread_idx],
        compressed_ptr,
        compress_bound,
        uncompressed_ptr,
        uncompressed_size,
        compressionLevel
    );
    if (ZSTD_isError(compressed_size)){
        throw std::runtime_error("[ZSTD] Compression Failed!");
    }

}

std::unique_ptr<uint8_t[]> ZSTDcompressor::get_decompressed_buffer(
    const size_t& original_size
){
    std::unique_ptr<uint8_t[]> decompressed_ptr(new uint8_t[original_size]);
    return decompressed_ptr;
}

void ZSTDcompressor::decompress(
    uint8_t* compressed_ptr,
    size_t& compressed_size,
    size_t& original_size,
    uint8_t* decompressed_ptr
){
    int thread_idx = thread_pool.get_thread_idx();
    size_t const decompressed_size = ZSTD_decompressDCtx(
        dctx_vec[thread_idx],
        decompressed_ptr,
        original_size,
        compressed_ptr,
        compressed_size
    );
    if (ZSTD_isError(decompressed_size)){
        throw std::runtime_error("[ZSTD] Decompression failed!");
    }
}

LZ4HCcompressor::LZ4HCcompressor(
    int compressionLevel
): compressionLevel(compressionLevel)
{}


std::unique_ptr<uint8_t[]> LZ4HCcompressor::get_compressed_buffer(
    const size_t& input_size,
    size_t& compress_bound
){
    compress_bound = LZ4_compressBound(input_size);
    std::unique_ptr<uint8_t[]> compressed_ptr(new uint8_t[compress_bound]);
    return compressed_ptr;
}

void LZ4HCcompressor::compress(
    void* uncompressed_ptr,
    size_t& uncompressed_size,
    size_t& compress_bound,
    void* compressed_ptr,
    size_t& compressed_size
){
    compressed_size = LZ4_compress_HC(
        reinterpret_cast<char*>(uncompressed_ptr),
        reinterpret_cast<char*>(compressed_ptr),
        uncompressed_size,
        compress_bound,
        compressionLevel
    );
    if (compressed_size<=0){
        throw std::runtime_error("[LZ4HC] Compression Failed!");
    }
}


std::unique_ptr<uint8_t[]> LZ4HCcompressor::get_decompressed_buffer(
    const size_t& original_size
){
    std::unique_ptr<uint8_t[]> decompressed_ptr(new uint8_t[original_size]);
    return decompressed_ptr;
}


void LZ4HCcompressor::decompress(
    uint8_t* compressed_ptr,
    size_t& compressed_size,
    size_t& original_size,
    uint8_t* decompressed_ptr
){
    int decompressed_size = LZ4_decompress_safe_partial(
        reinterpret_cast<char*>(compressed_ptr),
        reinterpret_cast<char*>(decompressed_ptr),
        compressed_size,
        original_size,
        original_size
    );
    if (decompressed_size<=0){
        throw std::runtime_error("[LZ4HC] Decompression failed!");
    }
}


LZ4compressor::LZ4compressor(
    int accelerationLevel
): accelerationLevel(accelerationLevel)
{}


std::unique_ptr<uint8_t[]> LZ4compressor::get_compressed_buffer(
    const size_t& input_size,
    size_t& compress_bound
){
    compress_bound = LZ4_compressBound(input_size);
    std::unique_ptr<uint8_t[]> compressed_ptr(new uint8_t[compress_bound]);
    return compressed_ptr;
}

void LZ4compressor::compress(
    void* uncompressed_ptr,
    size_t& uncompressed_size,
    size_t& compress_bound,
    void* compressed_ptr,
    size_t& compressed_size
){
    compressed_size = LZ4_compress_fast(
        reinterpret_cast<char*>(uncompressed_ptr),
        reinterpret_cast<char*>(compressed_ptr),
        uncompressed_size,
        compress_bound,
        accelerationLevel
    );
    if (compressed_size<=0){
        throw std::runtime_error("[LZ4] Compression Failed!");
    }

}


std::unique_ptr<uint8_t[]> LZ4compressor::get_decompressed_buffer(
    const size_t& original_size
){
    std::unique_ptr<uint8_t[]> decompressed_ptr(new uint8_t[original_size]);
    return decompressed_ptr;
}

void LZ4compressor::decompress(
    uint8_t* compressed_ptr,
    size_t& compressed_size,
    size_t& original_size,
    uint8_t* decompressed_ptr
){
    int decompressed_size = LZ4_decompress_safe_partial(
        reinterpret_cast<char*>(compressed_ptr),
        reinterpret_cast<char*>(decompressed_ptr),
        compressed_size,
        original_size,
        original_size
    );
    if (decompressed_size<=0){
        throw std::runtime_error("[LZ4] Decompression failed!");
    }
}


