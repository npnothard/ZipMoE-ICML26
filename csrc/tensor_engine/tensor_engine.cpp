// Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "tensor_engine.hpp"
#include <nvtx3/nvToolsExt.h>


std::unique_ptr<TensorEngine> kZipMoETensorEngine = nullptr;


cudaStream_t TensorEngine::initialize_stream(){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}


ZipMoECompressor TensorEngine::initialize_compressor(
    const std::string& CODE_TYPE
){
    if ( CODE_TYPE == "LZ4" ){
        return std::make_unique<LZ4compressor>(config_ptr->LZ4_accelerationLevel);
    }
    else if ( CODE_TYPE == "LZ4HC" ){
        return std::make_unique<LZ4HCcompressor>(config_ptr->LZ4HC_compressionLevel);
    }
    else if ( CODE_TYPE == "ZSTD" ){
        return std::make_unique<ZSTDcompressor>(
            config_ptr->num_compute_workers,
            config_ptr->ZSTD_compressionLevel,
            compute_pool
        );
    }
    else {
        DLOG_FATAL("[ZipMoE] Unknown Compressor Type !");
        return std::make_unique<LZ4HCcompressor>(config_ptr->LZ4HC_compressionLevel);
    }
}

TensorEngine::TensorEngine(
    EngineConfigPtr config_ptr
):  config_ptr(config_ptr),
    tensor_stream( initialize_stream() ),
    compute_pool( config_ptr->num_compute_workers, 0, config_ptr->bind_core ),
    IO_thread( 1 , config_ptr->num_compute_workers, config_ptr->bind_core ),
    compressor( initialize_compressor(config_ptr->CODE_TYPE) ),
    scheduler_core(
        config_ptr->num_compute_workers,
        config_ptr->num_file_chunks,
        config_ptr->decompression_delay,
        config_ptr->sm_io_delay
    )
{
    if (config_ptr->bind_core){
        atexit(cleanup_cpuset_wrapper);
        exclusive_cpuset_exception_handler(2+config_ptr->num_compute_workers);
    }
    compressed_buffer_base_ = (uint8_t*)malloc(config_ptr->num_compute_workers*config_ptr->shared_mem_size);
    compressed_buffer.resize(config_ptr->num_compute_workers);
    for (int i = 0; i<config_ptr->num_compute_workers; i++){
        compressed_buffer[i] = compressed_buffer_base_ + i*config_ptr->shared_mem_size;
    }
}


TensorEngine::~TensorEngine(){
    cudaStreamSynchronize(cudaStreamDefault);
    cudaStreamSynchronize(tensor_stream);
    free(compressed_buffer_base_);
    cudaStreamDestroy(tensor_stream);
}


std::unique_ptr<uint8_t[]> TensorEngine::compress_chunkfile_core(
    uint8_t* uncompressed_ptr,
    size_t input_size,
    size_t& compressed_size
){
    size_t compress_bound;
    return std::visit(
        [&](auto&& compressor_ptr){
            std::unique_ptr<uint8_t[]> compressed_ptr = compressor_ptr->get_compressed_buffer(
                input_size,
                compress_bound
            );
            compressor_ptr->compress(
                uncompressed_ptr,
                input_size,
                compress_bound,
                compressed_ptr.get(),
                compressed_size
            );
            return compressed_ptr;
        },
        compressor
    );
}


std::future<uint8_t*> TensorEngine::compress_submit(
    uint32_t tensor_id,
    int chunk_idx,
    uint8_t* uncompressed_ptr,
    size_t file_size
){
    auto [compress_task, compress_future] = create_task<uint8_t*>(
        tensor_id,
        std::to_string(tensor_id),
        [ this , tensor_id, chunk_idx, uncompressed_ptr, file_size ]
        ( std::shared_ptr<PreemptToken> token, int thread_idx){
            if (!uncompressed_ptr){
                std::cerr << "[Error] uncompressed_ptr is null, CANNOT compress chunk: "<< chunk_idx << std::endl;
                uint8_t* error_out;
                return error_out;
            }
            size_t compressed_size;
            std::unique_ptr<uint8_t[]> compressed_ptr = compress_chunkfile_core(
                uncompressed_ptr,
                file_size,
                compressed_size
            );
            std::lock_guard<std::mutex> lock(kTensorIndex->index_mutex);
            auto& meta = kTensorIndex->at(tensor_id);
            meta.compressed_sizes[chunk_idx] = compressed_size;

            return compressed_ptr.release();
        }
    );
    compute_pool.add_task(compress_task);
    return std::move(compress_future);
}


size_t TensorEngine::write_compressed_append(
    void* data_ptr,
    size_t data_size
){
    std::ofstream file(
        config_ptr->offload_file,
        std::ios::binary|std::ios::app
    );
    if(!file.is_open()){
        DLOG_FATAL("[ZipMoE] Could not open file for appending: ", config_ptr->offload_file);
    }
    auto start_pos = file.tellp();
    file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    file.flush();
    auto current_pos = file.tellp();
    size_t padding_size = 0;
    if ( current_pos % OS_PAGE_SIZE != 0 ){
        padding_size = OS_PAGE_SIZE - ( current_pos % OS_PAGE_SIZE );
    }
    if (padding_size > 0){
        std::vector<char> padding_buffer(padding_size, 0);
        file.write(padding_buffer.data(), padding_size);
    }
    file.flush();
    file.close();
    return start_pos;
}


void TensorEngine::PipelineCompress(
    const std::vector<py::array_t<uint8_t>>& exponents_chunks,
    const py::array_t<uint8_t>& sign_mantissa,
    uint32_t tensor_id
){
    std::vector<PythonNdArrayInfo> exp_chunks_meta;
    exp_chunks_meta.reserve(exponents_chunks.size());
    for (const auto& chunk : exponents_chunks ){
        py::buffer_info exp_buf = chunk.request();
        exp_chunks_meta.push_back({
            static_cast<uint8_t*>(exp_buf.ptr),
            static_cast<size_t>(exp_buf.size)
        });
    }
    py::buffer_info sm_buf = sign_mantissa.request();
    uint8_t* sm_ptr = static_cast<uint8_t*>(sm_buf.ptr);
    size_t sm_size = static_cast<size_t>(sm_buf.size);
    {
        py::gil_scoped_release release;
        std::vector<std::future<uint8_t*>> compress_futures;
        auto& meta = kTensorIndex->at(tensor_id);
        meta.sm_size = sm_size;
        assert(config_ptr->num_file_chunks==exp_chunks_meta.size());
        for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++){
            std::future<uint8_t*> compress_future = compress_submit(
                tensor_id,
                chunk_idx,
                exp_chunks_meta[chunk_idx].ptr,
                exp_chunks_meta[chunk_idx].size
            );
            compress_futures.push_back(std::move(compress_future));
            meta.original_sizes[chunk_idx] = exp_chunks_meta[chunk_idx].size;
        }
        for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++){
            uint8_t* compressed_ptr = compress_futures[chunk_idx].get();
            if (!compressed_ptr){
                DLOG_FATAL("[Error] Compression failed for chunk ", chunk_idx);
            }
            size_t exp_offset = write_compressed_append(
                compressed_ptr,
                meta.compressed_sizes[chunk_idx]
            );
            meta.exp_file_offsets[chunk_idx] = exp_offset;
            delete[] compressed_ptr;
        }
        size_t sm_offset = write_compressed_append(
            sm_ptr,
            sm_size
        );
        meta.sm_file_offset = sm_offset;
        DLOG_TRACE("[ZipMoE] Compression completes for tensor: ", tensor_id);
    }
}


void TensorEngine::BatchPipelineCompress(
    const std::vector<uint32_t>& tensor_ids,
    const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
    const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
){
    size_t batch_size = tensor_ids.size();
    if (batch_size == 0) return;
    struct TensorRawData {
        uint32_t id;
        std::vector<PythonNdArrayInfo> exp_chunks;
        PythonNdArrayInfo sm;
    };
    std::vector<TensorRawData> batch_raw_data(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        batch_raw_data[i].id = tensor_ids[i];
        const auto& chunks = batch_exponents_chunks[i];
        batch_raw_data[i].exp_chunks.reserve(chunks.size());
        for (const auto& chunk : chunks) {
            py::buffer_info buf = chunk.request();
            batch_raw_data[i].exp_chunks.push_back({
                static_cast<uint8_t*>(buf.ptr),
                static_cast<size_t>(buf.size)
            });
        }
        py::buffer_info sm_buf = batch_sign_mantissa[i].request();
        batch_raw_data[i].sm.ptr = static_cast<uint8_t*>(sm_buf.ptr);
        batch_raw_data[i].sm.size = static_cast<size_t>(sm_buf.size);
    }

    {
        py::gil_scoped_release release;
        std::vector<std::vector<std::future<uint8_t*>>> batch_futures(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            uint32_t tid = batch_raw_data[i].id;
            {
                std::lock_guard<std::mutex> lock(kTensorIndex->index_mutex);
                auto& meta = kTensorIndex->at(tid);
                meta.sm_size = batch_raw_data[i].sm.size;
                for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++) {
                    size_t original_size = batch_raw_data[i].exp_chunks[chunk_idx].size;
                    meta.original_sizes[chunk_idx] = original_size;
                }
            }
            for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++) {
                batch_futures[i].push_back(
                    compress_submit(
                        tid,
                        chunk_idx,
                        batch_raw_data[i].exp_chunks[chunk_idx].ptr,
                        batch_raw_data[i].exp_chunks[chunk_idx].size
                    )
                );
            }
        }
        for (size_t i = 0; i < batch_size; ++i) {
            uint32_t tid = batch_raw_data[i].id;
            auto& meta = kTensorIndex->at(tid);
            for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++) {
                uint8_t* compressed_ptr = batch_futures[i][chunk_idx].get();
                if (!compressed_ptr) {
                    DLOG_FATAL("[Error] Batch Compression failed for tensor ", tid, " chunk ", chunk_idx);
                }
                size_t exp_offset = write_compressed_append(
                    compressed_ptr,
                    meta.compressed_sizes[chunk_idx]
                );
                meta.exp_file_offsets[chunk_idx] = exp_offset;

                delete[] compressed_ptr; 
            }
            size_t sm_offset = write_compressed_append(
                batch_raw_data[i].sm.ptr,
                batch_raw_data[i].sm.size
            );
            meta.sm_file_offset = sm_offset;
            DLOG_TRACE("[ZipMoE] Batch Offload: Finished tensor ", tid);
        }
    }
}

void TensorEngine::OffloadDense(
    torch::Tensor& tensor,
    uint32_t tensor_id
){
    auto& meta = kTensorIndex->at(tensor_id);
    void* data_ptr = tensor.data_ptr();
    size_t nbytes = meta.size;
    size_t offset = write_compressed_append(
        data_ptr,
        nbytes
    );
    meta.sm_file_offset = offset;
    DLOG_TRACE("[ZipMoE] Dense offload completes for tensor: ", tensor_id);
}


void TensorEngine::ReadDense(
    uint8_t* target_gpu_ptr,
    size_t file_offset,
    size_t nbytes
){
    uint8_t* cpu_buffer = (uint8_t*)malloc(nbytes);
    if (!cpu_buffer){
        throw std::runtime_error("Read Dense malloc failed!");
    } 
    ssize_t n = pread_exact(
        this->fd_offload_file,
        cpu_buffer,
        nbytes,
        file_offset
    );
    cudaMemcpy(
        target_gpu_ptr,
        cpu_buffer,
        nbytes,
        cudaMemcpyHostToDevice
    );
    free(cpu_buffer);
}

void TensorEngine::OpenOffloadFile(){
    std::string offload_file_path = config_ptr->offload_file;
    fd_offload_file = open(offload_file_path.c_str(), O_RDONLY);

    if (fd_offload_file == -1){
        DLOG_FATAL("[ZipMoE] Offload File Does NOT Exist!");
    }

    DLOG_INFO("[ZipMoE] Opened offloaded file: ", offload_file_path);
}


std::unique_ptr<uint8_t[]> TensorEngine::decompress_chunkfile_core(
    uint8_t* compressed_ptr,
    size_t compressed_size,
    size_t original_size
){
    return std::visit(
        [&](auto&& compressor_ptr){
            nvtxRangePushA("get_buffer");
            std::unique_ptr<uint8_t[]> decompressed_ptr = compressor_ptr->get_decompressed_buffer(original_size);
            nvtxRangePop();
            nvtxRangePushA("decompress");
            compressor_ptr->decompress(
                compressed_ptr,
                compressed_size,
                original_size,
                decompressed_ptr.get()
            );
            nvtxRangePop();
            return decompressed_ptr;
        },
        compressor
    );
}


std::future<void> TensorEngine::io_sign_mantissa_submit(
    ZipMoETaskPtr task,
    size_t sm_file_offset,
    uint8_t* host_sm_pinned_ptr
){
    auto [operation, future] = create_task<void>(
        task->priority,
        "",
        [this, task, sm_file_offset, host_sm_pinned_ptr]
        (std::shared_ptr<PreemptToken> token, int thread_idx){

            try{
                nvtxRangePushA("pread: sm I/O");
                ssize_t n = pread_exact(
                    this->fd_offload_file,
                    host_sm_pinned_ptr,
                    this->config_ptr->shared_mem_size,
                    sm_file_offset
                );
                if (n == -1){
                    DLOG_INFO("[ZipMoE Error] pread SM failed! ");
                    std::cerr << "[ZipMoE Error] pread SM failed! Errno: " << errno << std::endl;
                }
                nvtxRangePop();

            } catch(...) {
                std::cerr << "[ZipMoE Critical] Exception inside SM IO thread" << std::endl;
            }
            if (task->on_operation_done()){
                task_ready_notify(task);
            };

        }
    );
    operation->preempt_token = task->preempt_token;
    IO_thread.add_task( operation );
    return std::move( future );
}



std::future<void> TensorEngine::io_decompress_mov_submit(
    ZipMoETaskPtr task,
    size_t exp_file_offset,
    size_t compressed_size,
    size_t original_size,
    uint8_t* host_exp_pinned_ptr_with_offset
){
    auto [operation, future] = create_task<void>(
        task->priority,
        "",
        [this, task, exp_file_offset, compressed_size, original_size, host_exp_pinned_ptr_with_offset]
        (std::shared_ptr<PreemptToken> token, int thread_idx){

            try {
                nvtxRangePushA("Access thread-private memory");
                uint8_t* compressed_ptr = this->compressed_buffer[thread_idx];
                nvtxRangePop();
                nvtxRangePushA("pread: exp I/O");
                ssize_t n = pread_exact(
                    this->fd_offload_file,
                    compressed_ptr,
                    compressed_size,
                    exp_file_offset
                );
                if (n == -1){
                    DLOG_INFO("[ZipMoE Error] pread failed in worker thread! ");
                    std::cerr << "[ZipMoE Error] pread failed in worker thread! Errno: " << errno << std::endl;
                    throw std::runtime_error("[ZipMoE] failed to pread chunk !");
                }
                nvtxRangePop();
                std::unique_ptr<uint8_t[]> decompressed_ptr = decompress_chunkfile_core(
                    compressed_ptr,
                    compressed_size,
                    original_size
                );
                nvtxRangePushA("memcpy");
                memcpy(
                    host_exp_pinned_ptr_with_offset,
                    decompressed_ptr.get(),
                    original_size
                );
                nvtxRangePop();
            } catch (...) {
                std::cerr << "[ZipMoE Critical] Exception inside worker thread for Task " << task->layer_idx << std::endl;
            }
            if (task->on_operation_done()){
                task_ready_notify(task);
            };
        }
    );
    operation->preempt_token = task->preempt_token;
    compute_pool.add_task( operation );
    return std::move( future );
}


void TensorEngine::task_ready_notify(const ZipMoETaskPtr& task){
    
    {
        std::lock_guard<std::mutex> lock_schedule(scheduled_queue_mutex);
        std::lock_guard<std::mutex> lock_ready(ready_queue_mutex);
        auto it = std::find(
            scheduled_task_queue.begin(),
            scheduled_task_queue.end(),
            task
        );
        if (it!=scheduled_task_queue.end()){
            ready_queue.emplace_back(std::move(*it));
            scheduled_task_queue.erase(it);
        }
    }
    ready_queue_cv.notify_all();
}


void TensorEngine::unregister_memory(ZipMoETaskPtr& task){
    kMemoryDock->UnregisterDockForNode(task->node);
    switch (task->state){
        case TaskState::EVICT:
            break;
        case TaskState::SMCACHE:
            kZipMoECacheHandle->UnregisterSMCacheForNode(task->node);
            break;
        case TaskState::NORMAL:
        case TaskState::PREFETCH:
            kZipMoECacheHandle->UnregisterCacheSlotForNode(task->node);
            break;
        default:
            assert(false);
    }
    task->node->mutex.unlock();
}


void TensorEngine::ToWorkers(ZipMoETaskPtr& task){
    {
        std::lock_guard<std::mutex> lock(scheduled_queue_mutex);
        scheduled_task_queue.push_back(task);
    }
    bool needs_sm_io = (task->node->device_sm_ptr == nullptr);
    kMemoryDock->RegisterDockForNode(task->node);
    auto dock = kMemoryDock->GetDock(task->node);
    uint8_t* host_sm_pinned_ptr;
    uint8_t* device_sm_pinned_ptr;
    switch (task->state){
        case TaskState::EVICT:{

            if (needs_sm_io){
                host_sm_pinned_ptr = dock->get_sm_pinned(0).first;
                device_sm_pinned_ptr = dock->get_sm_pinned(0).second;
            } else {
                host_sm_pinned_ptr = dock->get_sm_pinned(0).first;
                device_sm_pinned_ptr = task->node->device_sm_ptr;
            }
            task->receive_ptrs(
                dock->get_tensor(),
                dock->get_exp_pinned(0).second,
                device_sm_pinned_ptr
            );
            break;
        }
        case TaskState::SMCACHE:{
            kZipMoECacheHandle->RegisterSMCacheForNode(task->node);
            auto sm_cache = kZipMoECacheHandle->GetSMCacheSlot(task->node);
            task->receive_ptrs(
                dock->get_tensor(),
                dock->get_exp_pinned(0).second,
                sm_cache->ptr.second
            );
            host_sm_pinned_ptr = sm_cache->ptr.first;
            break;
        }
        case TaskState::NORMAL:
        case TaskState::PREFETCH:{

            if (needs_sm_io){
                host_sm_pinned_ptr = dock->get_sm_pinned(0).first;
                device_sm_pinned_ptr = dock->get_sm_pinned(0).second;
            } else {
                host_sm_pinned_ptr = nullptr;
                device_sm_pinned_ptr = task->node->device_sm_ptr;
            }
            kZipMoECacheHandle->RegisterCacheSlotForNode(task->node);
            auto gpu_cache = kZipMoECacheHandle->GetCacheSlotSlot(task->node);
            task->receive_ptrs(
                gpu_cache->ptr,
                dock->get_exp_pinned(0).second,
                device_sm_pinned_ptr
            );
            break;
        }
        default:{
            assert(false);
            break;
        }
    }
    for (int tensor_idx = 0; tensor_idx < task->node->tensor_ids.size(); tensor_idx++){
        auto& meta = kTensorIndex->at(task->node->tensor_ids[tensor_idx]);
        uint8_t* host_exp_pinned_ptr = dock->get_exp_pinned(tensor_idx).first;
        if ( needs_sm_io ){
            std::future<void> io_fut = io_sign_mantissa_submit(
                task,
                meta.sm_file_offset,
                host_sm_pinned_ptr
            );
        }
        size_t exp_offset = 0;
        for (int chunk_idx = 0; chunk_idx < config_ptr->num_file_chunks; chunk_idx++){
            size_t exp_file_offset = meta.exp_file_offsets[chunk_idx];
            size_t compressed_size = meta.compressed_sizes[chunk_idx];
            size_t original_size = meta.original_sizes[chunk_idx];
            uint8_t* host_exp_pinned_ptr_with_offset = host_exp_pinned_ptr + exp_offset;
            std::future<void> decomp_fut = io_decompress_mov_submit(
                task,
                exp_file_offset,
                compressed_size,
                original_size,
                host_exp_pinned_ptr_with_offset
            );
            exp_offset += meta.original_sizes[chunk_idx];
        }
        host_sm_pinned_ptr += config_ptr->shared_mem_size;
    }

}


void TensorEngine::ZipMoESchedulerFunc(){
    {
        std::lock_guard<std::mutex> lock(scheduled_queue_mutex);
        for (auto it = scheduled_task_queue.begin(); it != scheduled_task_queue.end(); ){
            auto& existing_task = *it;
            if (!existing_task->prediction_hit){
                existing_task->preempt_token->set_stop();
                unregister_memory(existing_task);
                it = scheduled_task_queue.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::priority_queue<
        ZipMoETaskPtr, std::vector<ZipMoETaskPtr>, ZipMoETaskPtrCompare
    > TypeI_pq;
    std::priority_queue<
        ZipMoETaskPtr, std::vector<ZipMoETaskPtr>, ZipMoETaskPtrCompare
    > TypeII_pq;

    {
        std::lock_guard<std::mutex> lock(new_queue_mutex);
        for (auto& task : new_task_queue){
            if (
                task->node->device_sm_ptr != nullptr
            ){
                TypeII_pq.push(task);
            } else {
                TypeI_pq.push(task);
            }
        }
        new_task_queue.clear();
    }
    scheduler_core.reset();
    double priority = 1;

    while (!(TypeI_pq.empty()&&TypeII_pq.empty())){
        bool end_block = false;
        if (!TypeI_pq.empty()){
            auto J1 = TypeI_pq.top();
            J1->priority = priority;
            ToWorkers(J1);
            priority += 1;
            TypeI_pq.pop();
            end_block = scheduler_core.add_and_update(true);
        }

        if (!end_block){
            while ( (!TypeII_pq.empty())&&(!end_block) ){
                auto J2 = TypeII_pq.top();
                J2->priority = priority;
                ToWorkers(J2);
                priority += 1;
                TypeII_pq.pop();
                end_block = scheduler_core.add_and_update(false);
            }
        }
    }

}





