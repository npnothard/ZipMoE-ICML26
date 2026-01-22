# ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling

This repository contains artifact of ZipMoE, an efficient, semantically lossless serving system for Mixture-of-Experts models on mobile and edge computing platforms. 

> While Mixture-of-Experts (MoE) architectures substantially bolster the expressive power of large-language models, their prohibitive memory footprint severely impedes the practical deployment on resource-constrained edge devices, especially when model behavior must be preserved without relying on lossy quantization. In this paper, we present ZipMoE, an efficient and semantically lossless on-device MoE serving system. ZipMoE exploits the synergy between the hardware properties of edge devices and the statistical redundancy inherent to MoE parameters via a caching-scheduling co-design with provable performance guarantee. Fundamentally, our design shifts the paradigm of on-device MoE inference from an I/O-bound bottleneck to a compute-centric workflow that enables efficient parallelization. We implement a prototype of ZipMoE and conduct extensive experiments on representative edge computing platforms using popular open-source MoE models and real-world workloads. Our evaluation reveals that ZipMoE achieves up to $72.77\%$ inference latency reduction and up to $6.76\times$ higher throughput than the state-of-the-art systems.

---

We provide instructions for building and running this demo.
Please note that this demo is currently under submission to ICML'26 and is only accessible to the reviewers.

**DO NOT DISTRIBUTE**.

For further proprietary details, please refer to the LICENSE and NOTICE documents.


# 💻 Hardware/Software Prerequisite
- Edge Computing Platform: Any UMA device compatible with CUDA and the required software environments. However, we recommend using the Jetson AGX Orin modules, as this configuration is guaranteed to work without needing to modify the CMakeLists.txt build scripts.
- Operating systems: Ubuntu 22.04 (Jetpack 6.2.1)
- Resource requirement
  - CPU: >= 6 cores
  - Memory: >= 8 GB
  - Disk: >= 64 GB
  - Network: No specific requirements
- Environment:
  - C++ 17
  - CMake: 4.0.0
  - python: 3.10.18
  - Conda environment name: zipmoe
  - torch: 2.8.0
  - Pybind11: 3.0.1
  - transformers: <4.50



# ⭐ Demo Instructions

**1. Download the GitHub repo.**

**2. Prepare models**
Download model weights from Huggingface. Currently, we support the following models:
```bash
  - Qwen1.5: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat
  - DeepSeekV2:https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
  - SwitchTransformers: https://huggingface.co/google/switch-large-128
```
Download the BF16 model in .safetensors format. All model weights should be placed under the directory /models/model_type.


3. Install the library denpendencies

Create a new conda environment named zipmoe. 

run: pip install accelerate chardet "datasets>=2.12.0" fastapi hjson ninja "numpy==1.22.4" openai "optimum>=1.17.1" "packaging>=20.0" pre-commit py-cpuinfo "pyarrow==12.0.0" "pydantic==1.10.12" scipy sentencepiece sphinx "torch>=2.1.1" "transformers>=4.37.1,<4.47" uvicorn

you can select form (or both) lz4 and zstd as your compressor backends. Make sure you have installed your selection.
  - lz4 (build from source https://github.com/lz4/lz4)
  - zstd (build from source https://github.com/facebook/zstd)

If you wish to use FlashAttention, make sure it is installed properly on Jetson. You can also run this demo without this. We still give instructions on how to build flash_atten on jetson platforms.

- flash_attn = 2.8.3 (build from source https://github.com/Dao-AILab/flash-attention.git, 
      -> add:
	    if "87" in cuda_archs():
		cc_flag.append("-gencode")
		cc_flag.append("arch=compute_87,code=sm_87") in (around) line 179 of setup.py
      -> git checkout v2.8.3
      -> export TORCH_CUDA_ARCH_LIST="8.7"
      -> export FLASH_ATTN_CUDA_ARCHS=87
      -> export MAX_JOBS=6
      -> pip install . --no-build-isolation
      )



4. Subtitute placeholder paths with your own environment

There are two placeholder paths, replace all of them.
- ZIPMOE-PREFIX: The father directory where you put the ZipMoE directory.
- CONDA-PREFIX: This can be verified using $CONDA_PREFIX. Replace it with your output of this command.


5. Build the project

enter csrc/build
run cmake ..
make -j$(n_proc)
make install


6. Evaluation

You can try stream chat mode by running 

python zipmoe_stream_chat.py

For large-scale evaluations, go to evaluation/, and you may run the following bash scripts:

cd evaluation/ 
./evaluate_mem_latency.sh for single-batch inference.
./evaluate_batch_tps.sh for batch processing.



Initialization Time: Before the first run of a new model, the system must offload parameters. This is a one-time process that may take approximately 10 minutes. Subsequent initializations of the same model will not require this step.

Warm-up: The first inference run in a session will be slower than subsequent runs, as tensors are not forced onto the GPU prior to the first prompt.

Note the before each new deployment of a model, it will first offload the parameters, which may take around 10 minutes. This is a one-shot process, further initialization of this model will not require offloading anymore.

The first run of each session is will be slower than the subsequent runs, because we do not force tensor to be filled on GPU prior to your first prompt.


---


# ⭕ Limitations

- ZipMoE is mainly tested on NVIDIA Jetson AGX Orin. The current implementation is not guaranteed to build successfully without modifying CMakeLists.txt.

- ZipMoE is mainly optimized for devices with uniform memory architectures. For discrete GPUs, its efficiency is not guaranteed.

