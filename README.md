# ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling  

This repository contains artifact of ZipMoE, an efficient, semantically lossless serving system for Mixture-of-Experts models on mobile and edge computing platforms. 

> While Mixture-of-Experts (MoE) architectures substantially bolster the expressive power of large-language models, their prohibitive memory footprint severely impedes the practical deployment on resource-constrained edge devices, especially when model behavior must be preserved without relying on lossy quantization. In this paper, we present ZipMoE, an efficient and semantically lossless on-device MoE serving system. ZipMoE exploits the synergy between the hardware properties of edge devices and the statistical redundancy inherent to MoE parameters via a caching-scheduling co-design with provable performance guarantee. Fundamentally, our design shifts the paradigm of on-device MoE inference from an I/O-bound bottleneck to a compute-centric workflow that enables efficient parallelization. We implement a prototype of ZipMoE and conduct extensive experiments on representative edge computing platforms using popular open-source MoE models and real-world workloads. Our evaluation reveals that ZipMoE achieves up to 72.77% inference latency reduction and up to $6.76\times$ higher throughput than the state-of-the-art systems.

---

###



<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.21198">Paper</a>
  &nbsp; | &nbsp;
  🚀 <a href="#-quick-start">Quick Start</a>
  &nbsp; | &nbsp;
  📚 <a href="#-citation">Citation</a>
</p>


## 🔥 News
- [2026/05/27] The code of our paper has been released.
- [2026/05/01] Our ZipMoE paper has been accepted to ICML 2026


*For further proprietary details, please refer to the LICENSE and NOTICE documents.*


## 💻 Hardware/Software Prerequisites
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
  - conda environment name: zipmoe
  - torch: 2.8.0
  - Pybind11: 3.0.1
  - transformers: <4.50
  - huggingface_hub: <1.0



## 🚀 Quick Start

### 1. Clone the GitHub Repository.
**Rename the directory of the code asset as: ZipMoE.**

### 2. Prepare Models.


Download model weights from Huggingface. Currently, we support the following models:
```bash
  - Qwen1.5: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat
  - DeepSeekV2:https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
  - SwitchTransformers: https://huggingface.co/google/switch-large-128
```
Download the BF16 model in .safetensors format. All model weights should be placed under the directory /models/model_type.


### 3. Install Library Dependencies.

Create a new conda environment with name zipmoe:
```bash
conda create -n zipmoe python=3.10
conda activate zipmoe
```

Then, install the required Python libraries by running:
```bash
pip install accelerate chardet "datasets>=2.12.0" fastapi hjson ninja "numpy==1.22.4" "huggingface_hub<1.0" openai "optimum>=1.17.1" "packaging>=20.0" pre-commit py-cpuinfo "pyarrow==12.0.0" "pydantic==1.10.12" scipy sentencepiece sphinx "torch>=2.1.1" "transformers>=4.37.1,<4.47" uvicorn nvtx
```

You can choose either lz4 or zstd as your compression backend. Make sure to install your selected option:
  - lz4 (build from source https://github.com/lz4/lz4)
  - zstd (build from source https://github.com/facebook/zstd)

If you plan to use FlashAttention, ensure it’s installed correctly on Jetson. Instructions for building it on Jetson are provided below. 
You can still run the demo without FlashAttention if you prefer.

To install FlashAttention:

3.1. Clone the repository:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3
```

3.2. Modify setup.py (around line 179):
```bash
# Note that 87 is for Jetson AGX Orin. The number is specific to your hardware.
if "87" in cuda_archs():
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_87,code=sm_87")
```

3.3. Set environment variables:
```bash
export TORCH_CUDA_ARCH_LIST="8.7"
export FLASH_ATTN_CUDA_ARCHS=87
export MAX_JOBS=6
```

3.4. Install FlashAttention:
```bash
# This may take hours.
pip install . --no-build-isolation
```



### 4. Subtitute placeholder paths with your own environment



There are two placeholder paths that need to be replaced as well:

- ZIPMOE-PREFIX: The parent directory where the ZipMoE project is located.

- CONDA-PREFIX: The path of your conda environment, which can be obtained by running the command: echo $CONDA_PREFIX.




### 5. Build the Project

Navigate to the csrc/build directory and run the following commands:
```bash
cd csrc/build
cmake ..
make -j$(nproc)
make install
```


### 6. Evaluation

Prior to evaluation, make sure your jetson is running on mode with maximum performance:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

You can set correct configurations specific to your hardware by profiling a inference with nsight systems.
The default configuration entrypoint is utils/config.py


You can try the stream chat mode by running:
```bash
python zipmoe_stream_chat.py
```
For large-scale evaluations, navigate to the evaluation/ directory and run the appropriate bash scripts:
```bash
cd evaluation/

# Single-batch inference:
./evaluate_mem_latency.sh

# Batch processing:
./evaluate_batch_tps.sh
```

**Note:** Before deploying a model for the first time, it will offload its parameters, which may take around 10 minutes. This is a one-time process. 
Subsequent initializations of the same model will not require offloading.
Also, the first run of each session may be slower than subsequent runs because the model tensors are not preloaded onto the GPU prior to the first prompt.



---


## 🚩 Limitations
- ZipMoE has been primarily tested on the NVIDIA Jetson AGX Orin. The current implementation is not guaranteed to work without modifying the CMakeLists.txt file for other platforms.
- ZipMoE is optimized for edge and mobile devices with uniform memory architectures. Efficiency on discrete GPUs is not guaranteed.


---

## 📚 Citation


If you find ZipMoE useful in your research, please consider citing:

```bibtex
@inproceedings{zipmoe2026,
  title = {ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling},
  author = {Yang, Yuchen and Zhao, Yaru and Yang, Pu and Wang, Shaowei and Zhou, Zhi-Hua},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026}
}
```
