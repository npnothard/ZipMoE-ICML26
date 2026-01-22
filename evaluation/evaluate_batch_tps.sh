#!/usr/bin/env bash
set -e
set -euo pipefail
#source /home/username/miniforge3/etc/profile.d/conda.sh
CONDA_BASE=$(conda info --base)
source "CONDA_BASE/etc/profile.d/conda.sh"
clear_cache() {
  echo ">> Clearing OS page cache"
  sudo sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

  echo ">> Clearing CUDA cache"
  stty sane
}

# Configure experiments
VIRTUAL_ENV=zipmoe
PYTHON=python3
SCRIPT=evaluation/evaluate.py

cd ZIPMOE-PREFIX/ZipMoE
export PYTHONPATH="$(pwd)"



OUTPUT_DIR=ZIPMOE-PREFIX/ZipMoE/evaluation/results/
SSD_TYPE=Samsung970EVO # Variable, choices=["Samsung970EVO","AigoP2000"]
CACHE_ALG=ZipMoE # Variable, choices=["ZipMoE","LRU","LFU","Marking","FIFO"]
PREFETCHER_TOPKS=0
ENGINE=ZipMoE # Variable choices=["ZipMoE+","ZipMoE","MoE-Infinity","Deepspeed","Accelerate"]


MODEL_TYPE=(switch) #(deepseek qwen switch)


DEVICE_MEMORY_FOOTPRINT=(20) #(10 15 20 25 30) # Experiment 1
BATCH_SIZE=(4 16) #(4 16 32) # Experiment 2
NUM_PROMPTS=(96)
MAX_NEW_TOKENS=(64) #(32 64 128 256 512) # Experiment 3


# activate
conda activate ${VIRTUAL_ENV}

echo ">> Please enter sudo password to enable cache clearing:"
sudo -v

# Keep sudo alive
while true; do
  sudo -n true
  sleep 60
  kill -0 "$$" || exit
done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap "kill ${SUDO_KEEPALIVE_PID} 2>/dev/null" EXIT

for MT in "${MODEL_TYPE[@]}"; do
    for DMF in "${DEVICE_MEMORY_FOOTPRINT[@]}"; do
        for BS in "${BATCH_SIZE[@]}"; do
            for MNT in "${MAX_NEW_TOKENS[@]}"; do

                EXP_NAME="mt${MT}_dmf${DMF}_bs${BS}_mnt${MNT}"

                clear_cache
                sleep 2

                echo "-----------------------------------------------------"
                echo ">> Starting Experiment: ${EXP_NAME}"
                echo "-----------------------------------------------------"

                ${PYTHON} ${SCRIPT} \
                --model_type ${MT} \
                --memory_footprint ${DMF} \
                --batch_size ${BS} \
                --prefetcher_topk ${PREFETCHER_TOPKS} \
                --max_new_tokens ${MNT} \
                --SSD_type ${SSD_TYPE} \
                --output_dir ${OUTPUT_DIR} \
                --num_test_prompts ${NUM_PROMPTS} \
                --cache_algorithm ${CACHE_ALG}

                echo "Finished ${EXP_NAME}"
                echo
                sleep 2

                

            done
        done
    done
done