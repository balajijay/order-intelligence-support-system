#!/bin/bash
set -e

source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python agent.py