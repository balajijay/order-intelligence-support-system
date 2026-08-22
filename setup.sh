#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "Creating Python 3.11 environment..."
python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  "numpy==1.26.4" \
  "pandas" \
  "scikit-learn==1.3.2" \
  "joblib" \
  "pillow" \
  "torch==2.2.2" \
  "torchvision==0.17.2" \
  "sentence-transformers==2.7.0" \
  "transformers==4.41.2" \
  "tokenizers==0.19.1" \
  "faiss-cpu==1.8.0" \
  "packaging" \
  "langgraph" \
  "pytest"

echo "Dependencies installed successfully."
echo "Activate the environment with: source .venv/bin/activate"
echo "Run the application with: ./run.sh"