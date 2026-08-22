# Order Intelligence Support System

A multimodal e-commerce support assistant combining:

- Return-risk prediction using a scikit-learn model
- Product-image classification using ResNet-18
- Policy retrieval using Sentence Transformers and FAISS
- Deterministic LangGraph routing
- Prompt-injection guardrails

## Requirements

- macOS
- Python 3.11
- Homebrew
- Git

Python 3.8 is not supported by the current LangGraph workflow.

## Setup

From the project root:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

Activate the environment:

```bash
source .venv/bin/activate
```

The setup pins compatible versions for NumPy, scikit-learn, PyTorch, Transformers, and FAISS.

## Run the application

```bash
./run.sh
```

Or:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python agent.py
```

## Run tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Current result:

```text
7 passed
```

## Project structure

```text
.
├── agent.py
├── tools.py
├── rag.py
├── return_risk.py
├── predict_order.py
├── image_classify.py
├── policy/
│   └── documents.json
├── models/
│   ├── return_risk_model.pkl
│   └── product_classifier.pt
├── data/
│   └── sample_images/
├── tests/
│   └── test_agent.py
├── setup.sh
└── run.sh
```

## System workflow

```text
User query
   │
   ▼
LangGraph router
   ├── Policy question → Sentence Transformer + FAISS retrieval
   ├── Risk question   → Return-risk model
   ├── Image question  → ResNet-18 classifier
   ├── General chat    → Help response
   └── Injection       → Guardrail response
```

## Policy retrieval

Policy content is stored in `policy/documents.json`.

`rag.py`:

1. Loads the policy documents.
2. Splits each document into sentence-level chunks.
3. Generates embeddings with `all-MiniLM-L6-v2`.
4. Indexes the embeddings with FAISS.
5. Returns the most relevant policy sentences.

Run a retrieval check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python rag.py
```

## Model-backed tools

### Return-risk prediction

`tools.py` loads:

```text
models/return_risk_model.pkl
```

The tool returns:

- Return probability
- Risk bucket
- Decision threshold

### Product-image classification

`tools.py` loads:

```text
models/product_classifier.pt
```

The classifier predicts one of the Fashion-MNIST categories, including Trouser, Coat, Shirt, Sneaker, Bag, and Ankle boot.

## Model results

The image classifier achieved:

- Validation accuracy: **88.68%**
- Test accuracy: **87.20%**

A sample image was classified as:

```text
Category: Trouser
Confidence: 99.97%
```

## Development notes

Use the virtual environment for all commands:

```bash
source .venv/bin/activate
```

The `.venv/` directory, Python caches, and generated feature caches are excluded from Git.

Warnings from Hugging Face about deprecated `resume_download` behavior do not affect execution.