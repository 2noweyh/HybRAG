# Hybrid Retrieval-Augmented Generation: Semantic and Structural Integration for Large Language Model Reasoning

HybRAG is a **hybrid retrieval-augmented generation (RAG)** framework that integrates **semantic node-level** and **structural path-level** retrievers for knowledge-grounded reasoning and question answering (QA).  
This repository includes all components required for preprocessing, training, inference, and evaluation of the HybRAG pipeline.

---

## 📂 Project Structure

```

HybRAG/
├── data/                   # Raw / processed datasets (ignored in repo)
├── data_preprocess/        # Scripts for dataset parsing and graph extraction
├── models/                 # GNN, retriever, and LLM model definitions
├── node_retriever/         # Semantic (dense) retriever modules
├── path-retriever/         # Structural (path-based) retriever modules
├── raft_training/          # RAFT-based fine-tuning (LoRA / DeepSpeed)
├── raft_inference/         # Inference scripts for generation & reasoning
├── raft_evaluation/        # Evaluation metrics and scoring modules
├── templates/              # Prompt and model configuration templates
└── outputs/                # Generated results and checkpoints (ignored)

````

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/2noweyh/HybRAG.git
cd HybRAG
conda create -n hybrag python=3.11
conda activate hybrag
pip install -r requirements.txt
````

---

### 2. Data Preparation

Place your dataset files under:

```
data/webqsp/
data/cwq/
```

*(Large data folders are excluded from version control via `.gitignore`.)*

---

### 3. Training

Run RAFT fine-tuning:

```bash
bash raft_training/train.sh
```

---

### 4. Inference

```bash
bash raft_inference/inference.sh
```

---

### 5. Evaluation

```bash
bash raft_evaluation/evaluation.sh
```

---

## 🧠 Core Idea

HybRAG bridges **semantic** and **structural** reasoning by combining:

* **Node Retriever:** Retrieves relevant entities and documents using dense embeddings.
* **Path Retriever:** Explores relational paths and subgraphs in the knowledge graph.
* **Hybrid Prompt Generator:** Integrates both retrieval types for grounded LLM reasoning.
* **RAFT Training:** Fine-tunes LLMs with graph-augmented instructions.

This hybrid architecture enhances **factual consistency**, **reasoning depth**, and **interpretability** across QA and scientific knowledge-grounded tasks.
