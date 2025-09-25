# Tiny RAG + Sentiment Classifier

This repository provides two lightweight, CPU-optimized projects designed for efficient experimentation and reproducible results:

1. **Retrieval-Augmented Generation (RAG)**: A compact question-answering system that delivers concise, factual responses based on a small document corpus.
2. **Sentiment Classification**: A lightweight sentiment classifier trained and evaluated on a compact text dataset.

Both components are optimized to run locally on CPU-only hardware, leveraging efficient models and libraries for accessibility and ease of use.

---

## Quick Start

Execute the core components from the project root using the following commands:

```bash
python src/rag_answer.py
python src/train.py
```

Output files will be generated in the `submissions/` directory.

---

## Setup Instructions

### 1. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Ensure the latest version of `pip` and install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file includes CPU-friendly libraries such as:

- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `scikit-learn`
- `numpy`
- `pandas`
- `tqdm`

Ensure compatibility by using versions optimized for CPU-only environments.

---

## Project Structure

```
data/
  corpus/
    docs.jsonl          # Document corpus for RAG (JSONL format)
    questions.json      # Questions for RAG (JSON format)
  sentiment/
    train.csv           # Sentiment training dataset: text, label
    dev.csv             # Sentiment development dataset
    test.csv            # Sentiment test dataset (labels hidden)
config.json            # Optional configuration for experiments and models
src/
  rag_answer.py        # Main script for RAG question answering
  train.py             # Main script for sentiment classifier training
  rag_experiments/
    rag_exp_1.py       # RAG experiment scripts
    rag_exp_2.py
    rag_exp_3.py
  sentiment_analysis_experiments/
    train_exp_1.py     # Sentiment analysis experiment scripts
    train_exp_2.py
    train_exp_3.py
submissions/           # Output directory for main script results
experiment_results/    # Output directory for experiment results
```

---

## Running the Projects

### Part A: Retrieval-Augmented Generation (RAG)

Generate concise answers for questions provided in `data/corpus/questions.json`:

```bash
python src/rag_answer.py
```

**Output**:

- `submissions/rag_answers.json`: A JSON file mapping question IDs (`q0`, `q1`, ...) to short, factual answers.

### Part B: Sentiment Classification

Train and evaluate a lightweight sentiment classifier, generating predictions for the hidden test set in `data/sentiment/test.csv`:

```bash
python src/train.py
```

**Output**:

- `submissions/sentiment_test_predictions.csv`: A CSV file with predictions in the format:

  ```
  text,label
  ```

  where `label` is `0` (negative) or `1` (positive).

---

## Experimentation

The repository includes additional scripts for ablation studies and experimental analysis.

### RAG Experiments

Located in `src/rag_experiments/`, these scripts explore variations in the RAG pipeline. Run an experiment with:

```bash
python src/rag_experiments/rag_exp_1.py
```

### Sentiment Analysis Experiments

Located in `src/sentiment_analysis_experiments/`, these scripts test different configurations for the sentiment classifier. Run an experiment with:

```bash
python src/sentiment_analysis_experiments/train_exp_1.py
```

**Output**: All experiment results are saved in the `experiment_results/` directory.

---

## Configuration and Defaults

- **CPU-Optimized**: All scripts are configured to run efficiently on CPU-only hardware, ensuring accessibility.
- **Model Defaults**:
  - **RAG Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - **RAG Local LLM**: `google/flan-t5-small` (or another small, instruction-tuned T5 model)
  - **Sentiment Baseline**: TF-IDF with Logistic Regression
- **Directory Creation**: The `submissions/` and `experiment_results/` directories are automatically created if they do not exist.
- **Corpus Size**: Input corpora are kept small (e.g., 10–100 documents) to enable rapid iteration and experimentation.

---

## Notes

- Ensure all dependencies are installed correctly to avoid compatibility issues.
- The small corpus size and lightweight models make this project ideal for quick prototyping and testing.
- For further details on model configurations, refer to `config.json` (if provided).