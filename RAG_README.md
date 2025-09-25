# 🧪 Retrieval-Augmented Generation (RAG) Experiments Report (Exp-1 to Exp-5)

## ⚖️ Assumptions
The document corpus consists of small-sized entries. Examples include LyraVision, PoseidonFS, and VulcanGraph documents, each containing concise factual text.

**Key assumption:** The documents are very small, so further chunking is unnecessary and could degrade retrieval quality by reducing contextual coherence.



## 🎯 Objective
This report evaluates lightweight **Retrieval-Augmented Generation (RAG)** pipelines optimized for CPU-only environments, focusing on small language models (LLMs). The experiments assess:






- **Retrieval Quality**: Comparing keyword-based (TF-IDF) and embedding-based (MiniLM, MPNet) retrievers.
- **Instruction Compliance**: Evaluating LLMs' adherence to strict and liberal prompts.
- **Answer Precision**: Measuring accuracy in reproducing exact phrases versus generating hallucinations.
- **Prompt Design Impact**: Analyzing the effect of strict versus flexible prompting strategies.

**Dataset**:
- **Corpus**: 15 documents (`data/corpus/docs.jsonl`)
- **Queries**: 15 questions (`data/corpus/questions.json`)

---

## 📊 Experiment Summaries

### Experiment 1: TF-IDF + TinyLlama
- **Retriever**: TF-IDF (top-k=5)
- **LLM**: TinyLlama-1.1B
- **Prompt**: Strict ("copy exact phrases or return 'I don’t know'")
- **Settings**: `max_new_tokens=20`

**Observations**:
- **Retrieval**: Ineffective; missed semantic matches due to reliance on keyword overlap.
- **LLM Performance**: Poor instruction-following; frequent truncation and rule echoing.
- **Challenges**: Struggled with factual accuracy, especially for numeric or configuration details.

**Takeaway**: TF-IDF is inadequate for semantic retrieval, and TinyLlama-1.1B is underpowered for strict factual question answering.

---

### Experiment 2: FAISS + MiniLM + TinyLlama
- **Retriever**: FAISS with `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: TinyLlama-1.1B
- **Prompt**: Strict ("copy exact phrases or return 'I don’t know'")

**Observations**:
- **Retrieval**: Improved; captured semantically relevant documents.
- **LLM Performance**: Reduced rule echoing, but truncation and hallucinations persisted.
- **Challenges**: The system often misses aliases and has limited understanding of complex semantic relationships, **as it missed examples like `port 7787` for VulcanGraph** 

**Takeaway**: FAISS with MiniLM provides better retrieval than TF-IDF, though the improvement is moderate, and TinyLlama remains a limiting factor.

---

### Experiment 3: FAISS + MPNet + TinyLlama
- **Retriever**: FAISS with `sentence-transformers/all-mpnet-base-v2`
- **LLM**: TinyLlama-1.1B
- **Prompt**: Strict ("copy exact phrases or return 'I don’t know'")

**Observations**:
- **Retrieval**: Superior semantic capture (e.g., correctly identified `port 7787` for VulcanGraph).
- **LLM Performance**: Reduced redundancy, but paraphrasing errors and truncations persisted.
- **Challenges**: The system’s LLM shows limited performance, struggling to reason over facts even when retrieval provides relevant context, highlighting that model capability—not retrieval—is the main bottleneck.

**Takeaway**: MPNet enhances retrieval quality over MiniLM, but TinyLlama’s limitations hinder overall performance.

---

### Experiment 4: FAISS + MPNet + FLAN-T5-small
- **Retriever**: FAISS with `sentence-transformers/all-mpnet-base-v2`
- **LLM**: FLAN-T5-small (77M, instruction-tuned) **Smaller than TinyLlama(1.1B)**
- **Prompt**: Strict ("copy exact phrases or return 'I don’t know'")

**Observations**:
- **Retrieval**: Consistent with Experiment 3; high-quality semantic retrieval.
- **LLM Performance**: Excellent instruction-following; minimal hallucinations.
- **Challenges**: Overly strict prompt led to frequent "I don’t know" responses, reducing coverage.
- **Limitation**:  Flan model strictly follows the prompt rather than generalizing beyond it.

**Takeaway**: FLAN-T5-small outperforms TinyLlama for factual QA, but strict prompting limits answer coverage.

---

### Experiment 5: FAISS + MPNet + FLAN-T5-small (Liberal Prompt)
- **Retriever**: FAISS with `sentence-transformers/all-mpnet-base-v2`
- **LLM**: FLAN-T5-small
- **Prompt**: Liberal ("exact phrases for features, slight flexibility for numeric values")

**Observations**:
- **Retrieval**: High-quality, consistent with Experiments 3 and 4.
- **LLM Performance**: Balanced strictness and flexibility; reduced unnecessary "I don’t know" responses.
- **Strengths**: Precise answers for both qualitative (e.g., "property graph database with Gremlin-like traversal") and numeric facts (e.g., `port 7787`).


**Takeaway**: The liberal prompt optimizes FLAN-T5-small’s performance, achieving the best balance of accuracy and coverage.

---

## 📈 Comparative Analysis

| Aspect                 | Exp-1                 | Exp-2                 | Exp-3                 | Exp-4                 | Exp-5                 |
|------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| **Retriever**          | TF-IDF                | FAISS + MiniLM        | FAISS + MPNet         | FAISS + MPNet         | FAISS + MPNet         |
| **LLM**                | TinyLlama-1.1B        | TinyLlama-1.1B        | TinyLlama-1.1B        | FLAN-T5-small         | FLAN-T5-small         |
| **Prompt**             | Strict                | Strict                | Strict                | Strict                | Liberal               |
| **Retriever Quality**  | Weak                  | Good                  | Best                  | Best                  | Best                  |
| **Instruction Following** | Poor               | Poor                  | Poor                  | Strong                | Strong                |
| **Numeric Accuracy**   | Poor                  | Poor                  | Moderate              | Limited               | High                  |
| **Hallucinations**     | Frequent              | Moderate              | Moderate              | Rare                  | Rare                  |
| **“I don’t know” Usage** | Low                 | Low                   | Low                   | High                  | Balanced              |
| **Overall Performance** | 🚫 Poor              | ⚠️ Fair              | ⚠️+ Improved          | ✅ Strong (Strict)    | 🌟 Optimal            |

---

## 🎯 Conclusions

### Retriever Performance
- **TF-IDF**: Ineffective for semantic retrieval; prone to noise.
- **MiniLM**: Decent semantic matching but limited in precision.
- **MPNet**: Superior for semantic retrieval, consistently delivering relevant documents.

### LLM Performance
- **TinyLlama-1.1B**: Underpowered for factual QA; prone to truncation and hallucinations.
- **FLAN-T5-small**: Despite its smaller size (77M parameters), its instruction-tuning enables robust performance for factual question answering.

### Prompt Design
- **Strict Prompts**: Enhance accuracy but reduce coverage due to frequent "I don’t know" responses.
- **Liberal Prompts**: Balance factuality and flexibility, maximizing answer coverage while maintaining precision.

---

## ✅ Recommended Configuration
**Experiment 5**: FAISS + MPNet + FLAN-T5-small with a liberal prompt  
- **Why**: Achieves the optimal balance of accuracy, coverage, and reliability for lightweight RAG pipelines.
- **Key Strengths**: Precise semantic retrieval, robust instruction-following, and minimal hallucinations.

---

## 🔹 Framework and Modularity
The experiments leveraged **LangChain** to streamline the RAG pipeline, enabling:
- **Modular Design**: Easy swapping of retrievers (TF-IDF, MiniLM, MPNet) and LLMs (TinyLlama, FLAN-T5-small).
- **Prompt Flexibility**: Rapid testing of strict and liberal prompt variations.
- **Efficient Orchestration**: Seamless integration of retrieval and generation steps.
- **Scalability**: Clean, reusable code for future experimentation.

LangChain’s modularity facilitated rapid iteration and ensured maintainable, readable code for reproducible results.

---

## 📌 Key Recommendations
For lightweight, CPU-friendly RAG systems:
- **Retriever**: Use embedding-based retrieval with FAISS and MPNet (`all-mpnet-base-v2`) for superior semantic matching.
- **LLM**: Select an instruction-tuned model like FLAN-T5-small for reliable factual QA.
- **Prompt**: Adopt a liberal prompt to balance strict factuality with broader answer coverage.

This configuration delivers the **optimal trade-off** for accuracy, coverage, and computational efficiency in resource-constrained environments.