# 🧪 RAG Experiments Report (Exp-1 → Exp-5)

## 🎯 Goal
To evaluate **lightweight Retrieval-Augmented Generation (RAG) pipelines** with small LLMs.  
Focus areas:
- Retrieval quality (keyword vs. embeddings)
- Model compliance with instructions
- Answer precision (copying exact phrases vs. hallucinations)
- Impact of prompt design  

**Corpus:** 15 documents (`docs.jsonl`)  
**Queries:** 15 questions (`questions.json`)

---

## 📊 Experiment Summaries

### **Experiment 1 – TF-IDF + TinyLlama**
- **Retriever:** TF-IDF (k=5)  
- **LLM:** TinyLlama-1.1B  
- **Prompt:** Strict copy or *“I don’t know”*  
- **Settings:** `max_new_tokens=20, temp=0.01`

**Observations**
- ❌ Weak retrieval: missed semantic matches.
- ❌ TinyLlama too small: poor instruction-following, truncated outputs.
- ❌ Prompt fragility: echoed rules, cut answers.  

**Takeaway:**  
TF-IDF not reliable for semantic facts. TinyLlama struggles with strict factual QA.

---

### **Experiment 2 – FAISS + MiniLM + TinyLlama**
- **Retriever:** FAISS + MiniLM (all-MiniLM-L6-v2)  
- **LLM:** TinyLlama-1.1B  
- **Prompt:** Strict copy or *“I don’t know”*  

**Observations**
- ✅ Retrieval better: semantically relevant docs.  
- ✅ Fewer repetitions.  
- ❌ Still weak on numeric/config facts.  
- ❌ Truncation & hallucinations remain.  

**Takeaway:**  
FAISS + MiniLM > TF-IDF for general retrieval, but TinyLlama is the bottleneck.

---

### **Experiment 3 – FAISS + MPNet + TinyLlama**
- **Retriever:** FAISS + MPNet (all-mpnet-base-v2)  
- **LLM:** TinyLlama-1.1B  

**Observations**
- ✅ Improved semantic capture (e.g., correct VulcanGraph port: `7787`).  
- ✅ Less redundancy.  
- ❌ Still paraphrasing mistakes & truncations.  
- ❌ Sparse facts depend on `top-k`.  

**Takeaway:**  
MPNet > MiniLM for retrieval, but TinyLlama remains underpowered.

---

### **Experiment 4 – FAISS + MPNet + FLAN-T5-small**
- **Retriever:** FAISS + MPNet  
- **LLM:** FLAN-T5-small (77M, instruction-tuned)  
- **Prompt:** Strict copy or *“I don’t know”*  

**Observations**
- ✅ Excellent instruction-following, minimal hallucinations.  
- ✅ More reliable than TinyLlama despite being smaller.  
- ❌ Too strict: frequent *“I don’t know”*.  
- ❌ Retrieval failure = answer failure.  

**Takeaway:**  
FLAN-T5-small > TinyLlama for factual QA, but overly strict prompt reduced coverage.

---

### **Experiment 5 – FAISS + MPNet + FLAN-T5-small (Liberal Prompt)**
- **Retriever:** FAISS + MPNet  
- **LLM:** FLAN-T5-small  
- **Prompt:** Liberal factual prompt (exact phrases for features, slight flexibility for numeric values).  

**Observations**
- ✅ Balanced strictness & flexibility → fewer unnecessary *“I don’t know”*.  
- ✅ Precise answers (e.g., `port 7787`, *“property graph database with Gremlin-like traversal”*).  
- ✅ Best coverage for both qualitative & numeric facts.  
- ❌ Still some truncation for complex numeric schemes.  
- ❌ Retrieval errors remain a bottleneck.  

**Takeaway:**  
Liberal prompt unlocked FLAN-T5’s potential → best overall setup.

---

## 📈 Comparative Overview

| Aspect                 | Exp-1       | Exp-2            | Exp-3            | Exp-4              | Exp-5               |
|-------------------------|-------------|------------------|------------------|--------------------|---------------------|
| **Retriever**           | TF-IDF      | FAISS + MiniLM   | FAISS + MPNet    | FAISS + MPNet      | FAISS + MPNet       |
| **LLM**                 | TinyLlama   | TinyLlama        | TinyLlama        | FLAN-T5-small      | FLAN-T5-small       |
| **Prompt**              | Strict      | Strict           | Strict           | Strict             | Liberal             |
| **Retriever Quality**   | Weak        | Good             | Best             | Best               | Best                |
| **Instruction Following** | Poor      | Poor             | Poor             | Strong             | Strong              |
| **Numeric Accuracy**    | Fails       | Fails            | Sometimes        | Missed literals    | Improved            |
| **Hallucinations**      | Frequent    | Some             | Some             | Rare               | Rare                |
| **“I don’t know” Usage** | Low        | Low              | Low              | High               | Balanced            |
| **Overall Performance** | 🚫          | ⚠️               | ⚠️+              | ✅ (strict)        | 🌟 Best balance     |

---

## 🎯 Final Conclusions

### 🔹 Retriever Evolution
- **TF-IDF** → noisy, weak.  
- **MiniLM** → decent but limited.  
- **MPNet** → best semantic retrieval.  

### 🔹 LLM Role
- **TinyLlama-1.1B** → too small, weak for factual QA.  
- **FLAN-T5-small** → far smaller, but instruction-tuned → much better.  

### 🔹 Prompting Matters
- **Strict prompts** → higher accuracy, lower coverage.  
- **Liberal prompts** → balance between factuality & usability.  

---

## ✅ Best Setup (Exp-5)
**FAISS + MPNet + FLAN-T5-small + Liberal Prompt**  
→ Most practical and precise balance of **accuracy, coverage, and reliability**.  

---

---

## 🔹 Framework & Modularity

For these experiments, I **used LangChain** to manage the RAG pipeline.  
This enabled:
- Modular and reusable code for different retrievers and LLMs
- Easy swapping of embeddings (MiniLM, MPNet) and LLMs (TinyLlama, FLAN-T5-small)
- Rapid experimentation with prompt styles and retrieval strategies
- Cleaner orchestration of retrieval + generation steps  

Using LangChain allowed me to **iterate quickly**, test multiple configurations, and maintain **readable, reusable code** for future RAG experiments.


### 📌 Key Lesson
For **lightweight RAG**:
- **Retriever:** Embedding-based (FAISS + MPNet)  
- **LLM:** Instruction-tuned (FLAN-T5-small)  
- **Prompt:** Balanced liberal prompt  

This combination gives the **sweet spot** for factual QA with small models.
