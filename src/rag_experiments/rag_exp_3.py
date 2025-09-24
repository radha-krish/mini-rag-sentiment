import json
from pathlib import Path
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import os

# -----------------------------
# HuggingFace API token
# -----------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# -----------------------------
# Load documents
# -----------------------------
docs_path = Path(r"data/corpus/docs.jsonl")
documents = []

with docs_path.open() as f:
    for line in f:
        obj = json.loads(line)
        documents.append(
            Document(
                page_content=obj["text"],
                metadata={"id": obj["id"], "title": obj["title"]}
            )
        )

print(f"Loaded {len(documents)} documents.")

# -----------------------------
# Load questions
# -----------------------------
questions_path = Path("data/corpus/questions.json")
with questions_path.open() as f:
    questions = json.load(f)

# -----------------------------
# Initialize HuggingFace Embeddings (MPNet)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# -----------------------------
# Build FAISS vector store retriever
# -----------------------------
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# Prompt template to reduce hallucination
# -----------------------------
prompt_template = """You are a helpful assistant that answers questions strictly using only the provided context. Copy facts exactly as written from the context. Do not infer, paraphrase, summarize, expand, or add anything.
Answer rules:
Output the shortest exact phrase/term from context that directly matches the question.
If multiple words are needed, return them as one phrase with commas only if shown in context (e.g., "tiny, vision, model").
No sentences, no explanations, no formatting, no extra words, no lists, no repetitions.
If nothing matches exactly, output: I don't know.
If the context is missing, empty, or does not explicitly contain the exact answer, output ONLY: I don't know. Do not guess, approximate, or derive anything
Context:
{context}
Question:
{question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# -----------------------------
# Initialize HuggingFace Pipeline LLM
# -----------------------------
hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,
    pipeline_kwargs={"max_new_tokens": 20,}
)

# -----------------------------
# Build RetrievalQA chain with custom prompt
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=hf,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# Prepare results dictionary with experiment metadata
# -----------------------------
results_dict = {
    "experiment_details": {
        "name": "Experiment-3",
        "objective": "Use FAISS retriever with MPNet embeddings + TinyLlama-1.1B for RAG",
        "corpus": "15 documents (docs.jsonl)",
        "questions": "15 questions (questions.json)",
        "retriever": "FAISS vector store using sentence-transformers/all-mpnet-base-v2 embeddings",
        "LLM": "TinyLlama-1.1B-Chat-v1.0 via HuggingFace pipeline",
        "prompt": "Strict copy-from-context instructions, output shortest exact phrase, or 'I don't know'",
    },
    "answers": {}
}

# -----------------------------
# Run QA on all questions and store results
# -----------------------------
for q in questions:
    question_id = q["id"]
    question_text = q["question"]

    # Invoke QA
    result = qa.invoke(question_text)

    # Extract answer text
    answer_text = result["result"].strip().split("Answer:")[-1].strip()

    # Store answer
    results_dict["answers"][question_id] = answer_text

    # Print for inspection
    print(f"\nQuestion ({question_id}): {question_text}")
    print(f"Answer------: {answer_text}")
    print("Retrieved Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['title']}: {doc.page_content}")

# -----------------------------
# Write results to JSON file
# -----------------------------
output_path = Path(r"experiments_results/rag_answers_exp_3.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\nSaved answers for {len(results_dict['answers'])} questions to {output_path}")
