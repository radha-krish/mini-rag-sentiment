import json
from pathlib import Path
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import os

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
# Initialize HuggingFace Embeddings (MiniLM)
# -----------------------------
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# -----------------------------
# Build FAISS vector store retriever
# -----------------------------
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# Liberal prompt template for Exp-5
# -----------------------------
prompt_template = """You are a helpful assistant that answers questions using only the provided context. Copy facts from the context exactly. Include all numbers, codes, units, or quantifiable measures present in the context. 
- For numeric, measurable, or coded values, you may slightly shorten phrases to capture all measurable info. 
- For qualitative descriptions or features, use the exact phrase as written in the context. 
Answer rules:
- Output the shortest clear phrase or term that fully answers the question.
- If multiple words are needed, return them as one phrase with commas only if shown in context.
- No sentences, explanations, formatting, lists, repetitions, or extra words.
- If nothing in the context answers the question, output: I don't know.
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
hf_model_id = "google/flan-t5-small"
hf = HuggingFacePipeline.from_model_id(
    model_id=hf_model_id,
    task="text2text-generation",
    device=-1,
    pipeline_kwargs={"max_new_tokens": 20, }
)

# -----------------------------
# Build RetrievalQA chain
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=hf,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# Run QA on all questions
# -----------------------------
results_dict = {}

for q in questions:
    question_id = q["id"]
    question_text = q["question"]

    # Invoke QA
    result = qa.invoke(question_text)

    # Extract answer text
    answer_text = result["result"].strip().split("Answer:")[-1].strip()

    # Store answer
    results_dict[question_id] = answer_text

     # Print for inspection
    print(f"\nQuestion ({question_id}): {question_text}")
    print(f"Answer------: {answer_text}")
    print("Retrieved Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['title']}: {doc.page_content}")



# -----------------------------
# Save answers + experiment metadata
# -----------------------------
experiment_metadata = {
    "experiment": "Exp-5",
    "description": "RAG with FLAN-T5 Small using liberal prompt",
    "documents_count": len(documents),
    "questions_count": len(questions),
    "embedding_model": embeddings_model_name,
    "vector_store": "FAISS",
    "retriever_k": 5,
    "llm_model": hf_model_id,
    "llm_params": {"max_new_tokens": 20, "temperature": 0.6},
    "prompt_type": "liberal, allows slight summarization, still factual",
    
}

output = {
    "answers": results_dict,
    "metadata": experiment_metadata
}

output_path = Path(r"experiments_results/rag_answers_exp5.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved answers and experiment metadata to {output_path}")
