import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

# load .env
load_dotenv()

# ---------------- CONFIG ----------------
DB_FAISS_PATH = "vectorstore/db_faiss"
LLM_MODEL = "google/flan-t5-base"
# --------------------------------------

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
db = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM via transformers (STABLE)
hf_pipeline = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("\n✅ PDF Chat Ready (type exit to quit)\n")

while True:
    query = input("Write Query Here: ")
    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    if not docs:
        print("❌ No relevant content found in PDF")
        continue

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\n🧠 ANSWER:\n", response)
    print("-" * 60)
