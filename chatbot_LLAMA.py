import streamlit as st
from llama_cpp import Llama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# ==== Configuration ====
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
MAX_CHUNKS = 4


# ==== Load Local LLM ====
@st.cache_resource
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,     # Adjust to your CPU cores
        n_batch=8,
        verbose=False
    )

# ==== Load ChromaDB & Embedder ====
@st.cache_resource
def load_db():
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DB_DIR,
        chroma_db_impl="duckdb+parquet"
    ))
    collection = client.get_collection(COLLECTION_NAME)
    return collection

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==== Query Chunks from Chroma ====
def get_top_chunks(query, embedder, collection):
    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=MAX_CHUNKS
    )
    return results["documents"][0]

# ==== Generate Response ====
def ask_llm(context, question, llm):
    prompt = f"""[INST] Answer the question based only on the context below.
    
Context:
{context}

Question: {question}
[/INST]
"""
    output = llm(prompt, max_tokens=512, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# ==== Streamlit UI ====
st.title("📄 Local PDF Chatbot")
st.markdown("Ask questions from your local PDF documents")

question = st.text_input("💬 Enter your question")

if question:
    with st.spinner("🤖 Thinking..."):
        llm = load_llm()
        embedder = load_embedder()
        collection = load_db()

        top_chunks = get_top_chunks(question, embedder, collection)
        context = "\n".join(top_chunks)
        response = ask_llm(context, question, llm)

    st.markdown("### 📚 Answer:")
    st.write(response)

    with st.expander("🔍 Retrieved Chunks"):
        st.code(context)
