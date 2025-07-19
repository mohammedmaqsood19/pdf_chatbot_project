import streamlit as st
import requests
from pdf_utils import get_pdf_chunks, build_faiss_index, search_similar_chunks

st.set_page_config(page_title="📄 PDF Chatbot with Ollama", layout="wide")
st.title("📄 PDF Chatbot (Ollama + Embeddings)")

pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Ask a question about your documents:")

def ask_ollama(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
You are an AI assistant. Use the following context from a PDF to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    result = response.json()
    return result.get("response", "No response")

if st.button("Get Answer"):
    if pdf_docs and query:
        with st.spinner("Processing documents..."):
            chunks = get_pdf_chunks(pdf_docs)
            index, embeddings, all_chunks = build_faiss_index(chunks)
            top_chunks = search_similar_chunks(query, index, all_chunks, embeddings)
            answer = ask_ollama(query, top_chunks)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please upload PDFs and enter a question.")