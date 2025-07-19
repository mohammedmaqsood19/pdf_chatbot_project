# import os
# import streamlit as st
# import google.generativeai as genai

# # Set your Gemini API key
# genai.configure(api_key="AIzaSyARd592ttSqSxIUbucFEs4heXv3gWWkn-A")  # Replace with your actual Gemini API key

# # Load all text chunks from the directory
# CHUNK_DIR = "data_chunks"

# def load_documents():
#     all_text = []
#     for file in os.listdir(CHUNK_DIR):
#         if file.endswith(".txt"):
#             with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
#                 all_text.append(f.read())
#     return "\n".join(all_text)

# # Function to query Gemini
# def ask_gemini(question, context):
#     model = genai.GenerativeModel(model_name="gemini-1.5-pro")  # <- Use the latest supported model name
#     response = model.generate_content(
#         f"Use the following document context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}"
#     )
#     return response.text

# # Streamlit UI
# st.title("📚 PDF Chatbot (Gemini API)")

# query = st.text_input("Ask a question about your documents:")

# if query:
#     with st.spinner("Thinking..."):
#         context_text = load_documents()
#         answer = ask_gemini(query, context_text)
#         st.success(answer)
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pdf_utils import get_pdf_chunks, build_faiss_index, search_similar_chunks

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="📄 PDF Chatbot with Gemini", layout="wide")
st.title("📄 PDF Chatbot (Gemini + Embeddings)")

pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Ask a question about your documents:")

model = genai.GenerativeModel("gemini-pro")

def ask_gemini(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
You are an AI assistant. Use the following context from a PDF to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text

if st.button("Get Answer"):
    if pdf_docs and query:
        with st.spinner("Processing documents..."):
            chunks = get_pdf_chunks(pdf_docs)
            index, embeddings, all_chunks = build_faiss_index(chunks)
            top_chunks = search_similar_chunks(query, index, all_chunks, embeddings)
            answer = ask_gemini(query, top_chunks)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please upload PDFs and enter a question.")