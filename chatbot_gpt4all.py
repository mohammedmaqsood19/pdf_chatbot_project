# import streamlit as st
# from gpt4all import GPT4All
# import os

# # Path to your model (adjust if needed)
# MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# # Load the GPT4All model
# @st.cache_resource  # prevent reloading on every interaction
# def load_model():
#     return GPT4All(model_name=MODEL_PATH)

# model = load_model()

# # Load all chunked documents into memory
# def load_documents():
#     chunks = []
#     data_dir = "data_chunks"
#     for filename in os.listdir(data_dir):
#         if filename.endswith("_chunked.txt"):
#             with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
#                 chunks.append(f.read())
#     return "\n".join(chunks)

# document_context = load_documents()

# # UI
# st.title("📚 PDF Chatbot (using GPT4All)")
# query = st.text_input("Ask a question about your documents:")

# if query:
#     with st.spinner("Thinking..."):
#         prompt = f"Use the following context to answer the question:\n\n{document_context}\n\nQuestion: {query}\nAnswer:"
#         response = model.generate(prompt, max_tokens=300)
#         st.markdown("### 🤖 Answer:")
#         st.write(response)
import streamlit as st
from gpt4all import GPT4All
from pdf_utils import get_pdf_chunks, build_faiss_index, search_similar_chunks

st.set_page_config(page_title="📄 PDF Chatbot with GPT4All", layout="wide")
st.title("📄 PDF Chatbot (GPT4All + Embeddings)")

pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=["pdf"])
query = st.text_input("Ask a question about your documents:")

@st.cache_resource
def load_model():
    return GPT4All(
        model_name="mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # file name only
        model_path="models"  # just the folder, not the full path
    )
# def load_model():
#     return GPT4All(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

def ask_gpt4all(model, question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
You are an AI assistant. Use the following context from a PDF to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate(prompt)
    return response

if st.button("Get Answer"):
    if pdf_docs and query:
        with st.spinner("Processing documents..."):
            chunks = get_pdf_chunks(pdf_docs)
            index, embeddings, all_chunks = build_faiss_index(chunks)
            top_chunks = search_similar_chunks(query, index, all_chunks, embeddings)
            model = load_model()
            answer = ask_gpt4all(model, query, top_chunks)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please upload PDFs and enter a question.")