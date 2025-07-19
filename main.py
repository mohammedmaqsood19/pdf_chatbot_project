import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Multi-PDF Query App with LangChain and Groq", layout="wide")
st.title("📄 Multi-PDF Query App with LangChain and Groq")

# Sidebar for API key input, PDF upload, and custom prompt
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    st.subheader("Custom Prompt")
    default_prompt = """You are a helpful assistant answering questions based on the provided PDF documents. Use the following context to answer the question accurately and concisely. If the answer is not found in the context, say so.

Context: {context}

Question: {question}

Answer:"""
    custom_prompt = st.text_area("Customize the Prompt", value=default_prompt, height=200)
    st.markdown("""
    ## About
    This app allows you to upload multiple PDFs, process them, and ask questions about their content using LangChain and Groq's free models.
    - **PDF Processing**: Extracts text, chunks it, and stores embeddings in FAISS.
    - **Querying**: Uses Groq's LLaMA model with a custom prompt to answer questions.
    """)

# Function to process multiple PDFs and create vector store
@st.cache_resource
def process_pdfs(files, _api_key, _prompt):
    all_chunks = []
    
    # Process each PDF
    for file in files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Load PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        except Exception as e:
            st.warning(f"Failed to load {file.name}: {str(e)}. Skipping this file.")
            os.unlink(tmp_file_path)
            continue

        # Check if documents contain text
        if not documents or all(not doc.page_content.strip() for doc in documents):
            st.warning(f"No extractable text found in {file.name}. Skipping this file.")
            os.unlink(tmp_file_path)
            continue

        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.warning(f"No chunks created from {file.name}. Possible empty or invalid content.")
            os.unlink(tmp_file_path)
            continue

        all_chunks.extend(chunks)
        os.unlink(tmp_file_path)

    # Check if any valid chunks were created
    if not all_chunks:
        raise ValueError("No valid text chunks extracted from any of the uploaded PDFs. Please ensure the PDFs contain extractable text.")

    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise ValueError(f"Failed to initialize embeddings: {str(e)}")

    # Create FAISS vector store
    try:
        vector_store = FAISS.from_documents(all_chunks, embeddings)
    except Exception as e:
        raise ValueError(f"Failed to create FAISS vector store: {str(e)}")

    # Initialize Groq model
    try:
        llm = ChatGroq(
            groq_api_key=_api_key,
            model_name="llama3-8b-8192"
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize Groq model: {str(e)}")

    # Define custom prompt template
    try:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=_prompt
        )
    except Exception as e:
        raise ValueError(f"Invalid prompt template: {str(e)}")

    # Create RetrievalQA chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    except Exception as e:
        raise ValueError(f"Failed to create RetrievalQA chain: {str(e)}")
    
    return qa_chain

# Main app logic
def main():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.chat_history = []

    # Process PDFs if uploaded and API key provided
    if uploaded_files and groq_api_key:
        with st.spinner("Processing PDFs..."):
            try:
                st.session_state.qa_chain = process_pdfs(uploaded_files, groq_api_key, custom_prompt)
                st.success(f"Processed {len(uploaded_files)} PDF(s) successfully! You can now ask questions.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

    # Query input and response
    if st.session_state.qa_chain:
        query = st.text_input("Ask a question about the PDFs:", placeholder="e.g., What is the main topic of the documents?")
        
        if query:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    answer = result["result"]
                    sources = result["source_documents"]
                    
                    st.write("**Answer:**")
                    st.write(answer)
                    
                    st.write("**Sources:**")
                    for i, doc in enumerate(sources, 1):
                        source_file = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "Unknown")
                        st.write(f"**Chunk {i} (File: {os.path.basename(source_file)}, Page {page}):**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                    st.session_state.chat_history.append({"query": query, "answer": answer})
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, chat in enumerate(st.session_state.chat_history[::-1], 1):
            st.write(f"**Q{i}:** {chat['query']}")
            st.write(f"**A{i}:** {chat['answer']}")
            st.write("---")

if __name__ == "__main__":
    main()