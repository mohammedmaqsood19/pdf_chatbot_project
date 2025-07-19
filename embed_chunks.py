import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHUNK_DIR = "data_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"

def load_chunks():
    documents = []
    ids = []
    sources = []

    for filename in os.listdir(CHUNK_DIR):
        if filename.endswith("_chunked.txt"):
            file_path = os.path.join(CHUNK_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split on --- Chunk N ---
            raw_chunks = [chunk.strip() for chunk in content.split("--- Chunk") if chunk.strip()]
            for i, chunk in enumerate(raw_chunks):
                chunk_text = chunk.split("---", 1)[-1].strip()  # remove label
                doc_id = f"{filename}_chunk_{i+1}"
                documents.append(chunk_text)
                ids.append(doc_id)
                sources.append(filename)

    return ids, documents, sources


def embed_and_store():
    print("📦 Loading model and chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    ids, texts, metadatas = load_chunks()

    print(f"🔢 Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts).tolist()

    print("🗃️ Initializing ChromaDB...")
    client = chromadb.Client(Settings(
        persist_directory="chroma_db",  # Saves database to disk
        anonymized_telemetry=False
    ))

    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("💾 Inserting into ChromaDB...")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": src} for src in metadatas]
    )

    print(f"✅ Stored {len(texts)} documents in collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    embed_and_store()
