import os

INPUT_DIR = "data"
OUTPUT_DIR = "data_chunks"

CHUNK_SIZE = 500      # Number of characters in each chunk
CHUNK_OVERLAP = 100   # Overlap between chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def process_text_files():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Folder '{INPUT_DIR}' not found. Please run extract step first.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_DIR, filename)
            print(f"📄 Chunking: {filename}")
            
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)

            output_filename = os.path.splitext(filename)[0] + "_chunked.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "w", encoding="utf-8") as out:
                for i, chunk in enumerate(chunks, 1):
                    out.write(f"--- Chunk {i} ---\n{chunk}\n\n")

            print(f"✅ Saved chunks to: {output_path}")

if __name__ == "__main__":
    process_text_files()
