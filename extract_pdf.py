import os
import PyPDF2

PDF_DIR = "pdf's"
OUTPUT_DIR = "data"

def extract_text_from_pdf(pdf_path):
    """Extract all text from a single PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_all_pdfs():
    # Check if the PDFs folder exists
    if not os.path.exists(PDF_DIR):
        print(f"❌ Folder '{PDF_DIR}' not found. Please create it and add some PDF files.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process each PDF in the folder
    files_processed = 0
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"📄 Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                output_filename = os.path.splitext(filename)[0] + ".txt"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Saved text to: {output_path}")
                files_processed += 1
            else:
                print(f"⚠️ No text found in: {filename}")

    if files_processed == 0:
        print("⚠️ No valid PDF files were found or processed.")

if __name__ == "__main__":
    process_all_pdfs()
