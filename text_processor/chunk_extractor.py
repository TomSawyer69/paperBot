import fitz  # PyMuPDF
from PIL import Image
import io
import os
from tabula import read_pdf


# Function to extract images from PDF
def extract_pdf_image(pdf_path):
    pdf_document = fitz.open(pdf_path)
    image_chunks = []
    for page in pdf_document:
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_data = base_image['image']
            image_chunks.append((img_index, Image.open(io.BytesIO(image_data))))  # Store as PIL image
    return image_chunks


# Function to extract text from PDF
def extract_pdf_text(pdf_path, chunk_size=512):
    pdf_document = fitz.open(pdf_path)
    full_text = ""
    # Extract text from each page of the PDF
    for page in pdf_document:
        full_text += page.get_text()
    # Remove any leading/trailing whitespace
    full_text = full_text.strip()
    # Split the full text into chunks of the specified size
    text_chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return text_chunks


# Function to extract tables from PDF
def extract_pdf_tables(pdf_path):
    # Use tabula to read tables from the PDF
    tables = read_pdf(pdf_path, pages='all', multiple_tables=True)
    table_chunks = []
    for i, table in enumerate(tables):
        # Convert each table to a DataFrame and store it
        table_chunks.append(table)
    return table_chunks


# Path to the directory containing PDFs
uploads_dir = "../uploads/papers"
extracted_dir = "../extracted"

# Ensure the extracted directory exists
os.makedirs(extracted_dir, exist_ok=True)

# Process each PDF file in the uploads directory
for pdf_file in os.listdir(uploads_dir):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(uploads_dir, pdf_file)

        image_chunks = extract_pdf_image(pdf_path)
        text_chunks = extract_pdf_text(pdf_path)
        table_chunks = extract_pdf_tables(pdf_path)

        pdf_base_name = os.path.splitext(pdf_file)[0]
        pdf_extracted_path = os.path.join(extracted_dir, pdf_base_name)

        # Create directories for saving files
        os.makedirs(f"{pdf_extracted_path}/images", exist_ok=True)
        os.makedirs(f"{pdf_extracted_path}/tables", exist_ok=True)
        os.makedirs(f"{pdf_extracted_path}/text", exist_ok=True)

        # Save images
        total_images = len(image_chunks)
        for i, (img_index, img) in enumerate(image_chunks):
            img.save(f"{pdf_extracted_path}/images/{pdf_base_name}_image_{i + 1}.png")
            print(
                f"Saved Image {i + 1}/{total_images} as {pdf_extracted_path}/images/{pdf_base_name}_image_{i + 1}.png")

        # Save tables
        for i, table in enumerate(table_chunks):
            table.to_csv(f"{pdf_extracted_path}/tables/{pdf_base_name}_table_{i + 1}.csv", index=False)
            print(
                f"Saved Table {i + 1}/{len(table_chunks)} as {pdf_extracted_path}/tables/{pdf_base_name}_table_{i + 1}.csv")

        # Save text chunks with a special delimiter
        delimiter = "<|endofchunk|>"
        with open(f"{pdf_extracted_path}/text/{pdf_base_name}_text.txt", "w", encoding="utf-8") as text_file:
            text_file.write(delimiter.join(text_chunks))  # Join chunks with the delimiter
            print(f"Saved text as {pdf_extracted_path}/text/{pdf_base_name}_text.txt")

        # Create and save metadata
        metadata_content = f"""
PDF Filename: {pdf_base_name}
Number of Images: {len(image_chunks)}
Number of Tables: {len(table_chunks)}
Number of Text Chunks: {len(text_chunks)}
"""
        with open(f"{pdf_extracted_path}/metadata.txt", "w", encoding="utf-8") as metadata_file:
            metadata_file.write(metadata_content.strip())  # Save metadata without extra new lines
            print(f"Saved metadata as {pdf_extracted_path}/metadata.txt")
