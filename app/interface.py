import streamlit as st
import zipfile
import shutil
import fitz  # PyMuPDF
from PIL import Image
import io
from tabula import read_pdf
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

# Define the folder where files will be uploaded
UPLOAD_FOLDER = 'resources\\uploads'
QUERY_FILE = os.path.join(UPLOAD_FOLDER, 'query.txt')

def io():
    # Function to upload and extract zip files
    def upload_zip_file():
        """Upload a zip file and extract its contents to the upload folder."""
        # Clear the upload directory
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)
        st.success("Previous upload session cleared, proceed to upload.")

        uploaded_file = st.file_uploader("Upload a zip file with PDFs", type="zip")
        if uploaded_file:
            zip_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
            st.success("Uploaded and extracted successfully!")

            return True  # Indicate that upload is complete
        return False

    # Display the query and the output file
    def display_output_file():
        reply_file = '../resources/reply.txt'
        if os.path.exists(reply_file):
            with open(reply_file, 'r', encoding='utf-8') as f:
                output = f.read()
                st.text_area("Generated Answer:", output)

    # Page layout and title
    st.set_page_config(page_title="File Upload and paperBot", page_icon="📂", layout="centered")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .title {
                text-align: center;
                color: #4CAF50;
                font-size: 2.5em;
                margin-bottom: 20px;
            }
            .subtitle {
                text-align: center;
                color: #555;
                font-size: 1.2em;
                margin-bottom: 40px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Navigation for multi-page app
    page = st.sidebar.selectbox("Select Page", ("Upload Files", "Send Query"))

    if page == "Upload Files":
        # Title and description for file upload page
        st.markdown('<div class="title">File Upload</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Upload your zip file containing PDFs below.</div>', unsafe_allow_html=True)

        # Call the upload function and proceed if upload is successful
        if upload_zip_file():
            # Run the entire pipeline after successful upload
            ce()  # Run the content extraction
            eg()  # Generate and save embeddings
            qr()  # Run the question-answering process
            display_output_file()  # Display the final output

    elif page == "Send Query":
        # Title and description for paperBot page
        st.markdown('<div class="title">paperBot</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Ask your question below:</div>', unsafe_allow_html=True)

        # User query input
        user_query = st.text_input("Your Question:", placeholder="Type your question here...")

        # Submit button for user query
        if st.button("Submit"):
            if user_query:
                with open(QUERY_FILE, "a") as f:  # Use "a" to append
                    f.write(user_query + "\n")
                st.success("Query Added")

                # Run the entire pipeline after query submission
                ce()  # Run the content extraction
                eg()  # Generate and save embeddings
                qr()  # Run the question-answering process
                display_output_file()  # Display the final output
            else:
                st.warning("Please enter a question before submitting.")

def ce():
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
        return text_chunks, full_text

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
    uploads_dir = "../resources/uploads/papers"
    extracted_dir = "../resources/extracted"

    # Ensure the extracted directory exists
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)

    # Process each PDF file in the uploads directory
    for pdf_file in os.listdir(uploads_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(uploads_dir, pdf_file)

            image_chunks = extract_pdf_image(pdf_path)
            text_chunks, text = extract_pdf_text(pdf_path)
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

            # Save tables
            for i, table in enumerate(table_chunks):
                table.to_csv(f"{pdf_extracted_path}/tables/{pdf_base_name}_table_{i + 1}.csv", index=False)

            # Save text chunks with a special delimiter
            delimiter = "<|endofchunk|>"
            with open(f"{pdf_extracted_path}/text/{pdf_base_name}_text.txt", "w", encoding="utf-8") as text_file:
                text_file.write(delimiter.join(text_chunks))  # Join chunks with the delimiter

            with open(f"{pdf_extracted_path}/text/{pdf_base_name}_text_unchunked.txt", "w", encoding="utf-8") as text_file:
                text_file.write(text)

            # Create and save metadata
            metadata_content = f"""
    PDF Filename: {pdf_base_name}
    Number of Images: {len(image_chunks)}
    Number of Tables: {len(table_chunks)}
    Number of Text Chunks: {len(text_chunks)}
    """
            with open(f"{pdf_extracted_path}/metadata.txt", "w", encoding="utf-8") as metadata_file:
                metadata_file.write(metadata_content.strip())

def eg():
    def load_pdf_chunks(pdf_txt_path):
        chunks = []
        if os.path.exists(pdf_txt_path):
            with open(pdf_txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
                pdf_chunks = text.split('<|endofchunk|>')
                chunks = [chunk.strip() for chunk in pdf_chunks if chunk.strip()]
        return chunks

    def generate_and_save_embeddings(extracted_dir, save_dir='..\\resources\\embeddings',
                                     model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for pdf_name in os.listdir(extracted_dir):
            pdf_text_path = os.path.join(extracted_dir, pdf_name, 'text', f'{pdf_name}_text.txt')
            if os.path.exists(pdf_text_path):
                chunks = load_pdf_chunks(pdf_text_path)
                if chunks:
                    # Generate embeddings for each chunk
                    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

                    # Save embeddings as .npy file for each PDF
                    np.save(os.path.join(save_dir, f'{pdf_name}_embeddings.npy'), embeddings)

                    # Optionally save metadata (e.g., chunk text or indexes)
                    with open(os.path.join(save_dir, f'{pdf_name}_chunks.txt'), 'w', encoding='utf-8') as f:
                        for chunk in chunks:
                            f.write(chunk + '\n<|endofchunk|>\n')

                    print(f'Embeddings and chunks saved for {pdf_name}')

    # Example usage:
    extracted_dir = '..\\resources\\extracted'
    os.makedirs(extracted_dir, exist_ok=True)
    generate_and_save_embeddings(extracted_dir)

def qr():
    qa_model_name = 'distilbert-base-uncased-distilled-squad'
    qa_pipeline = pipeline("question-answering", model=qa_model_name)

    # Load the summarization model to condense the answers
    summarization_model_name = 'facebook/bart-large-cnn'
    summarization_pipeline = pipeline("summarization", model=summarization_model_name)

    def read_chunks_from_file(filename):
        # Adjust the path to where your extracted text chunks are stored
        file_path = os.path.join('..', 'resources', 'extracted', filename, 'text', f'{filename}_text.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.split('<|endofchunk|>')

    def parse_chunks_with_delimiter(filename, delimiter="<|endofchunk|>"):
        """Parse the input text and extract chunks based on the given delimiter."""
        chunks = read_chunks_from_file(filename)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty chunks
        return [(i + 1, chunk) for i, chunk in enumerate(chunks)]  # Assign chunk numbers

    def get_abstract_chunks(chunks):
        """Get the initial chunks for the abstract."""
        abstract_chunks = []

        # Assume first few chunks (0, 1, 2) form the abstract
        for i in range(min(3, len(chunks))):
            abstract_chunks.append(chunks[i][1])  # Collect content of chunk i

        print("\nAbstract Chunks Selected:")
        for idx, chunk in enumerate(abstract_chunks):
            print(f"Chunk {idx}: {chunk[:100]}...")  # Log abstract chunks (truncated)

        return " ".join(abstract_chunks)

    def get_contextual_chunks(chunks, best_index, context_range=1):
        """Get the best chunk with its surrounding chunks (previous and next)."""
        start = max(0, best_index - context_range)  # Ensure we don't go out of bounds
        end = min(len(chunks), best_index + context_range + 1)  # Ensure we don't exceed chunk list length
        selected_chunks = chunks[start:end]

        # Log the surrounding chunks being included
        print(f"\nSelected chunks for final context (Chunks {selected_chunks[0][0]} to {selected_chunks[-1][0]}):")
        for idx, (chunk_number, chunk) in enumerate(selected_chunks):
            print(f"Chunk {chunk_number}: {chunk[:100]}...")  # Log chunk number and truncated content

        return " ".join([chunk[1] for chunk in selected_chunks])

    def process_chunks_with_context(file_name, user_prompt, best_index, context_range=1):
        """Process text chunks, find the best chunk, append surrounding context, and answer the prompt."""

        chunks = parse_chunks_with_delimiter(file_name)  # Extract chunks using delimiter

        print(f"Processing {len(chunks)} chunks...\n")

        # Get the initial chunks for the abstract
        abstract_text = get_abstract_chunks(chunks)

        # Retrieve surrounding chunks (previous and next based on context_range)
        contextual_text = get_contextual_chunks(chunks, best_index, context_range)

        # Combine abstract chunks with selected chunks
        final_context = abstract_text + " " + contextual_text

        # Log the full concatenated context
        print("\nFinal concatenated context for summarization:")
        print(final_context[:500] + "...")  # Truncate for neatness

        # Answer based on the best chunk plus context
        final_answer = qa_pipeline(question=user_prompt, context=final_context)

        # Summarize the full combined answer (sum of all selected chunks)
        summarized_answer = summarization_pipeline(
            final_context,
            max_length=200,  # Adjust max length of summary if needed
            min_length=50,
            do_sample=False
        )

        # Log the final summarized answer
        print("\nFinal Summarized Answer:")
        print(summarized_answer[0]['summary_text'])

        return summarized_answer[0]['summary_text']

    with open('../resources/uploads/query.txt', 'r') as file:
        lines = file.readlines()

    query = lines[0].strip()
    query = query + ". Give a properly structured output."  # Add custom prompt here

    pdf_name = None
    chunk_idx = None

    # Read and process the file
    with open('../resources/out/out1.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()  # Remove leading/trailing whitespace
            if stripped_line:  # Check if the line is not empty
                # Split the line into parts
                parts = stripped_line.split(', ')
                pdf_name = parts[0].split(': ')[1]  # Get the PDF name
                chunk_idx = int(parts[1].split(': ')[1])  # Get the chunk index

    # Generate the summary with context
    final_output = process_chunks_with_context(pdf_name, query, chunk_idx)
    print("\nFinal Answer:", final_output)

    reply_file = '../resources/reply.txt'
    with open(reply_file, 'w', encoding='utf-8') as f:
        f.write(f"{final_output}")


io()