import os
from sentence_transformers import SentenceTransformer
import numpy as np


def load_pdf_chunks(pdf_txt_path):
    chunks = []
    if os.path.exists(pdf_txt_path):
        with open(pdf_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            pdf_chunks = text.split('<|endofchunk|>')
            chunks = [chunk.strip() for chunk in pdf_chunks if chunk.strip()]
    return chunks


def generate_and_save_embeddings(extracted_dir, save_dir='..\\embeddings', model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("here")
    for pdf_name in os.listdir(extracted_dir):
        pdf_text_path = os.path.join(extracted_dir, pdf_name, 'text', f'{pdf_name}_text.txt')
        print(pdf_text_path)
        if os.path.exists(pdf_text_path):

            chunks = load_pdf_chunks(pdf_text_path)
            if chunks:
                embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
                np.save(os.path.join(save_dir, f'{pdf_name}_embeddings.npy'), embeddings)
                print(f'Embeddings saved for {pdf_name}')


# Example usage:
extracted_dir = '..\\extracted'
generate_and_save_embeddings(extracted_dir)
