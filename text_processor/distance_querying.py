import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load embeddings and metadata (chunks)
def load_embeddings_and_metadata(embedding_dir):
    all_embeddings = []
    metadata = []  # To store the source (PDF name) and chunk index

    for embedding_file in os.listdir(embedding_dir):
        if embedding_file.endswith('_embeddings.npy'):
            embeddings = np.load(os.path.join(embedding_dir, embedding_file))
            pdf_name = embedding_file.replace('_embeddings.npy', '')

            # Load chunk metadata
            with open(os.path.join(embedding_dir, f'{pdf_name}_chunks.txt'), 'r', encoding='utf-8') as f:
                chunks = f.read().split('<|endofchunk|>\n')

            # Store embeddings and associated metadata
            all_embeddings.append(embeddings)
            metadata.extend([(pdf_name, i, chunk) for i, chunk in enumerate(chunks) if chunk.strip()])

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings, metadata


# Build FAISS index
def build_faiss_index(embeddings, d):
    index = faiss.IndexFlatL2(d)  # Using L2 (Euclidean) distance
    index.add(embeddings)  # Add embeddings to the FAISS index
    return index


# Query FAISS index
def query_faiss_index(query, index, model, metadata, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    # Retrieve the top-k results
    results = [(metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results


# Save average distance for each PDF to a text file
def save_avg_distance_to_file(results, query, output_file='..\\resources\\avg_distances.txt'):
    pdf_distances = {}

    # Aggregate distances for each PDF
    for (pdf_name, _, _), distance in results:
        if pdf_name in pdf_distances:
            pdf_distances[pdf_name].append(distance)
        else:
            pdf_distances[pdf_name] = [distance]

    # Calculate average distances
    avg_distances = {pdf_name: np.mean(distances) for pdf_name, distances in pdf_distances.items()}

    # Write query and averages to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n\n")  # Write the query at the top
        for pdf_name, avg_distance in avg_distances.items():
            f.write(f"PDF: {pdf_name}, Average Distance: {avg_distance:.4f}\n")

# Example usage
embedding_dir = '..\\resources\\embeddings'
extracted_dir = '..\\resources\\extracted'
out_dir = '..\\resources\\out'

os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(extracted_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
# Step 1: Load embeddings and metadata
all_embeddings, metadata = load_embeddings_and_metadata(embedding_dir)

# Step 2: Initialize FAISS index
embedding_dimension = all_embeddings.shape[1]
faiss_index = build_faiss_index(all_embeddings, embedding_dimension)

# Step 3: Query the index
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('../resources/out/query.txt', 'r') as file:
    lines = file.readlines()
# Extract the query
query = lines[0].strip().split(": ", 1)[1]
results = query_faiss_index(query.strip(), faiss_index, model, metadata)

for i, ((pdf_name, chunk_idx, chunk_text), distance) in enumerate(results):
    out_file = os.path.join(out_dir, f'out{i + 1}.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"PDF: {pdf_name}, chunk: {chunk_idx}\n")
# save_avg_distance_to_file(results, query)