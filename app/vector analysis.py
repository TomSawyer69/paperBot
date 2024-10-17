from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample paper citations
papers = [
    {"title": "Understanding AI", "authors": "Alice Smith", "doi": "10.1000/xyz123"},
    {"title": "Advances in Machine Learning", "authors": "Bob Johnson", "doi": "10.1000/xyz456"},
    {"title": "Deep Learning for Computer Vision", "authors": "Charlie Brown", "doi": "10.1000/xyz789"},
]

# Sample text chunks for the papers (these should be actual chunks extracted from the papers)
text_chunks = [
    "This paper discusses the fundamental concepts of AI and its applications.",
    "This research presents advances in machine learning techniques.",
    "In this work, we explore deep learning methods for computer vision tasks.",
]

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to embed text chunks
def embed_text_chunks(text_chunks):
    return model.encode(text_chunks, show_progress_bar=True)

# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Function to find the most similar paper for a given prompt
def find_most_similar_paper(prompt_embedding, paper_embeddings, papers):
    max_similarity = -1
    best_paper_index = -1

    for i, paper_embedding in enumerate(paper_embeddings):
        similarity = compute_similarity(prompt_embedding, paper_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_paper_index = i

    return best_paper_index, max_similarity

# Generate embeddings for the text chunks
paper_embeddings = embed_text_chunks(text_chunks)

# Example user prompt
user_prompt = "What are the latest advancements in AI?"
user_prompt_embedding = model.encode(user_prompt)

# Find the most similar paper
best_paper_index, similarity_score = find_most_similar_paper(user_prompt_embedding, paper_embeddings, papers)

# Print the result
if best_paper_index != -1:
    best_paper = papers[best_paper_index]
    print(f"Most Similar Paper: {best_paper['title']}")
    print(f"Authors: {best_paper['authors']}")
    print(f"DOI: {best_paper['doi']}")
    print(f"Similarity Score: {similarity_score}")
else:
    print("No similar papers found.")
