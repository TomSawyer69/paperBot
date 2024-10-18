from transformers import pipeline
import os

# Load the question answering model for extracting answers
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
final_output = process_chunks_with_context(pdf_name, query,  chunk_idx)
print("\nFinal Answer:", final_output)

reply_file = '../resources/reply.txt'
with open(reply_file, 'w', encoding='utf-8') as f:
    f.write(f"{final_output}")