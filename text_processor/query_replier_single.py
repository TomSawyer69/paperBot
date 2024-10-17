import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import os

def read_chunks_from_file(filename):
    # Adjust the path to where your extracted text chunks are stored
    file_path = os.path.join('..', 'resources', 'extracted', filename, 'text', f'{filename}_text.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.split('<|endofchunk|>')


def get_chunk_range(chunks, x, range_size=2):
    start = max(0, x - range_size)
    end = min(len(chunks), x + range_size + 1)
    return chunks[start:end]


def process_chunks_and_query(chunks, chunk_number, query):
    # Get the range of chunks
    relevant_chunks = get_chunk_range(chunks, chunk_number)

    # Combine chunks into a paragraph
    paragraph = " ".join(relevant_chunks)

    # Initialize the language model
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Prepare input for the model
    input_text = f"Paragraph: {paragraph}\n\nQuery: {query}\n\nDetailed response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the generated part
    response = response.split("Detailed response:")[-1].strip()

    # Post-process to ensure complete sentences
    sentences = re.split('(?<=[.!?]) +', response)
    if len(sentences) > 1:
        response = ' '.join(sentences[:-1])  # Remove the last incomplete sentence if any

    return paragraph, response


with open('../resources/out/query.txt', 'r') as file:
    lines = file.readlines()

# Extract the query
query = lines[0].strip().split(": ", 1)[1]
query = query + ". Give a properly structured output."  # add custom prompt here

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
            chunk_idx = int(parts[1].split(': ')[1])


chunks = read_chunks_from_file(pdf_name.strip())


paragraph, llm_response = process_chunks_and_query(chunks, chunk_idx, query.strip())

reply_file = '..\\resources\\out\\reply.txt'

with open(reply_file, 'w', encoding='utf-8') as f:
    f.write(f"{llm_response}")