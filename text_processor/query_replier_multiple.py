import numpy as np
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import re
import os


# Initialize the GPT-Neo model and tokenizer outside the loop
model_name = "EleutherAI/gpt-neo-2.7B"  # This is a much larger, more capable model
cache_dir = 'E:/huggingface_cache'


# Load the model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir=cache_dir)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


def generate_output(filename, chunk_number, query):
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

        # Prepare input for the model
        input_text = f"Paragraph: {paragraph}\n\nQuery: {query}\n\nDetailed response:"
        inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)

        # Generate response using GPT-Neo
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

        return response

    chunks = read_chunks_from_file(filename)
    llm_response = process_chunks_and_query(chunks, chunk_number, query)

    return filename, llm_response


# Main loop to read and process lines from the avg_distances.txt file
with open('../resources/avg_distances.txt', 'r') as file:
    lines = file.readlines()

# Extract the query
query = lines[0].strip().split(": ", 1)[1]
query = query + ". Give a properly structured output."  # add custom prompt here

output = []

# Open a text file for writing the final output
with open('output.txt', 'w', encoding='utf-8') as outfile:
    outfile.write("Generated Responses:\n")

    for line in lines[1:]:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Only process non-empty lines
            pdf_info = line.split(", ")
            pdf_name = pdf_info[0].split(": ")[1]  # Extract PDF name
            chunk_number = int(pdf_info[1].split(": ")[1])  # Extract chunk number

            print(f"Processing PDF: {pdf_name}, Chunk number: {chunk_number}")  # Debugging: Show current processing

            result = generate_output(pdf_name, chunk_number, query)
            output.append(result)

            # Write the output to the text file
            outfile.write(f"PDF: {result[0]}, Response: {result[1]}\n\n")

# Optionally print the final output list
print(output)
