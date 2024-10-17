# paperBot

This project implements a Question-Answering (QA) bot that leverages a knowledge base of research papers. The bot is designed to extract meaningful information from hundreds of PDF documents, generate embeddings for text chunks, and allow users to query these embeddings to retrieve relevant answers along with their sources.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

## Features

- Extracts text, tables, and images from PDF research papers.
- Generates embeddings for text chunks using Sentence Transformers.
- Stores embeddings in a vector database using FAISS.
- Allows users to query the database and retrieve relevant PDFs and chunks.
- Outputs query results in both plain text and XML formats.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### Clone the Repository

1. Clone the repository:
   
```bash
   git clone https://github.com/yourusername/research-paper-qa-bot.git
   cd research-paper-qa-bot
```

2. Create a virtual environment:
   
```bash
   python -m venv venv
```

3. Activate the virtual environment:

   - On Windows:
     
```bash
     venv\Scripts\activate
```
   - On macOS/Linux:

```bash
     source venv/bin/activate
```

### Install Required Packages

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
## Usage

1. To be added

## Directory Structure

/research-paper-qa-bot
├── /app                            # Directory for UI Elements
│   └── /interface.py               
├── /resources                      # Project Resources Directory (git-ignored)
│   ├── /embeddings                 # Output directory for embeddings and chunks data
│   ├── /extracted                  # Output directory for extracted content
│   └── /uploads                    # Output directory for pdf files extracted
├── /text_processor                 # Directory for text processing
│   ├── /chunk_extractor.py         
│   ├── /distance_querying.py       
│   └── /embeddings_generator.py    
├── venv                            # Virtual Environment of the project (git-ignored)
├── .gitignore                      
├── README.md                       
└── requirements.txt                

