# paperBot

This project implements a Question-Answering (QA) bot that leverages a knowledge base of research papers. The bot is designed to extract meaningful information from hundreds of PDF documents, generate embeddings for text chunks, and allow users to query these embeddings to retrieve relevant answers along with their sources.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Future Improvements](#improvements)
- [Pending Modules](#pending-modules)

## Features

- Extracts text, tables, and images from PDF research papers.
- Generates embeddings for text chunks using Sentence Transformers.
- Stores embeddings in a vector database using FAISS.
- Allows users to query the database and retrieve relevant PDFs and chunks.
- Outputs query results plain text format


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

1. Open the upload app interface to upload the zip file and enter the query.
```bash
     streamlit run /paperBot/app/app.py
```
   After this, close the window and continue.
   Note: The uploaded files are cleared everytime the page is reloaded or when switching from the query page.
   

2. Run the python files in cmd in this order:
```bash
    '/paperBot/text_processor/chunk_extractor.py'
    '/paperBot/text_processor/embeddings_generator.py'
    '/paperBot/text_processor/distance_querying.py'
    '/paperBot/text_processor/query_replier_single.py'
```

3. After running these, the output.txt file is stored in `/paperBot/resources/output.txt`

## Directory Structure

```bash
/research-paper-qa-bot
├── /app                            # Directory for UI Elements
│   └── /interface.py               
├── /resources                      # Project Resources Directory (git-ignored)
├── /text_processor                 # Directory for text processing
│   ├── /chunk_extractor.py         
│   ├── /distance_querying.py
│   ├── /embeddings_generator.py
│   ├── /query_replier_multiple.py  
│   └── /query_replier_single.py    
├── venv                            # Virtual Environment of the project (git-ignored)
├── .gitignore                      
├── README.md                       
└── requirements.txt                
```

## Future Improvements 

- Retrieval of text chunks based on layouts or similar methods for proper chunking.
- Usage of better model to answer the query after finding the matching papers.
- Implementation of images and table embeddings.


## Pending modules

- Seamless implementation of the project and integration of all the various modules to work together.
