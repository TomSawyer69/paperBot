# paperBot

This project implements a Question-Answering (QA) bot that leverages a knowledge base of research papers. The bot is designed to extract meaningful information from PDF documents, generate embeddings for text chunks, and allow users to query these embeddings to retrieve relevant answers along with their sources.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

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

```bash
git clone https://github.com/yourusername/research-paper-qa-bot.git
cd research-paper-qa-bot
