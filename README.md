# Complaint-Answering Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot to answer questions about consumer complaints from the CFPB dataset (`complaint.csv`). The chatbot uses natural language processing to retrieve relevant complaint excerpts and generate informative responses. It features a user-friendly Gradio interface, allowing users to input questions, view answers, and inspect source texts for transparency.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data](#data)
- [Tasks and Implementation](#tasks-and-implementation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Complaint-Answering Chatbot processes consumer complaints from the Consumer Financial Protection Bureau (CFPB) dataset, focusing on five product categories: Credit card, Personal loan, Buy Now, Pay Later (BNPL), Savings account, and Money transfers. The system uses a RAG pipeline to:
1. Retrieve relevant complaint text chunks using semantic search.
2. Generate answers with a language model.
3. Provide a web interface for user interaction, displaying both answers and source texts.

## Features
- **Exploratory Data Analysis (EDA)**: Analyzes complaint distribution and narrative lengths.
- **Text Preprocessing**: Filters and cleans complaint narratives for embedding.
- **Text Chunking and Embedding**: Splits narratives into manageable chunks and embeds them using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store**: Stores embeddings in ChromaDB with metadata for traceability.
- **RAG Pipeline**: Retrieves top-5 relevant chunks and generates answers using Mistral-7B.
- **Interactive Interface**: Gradio-based web app with question input, answer display, and source text visibility for trust.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/complaint-answering-chatbot.git
   cd complaint-answering-chatbot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the Dataset**:
   - Place the CFPB dataset (`complaint.csv`) in the `data/` directory.

2. **Run EDA and Preprocessing**:
   ```bash
   python src/eda_preprocessing.py
   ```
   This generates `data/filtered_complaints.csv` and visualizations in `notebooks/`.

3. **Chunk, Embed, and Index**:
   ```bash
   python src/chunk_embed_index.py
   ```
   This creates a vector store in `vector_store/`.

4. **Test the RAG Pipeline**:
   ```bash
   python src/rag_pipeline.py
   ```
   This runs sample questions and outputs answers with sources.

5. **Launch the Gradio Interface**:
   ```bash
   python src/app.py
   ```
   Open the provided URL (e.g., `http://127.0.0.1:7860`) in a browser to interact with the chatbot.

## Project Structure
```
complaint-answering-chatbot/
├── data/
│   ├── complaint.csv              # Input dataset
│   └── filtered_complaints.csv    # Filtered and cleaned dataset
├── notebooks/
│   ├── product_distribution.png   # Visualization of complaints by product
│   └── narrative_length_distribution.png  # Visualization of narrative lengths
├── src/
│   ├── eda_preprocessing.py       # EDA and preprocessing script
│   ├── chunk_embed_index.py       # Text chunking, embedding, and indexing
│   ├── rag_pipeline.py            # RAG pipeline logic and evaluation
│   └── app.py                    # Gradio web interface
├── vector_store/                 # ChromaDB vector store
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Dependencies
- Python 3.8+
- Key libraries (see `requirements.txt`):
  - `pandas`
  - `matplotlib`
  - `langchain`
  - `sentence-transformers`
  - `chromadb`
  - `transformers`
  - `gradio`

Install them with:
```bash
pip install pandas matplotlib langchain sentence-transformers chromadb transformers gradio
```

## Data
- **Input**: `complaint.csv` (CFPB dataset with columns like `Complaint ID`, `Product`, `Consumer complaint narrative`).
- **Processed**: `filtered_complaints.csv` contains cleaned narratives for the specified products.
- **Note**: Ensure `complaint.csv` is in the `data/` directory before running scripts.

## Tasks and Implementation
The project was completed in four tasks:
1. **EDA and Preprocessing**:
   - Analyzed complaint distribution and narrative lengths.
   - Filtered for five products and removed empty narratives.
   - Cleaned text by lowercasing and removing special characters.
2. **Text Chunking, Embedding, and Indexing**:
   - Used `RecursiveCharacterTextSplitter` with `chunk_size=512` and `chunk_overlap=50`.
   - Embedded chunks with `sentence-transformers/all-MiniLM-L6-v2`.
   - Indexed in ChromaDB with metadata (Complaint ID, Product).
3. **RAG Pipeline**:
   - Retrieves top-5 chunks using semantic search.
   - Generates answers with Mistral-7B and a custom prompt template.
   - Evaluated with sample questions (results in the report).
4. **Interactive Interface**:
   - Built with Gradio, featuring question input, answer display, and source texts.

For detailed findings, see the project report (`Task1_and_Task2_Report.md`).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`) and push (`git push origin feature-name`).
4. Open a pull request.

## License
This project is licensed under the MIT License.