import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load filtered dataset
df = pd.read_csv('data/filtered_complaints.csv')

# Text chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
df['chunks'] = df['cleaned_narrative'].apply(lambda x: text_splitter.split_text(x))

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.Client(Settings(persist_directory="vector_store"))
collection = client.create_collection("complaints")

# Embed and index
def add_to_vector_store(row):
    complaint_id = row['Complaint ID']
    product = row['Product']
    chunks = row['chunks']
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"{complaint_id}_{i}"],
            embeddings=[embedding],
            metadatas=[{"complaint_id": complaint_id, "product": product, "chunk_index": i}]
        )

df.apply(add_to_vector_store, axis=1)
client.persist()
print("Vector store created and persisted in 'vector_store/'")
