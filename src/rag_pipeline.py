from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Load embedding model and ChromaDB
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.Client(Settings(persist_directory="vector_store"))
collection = client.get_collection("complaints")

# Retriever
def retrieve_relevant_chunks(question, k=5):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=k)
    chunks = [result['documents'][0] for result in results['results']]
    metadatas = [result['metadatas'][0] for result in results['results']]
    return chunks, metadatas

# Prompt template
prompt_template = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""

# Generator
llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", max_length=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def generate_answer(question, chunks):
    context = "\n".join(chunks)
    prompt = prompt_template.format(context=context, question=question)
    answer = llm(prompt)
    return answer

# RAG pipeline
def rag_pipeline(question):
    chunks, metadatas = retrieve_relevant_chunks(question)
    answer = generate_answer(question, chunks)
    return answer, chunks, metadatas

# Evaluation
questions = [
    "What are the common issues with credit card complaints?",
    "How do customers feel about personal loan customer service?",
    "Are there recurring problems with BNPL services?",
    "What are the main concerns regarding savings accounts?",
    "How efficient are money transfer services according to complaints?"
]

for q in questions:
    answer, chunks, _ = rag_pipeline(q)
    print(f"Question: {q}\nAnswer: {answer}\nSources: {chunks[:2]}\n")
