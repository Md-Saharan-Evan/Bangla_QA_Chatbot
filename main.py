import os
import re
import unicodedata

import pdfplumber
import torch
from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Disable Chroma telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Force CPU usage for PyTorch
torch_device = torch.device("cpu")

class SentenceTransformerEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device=str(torch_device))
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).astype(float).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).astype(float)[0].tolist()

embedding_model = SentenceTransformerEmbeddings('paraphrase-multilingual-MiniLM-L12-v2')
generator = pipeline('text2text-generation', model='google/flan-t5-base', device=-1)

chat_history = []

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print("PDF not found. Using sample text.")
        return "চতুর্থপাশ হল অমুক্তসব তামাস পুস্তক। মামাকে অমুক্তসব তামা বলা হয়েছে। বিস্নেব সময় ১৫ বছর।"
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[^\w\s।?!]', '', text)
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["।", "!", "?", "\n"]
    )
    return text_splitter.split_text(text)

def create_vector_store(chunks, persist_directory="./chroma_db"):
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vector_store

def load_vector_store(persist_directory="./chroma_db"):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def retrieve_relevant_chunks(query, vector_store, top_k=3):
    index_size = len(vector_store._collection.get()["ids"])
    top_k = min(top_k, index_size)
    results = vector_store.similarity_search(query, k=top_k)
    return [result.page_content for result in results]

def generate_answer(query, chunks):
    context = "\n".join(chunks)
    print(f"DEBUG: Retrieved Chunks for Generation:\n{context}\n")  # Debug print to check chunks
    prompt = (
        f"Please answer the question based on the context below.\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    response = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False  # deterministic generation
    )[0]['generated_text']
    update_chat_history(query, response)
    return response

def update_chat_history(query, response):
    chat_history.append({"query": query, "response": response})
    if len(chat_history) > 5:
        chat_history.pop(0)

def evaluate_retrieval(query, chunks):
    try:
        query_embedding = embedding_model.embed_query(query)
        chunk_embeddings = embedding_model.embed_documents(chunks)
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        return similarities.mean()
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return 0.0

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Multilingual RAG API is working!"}

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        vector_store = load_vector_store()
        chunks = retrieve_relevant_chunks(request.query, vector_store)
        answer = generate_answer(request.query, chunks)
        similarity_score = evaluate_retrieval(request.query, chunks)
        return {
            "query": request.query,
            "answer": answer,
            "similarity_score": float(similarity_score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    pdf_path = "HSC26_Bangla_1st_paper.pdf"
    persist_directory = "./chroma_db"
    try:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print("Loading existing vector store...")
            vector_store = load_vector_store()
        else:
            print("Creating new vector store...")
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned_text = preprocess_text(raw_text)
            document_chunks = chunk_text(cleaned_text)
            vector_store = create_vector_store(document_chunks)

        sample_queries = [
            "অমুক্তসব তামাস পুস্তকন কাকে বলা হয়েছে?",
            "কাকে অমুক্তসব তামা (সেবার বলে উল্লেখ করা হয়েছে?",
            "বিস্নেব সময় কতটারি প্রকৃত বসস কত দিল?"
        ]
        for query in sample_queries:
            chunks = retrieve_relevant_chunks(query, vector_store)
            answer = generate_answer(query, chunks)
            similarity_score = evaluate_retrieval(query, chunks)
            print(f"Query: {query}\nAnswer: {answer}\nSimilarity Score: {similarity_score}\n")

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except Exception as e:
        print(f"Error: {str(e)}")
