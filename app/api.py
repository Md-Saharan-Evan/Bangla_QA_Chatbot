from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Query):
    answer = rag.generate_answer(q.question)
    return {"answer": answer}

@app.get("/debug")
def debug(query: str):
    return {"chunks": rag.retrieve_chunks(query)}






