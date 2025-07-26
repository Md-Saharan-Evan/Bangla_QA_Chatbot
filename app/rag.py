# app/rag.py — Updated to Use Ollama Mistral

import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from app.config import CHROMA_DIR, EMBED_MODEL


class RAGPipeline:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=CHROMA_DIR
        )

    def retrieve_chunks(self, query, top_k=3):
        results = self.vectordb.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

    def generate_answer(self, query):
        chunks = self.retrieve_chunks(query)
        if not chunks:
            return "কোন প্রাসংগিক তথ্য পাওয়া যায়নি।"

        context = "\n".join(chunks)
        prompt = f"""তুমি একজন সহায়ক বাংলা শিক্ষক। নিচের প্রসঙ্গ থেকে প্রশ্নের সংক্ষিপ্ত ও নির্ভুল উত্তর দাও।

প্রসঙ্গ:
{context}

প্রশ্ন: {query}
উত্তর:"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"উত্তর পাওয়া যায়নি। সার্ভার ত্রুটি: {response.status_code}"
        except Exception as e:
            return f"উত্তর তৈরি করতে সমস্যা হয়েছে: {str(e)}"


