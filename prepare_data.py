# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain.vectorstores import Chroma

# from app.config import CHROMA_DIR, EMBED_MODEL, PDF_PATH
# from app.utils import chunk_text, extract_text_from_pdf

# # Extract and chunk text
# text = extract_text_from_pdf(PDF_PATH)
# chunks = chunk_text(text)
# docs = [Document(page_content=chunk) for chunk in chunks]

# # Embed & store
# embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# chroma = Chroma.from_documents(
#     documents=docs,
#     embedding=embedding,
#     persist_directory=CHROMA_DIR
# )
# chroma.persist()

# print("✅ ChromaDB index built and saved.")


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from app.config import CHROMA_DIR, EMBED_MODEL, PDF_PATH
from app.utils import chunk_text, extract_text_from_pdf, remove_mcq_lines

text = extract_text_from_pdf(PDF_PATH)
text = remove_mcq_lines(text)
chunks = chunk_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
chroma = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=CHROMA_DIR
)
chroma.persist()
print("\u2705 ChromaDB index built and saved.")