import re

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Keep Bangla, English, numbers and punctuation
                page_text = re.sub(r'[^\u0980-\u09FF0-9A-Za-z\s,.?!:\-\n()]', '', page_text)
                page_text = re.sub(r'\n+', '\n', page_text)
                text += page_text + "\n"
    return text

def remove_mcq_lines(text):
    lines = text.splitlines()
    clean_lines = [line for line in lines if not re.match(r'^\(?[\u0995-\u0998]\)?', line.strip())]
    return "\n".join(clean_lines)

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
    
