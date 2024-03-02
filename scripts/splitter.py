import re
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfFileReader
from docx import Document

def split_into_sentences_from_file(file_path):
    if file_path.endswith(".html"):
        # If file is HTML
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
    elif file_path.endswith(".pdf"):
        # If file is PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
    elif file_path.endswith(".docx"):
        # If file is DOCX
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        text = '\n'.join(paragraphs)
    elif file_path.endswith(".txt"):
        # If file is plain text
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_path.startswith("http"):
        # If file path is a URL
        response = requests.get(file_path)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
        elif 'pdf' in content_type:
            pdf_reader = PdfFileReader(response.content)
            text = ""
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
        else:
            raise ValueError("Unsupported content type from URL")
    else:
        raise ValueError("Unsupported file type")

    # Split text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)

    return sentences
