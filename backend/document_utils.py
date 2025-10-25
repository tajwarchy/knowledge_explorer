import docx2txt, html2text
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_path: str) -> str:
    return docx2txt.process(file_path)


def extract_text_from_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return html2text.html2text(f.read())


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
