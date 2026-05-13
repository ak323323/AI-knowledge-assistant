from pypdf import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup
import fitz
from typing import Optional
from utils.text_cleaner import clean_text

def extract_text(path: str)-> Optional[str]:
    try:
        doc = fitz.open(path)
        text_parts : list[str]=[]

        for i in range(len(doc)):
            page = doc[i]
            try:
                page_text = page.get_text("text")

                if isinstance(page_text, str):
                    cleaned = clean_text(page_text)
                    if cleaned:
                            text_parts.append(cleaned)
                    else:
                            print(f"[WARN] Page {i} empty")
                else:
                        print(f"[WARN] Unexpected type on page {i}: {type(page_text)}") 

            except Exception as e:
                print(f"[ERROR] Failed reading page {i}: {e}")

        full_text = "\n".join(text_parts)

        if not full_text.strip():
            print("[ERROR] No text extracted from PDF")
            return None

        return full_text

    except Exception as e:
        print("[ERROR] Could not open PDF:", str(e))
        return None


def load_docx(path):
    doc = Document(path)
    raw = "\n".join([p.text for p in doc.paragraphs])
    return clean_text(raw)


def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        return clean_text(soup.get_text())


def load_file(path):
    if path.endswith(".pdf"):
        return extract_text(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    elif path.endswith(".md"):
        return load_md(path)
    else:
        raise ValueError("Unsupported file type")