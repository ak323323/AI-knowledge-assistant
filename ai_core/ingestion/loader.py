from typing import Optional
import os

import fitz
import markdown
import pandas as pd

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

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
    
# =========================================================
# LOAD EXCEL FILE
# =========================================================

def load_excel(path: str) -> str:
    """
    Load Excel workbook and convert sheets
    into semantic text for RAG ingestion.

    Strategy:
    ----------
    - Read every sheet
    - Convert rows into natural language
    - Preserve sheet structure
    - Keep column relationships
    """

    print(f"[EXCEL] Loading workbook: {path}")

    workbook_text = []

    # Read all sheets
    excel_data = pd.read_excel(
        path,
        sheet_name=None
    )

    # =====================================================
    # PROCESS EACH SHEET
    # =====================================================

    for sheet_name, df in excel_data.items():

        print(f"[EXCEL] Processing sheet: {sheet_name}")

        workbook_text.append(
            f"\n\n=== SHEET: {sheet_name} ===\n"
        )

        # Remove empty rows
        df = df.dropna(how="all")

        # Fill NaN values
        df = df.fillna("")

        # =================================================
        # CONVERT ROWS TO SEMANTIC TEXT
        # =================================================

        for _, row in df.iterrows():

            # Skip fully empty rows
            if row.isnull().all():
                continue

            row_parts = []

            for column, value in row.items():

                value = str(value).strip()

                if not value:
                    continue

                row_parts.append(
                    f"{column}: {value}"
                )

            # Skip empty rows
            if not row_parts:
                continue

            row_text = (
                        f"Sheet: {sheet_name}\n"
                        + "\n".join(row_parts)
                    )

            workbook_text.append(row_text)
            workbook_text.append("\n")
    
    final_text = "\n".join(workbook_text)

    print("\n[EXCEL DEBUG]")
    print(final_text[:1000])

    return final_text



# =========================================================
# LOAD CSV FILE
# =========================================================

def load_csv(path: str) -> str:
    """
    Load CSV file and convert rows into
    semantic RAG-friendly text.

    Why semantic conversion matters:
    -------------------------------

    Semantic formatting dramatically improves:
    - embeddings
    - retrieval
    - reranking
    """

    print(f"[CSV] Loading file: {path}")

    try:

        # =================================================
        # READ CSV
        # =================================================

        df = pd.read_csv(path)

        # Remove empty rows
        df = df.dropna(how="all")

        # Fill missing values
        df = df.fillna("")

        csv_text = []

        # =================================================
        # PROCESS ROWS
        # =================================================

        for _, row in df.iterrows():

            # Skip fully empty rows
            if row.isnull().all():
                continue

            row_parts = []

            # ---------------------------------------------
            # Convert columns into semantic text
            # ---------------------------------------------

            for column, value in row.items():

                value = str(value).strip()

                if not value:
                    continue

                row_parts.append(
                    f"{column}: {value}"
                )

            # Skip empty rows
            if not row_parts:
                continue

            # ---------------------------------------------
            # Better semantic structure
            # ---------------------------------------------

            row_text = "\n".join(row_parts)

            csv_text.append(row_text)

            # Preserve row boundaries
            csv_text.append("\n")

        # =================================================
        # FINAL TEXT
        # =================================================

        final_text = "\n".join(csv_text)

        print("\n[CSV DEBUG]")
        print(final_text[:1000])

        return final_text

    except Exception as e:

        print(f"[CSV ERROR] {e}")

        return ""


# =========================================================
# UNIVERSAL FILE LOADER
# =========================================================

def load_file(path: str):

    """
    Main ingestion dispatcher.

    Routes files to appropriate loaders
    based on extension.
    """

    ext = os.path.splitext(path)[1].lower()

    print(f"[LOADER] File extension: {ext}")

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if ext == ".pdf":
        return extract_text(path)

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    elif ext == ".docx":
        return load_docx(path)

    # -----------------------------------------------------
    # MARKDOWN
    # -----------------------------------------------------

    elif ext == ".md":
        return load_md(path)

    # -----------------------------------------------------
    # EXCEL
    # -----------------------------------------------------

    elif ext in [".xlsx", ".xls"]:
        return load_excel(path)
    
    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    elif ext == ".csv":
        return load_csv(path)

    # -----------------------------------------------------
    # UNSUPPORTED
    # -----------------------------------------------------

    raise ValueError(
        f"Unsupported file type: {ext}"
    )
    
