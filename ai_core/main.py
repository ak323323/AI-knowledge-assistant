from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import numpy as np
import faiss
import logging
import time
from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.cors import CORSMiddleware
from embeddings.model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from retreival.retriever import Retriever
from llm.ollama_client import OllamaClient
from rag.pipeline import RAGPipeline
from ingestion.pipeline import ingest_file
from uuid import uuid4
from fastapi import FastAPI, Response
from services.export_service import generate_pdf, generate_docx, generate_excel, generate_markdown, generate_csv
from sentence_transformers import CrossEncoder
from typing import Optional, Dict
from pydantic import BaseModel

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)


logging.info("STARTING API...")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Initialize components ONCE
embed_model = None

def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = EmbeddingModel("BAAI/bge-base-en-v1.5")
    return embed_model


print("LOADING VECTORSTORE...")
vectorstore = FAISSStore(dim=768, embed_model=get_embed_model())
print("VECTORSTORE LOADED")
rag = None


def get_rag():
    global rag

    if rag is None:
        model = get_embed_model()
        retriever = Retriever(model, vectorstore)
        llm = OllamaClient("llama3")
        rag = RAGPipeline(retriever, llm)

    return rag

reranker_model = None

def get_reranker():
    global reranker_model

    if reranker_model is None:
        print("[MODEL] Loading reranker...")

        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return reranker_model


# Request schema
class Query(BaseModel):
    """
    Incoming user query payload.
    """

    retrieval_mode: str = "semantic"
    chat_history: Optional[list] = []

    # User question
    question: str

    # Optional selected file
    selected_file: Optional[str] = None

    # Optional metadata filters
    filters: Optional[Dict[str, str]] = None

    # Retrieval config
    top_k: int = 8


@app.on_event("startup")
def startup_check():

    logging.info(
        "Running startup checks..."
    )

    try:

        _ = get_embed_model()

        logging.info(
            "Embedding model loaded"
        )

    except Exception as e:

        logging.error(str(e))

@app.get("/stats")
def stats():

    unique_docs = set()

    for t in vectorstore.texts:

        unique_docs.add(
            t.get("doc_id")
        )

    return {

        "documents": len(unique_docs),

        "chunks": len(vectorstore.texts),

        "vectors": vectorstore.index.ntotal,

        "embedding_model":
            "BAAI/bge-base-en-v1.5",

        "llm": "llama3"
    }

# Health Check Endpoint
@app.get("/")
def root():
    return {

    "status": "running",

    "supported_formats": [

        "pdf",
        "docx",
        "md",
        "xlsx",
        "xls",
        "csv"
    ],

    "vector_count": vectorstore.index.ntotal
}

# ---------------------------------------------------
# ASK ENDPOINT
# ---------------------------------------------------
@app.post("/ask")
def ask(query: Query):
    """
    Main RAG endpoint.

    Flow:
    1. Validate request
    2. Build metadata filters
    3. Run retrieval
    4. Generate answer
    5. Return grounded response
    """

    try:

        print("\n" + "=" * 60)
        print("[ASK API] Incoming Question")
        print("=" * 60)

        print("Question:", query.question)

        # ---------------------------------------------------
        # SAFETY: Empty question
        # ---------------------------------------------------
        if not query.question.strip():

            return {
                "answer": "Question cannot be empty.",
                "sources": []
            }

        # ---------------------------------------------------
        # SAFETY: Empty vector DB
        # ---------------------------------------------------
        if vectorstore.index.ntotal == 0:

            print("[ASK API] No indexed documents")

            return {
                "answer": (
                    "No documents indexed yet. "
                    "Please upload a document first."
                ),
                "sources": []
            }

        # ---------------------------------------------------
        # BUILD METADATA FILTERS
        # ---------------------------------------------------
        filters = query.filters or {}

        # File selection filter
        if query.selected_file:

            filters["file_name"] = query.selected_file

        print("\n[FILTER DEBUG]")
        print(filters)

        # ---------------------------------------------------
        # GET RAG PIPELINE
        # ---------------------------------------------------
        rag_pipeline = get_rag()

        # ---------------------------------------------------
        # RUN RAG PIPELINE
        # ---------------------------------------------------
        start = time.time()
        result = rag_pipeline.run(
            question=query.question,
            filters=filters,
            top_k=query.top_k
        )
        end = time.time()
        print(f"[TIMING] Total RAG time: {end - start:.2f}s")

        # ---------------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------------
        print("\n[RAG RESULT DEBUG]")

        print(
            f"Sources Returned: "
            f"{len(result.get('sources', []))}"
        )

        for i, source in enumerate(
            result.get("sources", [])[:5]
        ):

            print(
                f"""
                SOURCE {i+1}
                File      : {source.get('file_name')}
                Score     : {source.get('score')}
                Category  : {source.get('category')}
                Preview   : {source.get('content', '')[:120]}
                """
            )

        print("\n[ASK API] Completed Successfully")

        return result

    # ---------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------
    except Exception as e:

        print("\n[ASK API ERROR]")
        print(str(e))

        return {
            "answer": (
                "An internal error occurred while "
                "processing the query."
            ),
            "sources": [],
            "error": str(e)
        }


# Upload + ingest
@app.post("/upload")

async def upload(file: UploadFile = File(...)):
    try:
        
        if not file.filename:

            return {
                "error": "Invalid file"
            }

        # Guaranteed filename string
        filename = str(file.filename)

        ext = os.path.splitext(filename)[1].lower()
        ALLOWED_EXTENSIONS = {
            ".pdf",
            ".docx",
            ".md",
            ".xlsx",
            ".xls",
            ".csv"
            }

        if ext not in ALLOWED_EXTENSIONS:

            return {
                "error": (
                    f"Unsupported file type: {ext}"
                )
            }

        path = os.path.join(DATA_DIR, filename)

        with open(path, "wb") as f:
            f.write(await file.read())

        model = get_embed_model()
        if model is None:
            return{"return: Embedding model failed to load"}
            

        result = ingest_file(path, model, vectorstore)

        return {

        "file_name": file.filename,

        "doc_id": result.get("doc_id"),

        "chunks_created": result.get("chunks", 0),

        "vectors_added": result.get("vectors", 0),

        "total_vectors":
            result.get(
                "total_index",
                vectorstore.index.ntotal
            ),

        "sections":
            result.get("sections", []),

        "keywords":
            result.get("keywords", [])[:10]
    }

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {
            "message": "Upload failed",
            "error": str(e)
        }
    
# Delete Endpoint
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):

    try:

        print(f"\n[API DELETE] Requested doc_id: {doc_id}")

        # =====================================================
        # FIND DOCUMENT
        # =====================================================

        document = None

        for chunk in vectorstore.texts:

            if chunk.get("doc_id") == doc_id:

                document = chunk

                break

        if not document:

            return {
                "success": False,
                "message": "Document not found"
            }

        # =====================================================
        # GET FILE PATH
        # =====================================================

        source_path = document.get("source", "")

        print(f"[DELETE] Source Path: {source_path}")

        # =====================================================
        # DELETE FROM VECTORSTORE
        # =====================================================

        success = vectorstore.delete_document(doc_id)

        if not success:

            return {
                "success": False,
                "message": "Vector deletion failed"
            }

        # =====================================================
        # DELETE PHYSICAL FILE
        # =====================================================

        if source_path:

            try:

                if os.path.exists(source_path):

                    os.remove(source_path)

                    print(f"[DELETE] File removed")

                else:

                    print("[DELETE] File already missing")

            except Exception as file_error:

                print(
                    "[DELETE FILE ERROR]",
                    str(file_error)
                )

        # =====================================================
        # SUCCESS
        # =====================================================

        return {
            "success": True,
            "message": "Document deleted successfully"
        }

    except Exception as e:

        print("\n[DELETE API ERROR]")
        print(str(e))

        return {
            "success": False,
            "message": str(e)
        }

# Export Endpoint
@app.post("/export")
def export_response(data: dict):
    """
    data = {
        "answer": "...",
        "sources": [...],
        "format": "pdf" or "docx"
    }
    """

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    fmt = data.get("format", "pdf")

    if fmt == "pdf":
        file = generate_pdf(answer, sources)
        return Response(
            content=file.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=answer.pdf"}
        )

    elif fmt == "docx":

        file = generate_docx(answer, sources)

        return Response(
            content=file.read(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition":
                "attachment; filename=answer.docx"
            }
        )

    # =====================================================
    # CSV EXPORT
    # =====================================================

    elif fmt == "csv":

        file = generate_csv(answer, sources)

        return Response(
            content=file.read(),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=answer.csv"
            }
        )

    elif fmt == "md":
        file = generate_markdown(
            answer,
            sources
        )

        return Response(
            content=file.read(),
            media_type="text/markdown",
            headers={
                "Content-Disposition":
                "attachment; filename=answer.md"
            }
        )

    elif fmt == "xlsx":

        file = generate_excel(
            answer,
            sources
        )

        return Response(
            content=file.read(),
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition":
                "attachment; filename=answer.xlsx"
            }
        )

    return {"error": "Invalid format"}

# Indexed Files Endpoint
@app.get("/documents")
def get_documents():

    docs = {}

    for t in vectorstore.texts:
        doc_id = t.get("doc_id")
        source = t.get("source")

        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": source
            }

    return list(docs.values())


@app.get("/documents/{doc_id}")
def get_document_details(doc_id: str):

    chunks = [
        t for t in vectorstore.texts
        if t.get("doc_id") == doc_id
    ]

    if not chunks:
        return {
            "error": "Document not found"
        }

    first = chunks[0]

    return {

        "doc_id": doc_id,

        "file_name": first.get("file_name"),

        "document_title": first.get(
            "document_title"
        ),

        "total_chunks": len(chunks),

        "sections": list(set(
            c.get("section", "General")
            for c in chunks
        )),

        "source": first.get("source"),

        "sample_keywords": first.get(
            "keywords",
            []
        )[:10]
    }