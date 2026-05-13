from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import numpy as np
import faiss
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
from services.export_service import generate_pdf, generate_docx
from sentence_transformers import CrossEncoder
from typing import Optional, Dict
from pydantic import BaseModel

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


print("STARTING API...")
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

    # User question
    question: str

    # Optional selected file
    selected_file: Optional[str] = None

    # Optional metadata filters
    filters: Optional[Dict[str, str]] = None

    # Retrieval config
    top_k: int = 5


# Health Check Endpoint
@app.get("/")
def root():
    return {"status": "running"}

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
        result = rag_pipeline.run(
            question=query.question,
            filters=filters,
            top_k=query.top_k
        )

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
            return {"error": "Invalid file"}

        path = os.path.join(DATA_DIR, file.filename)

        with open(path, "wb") as f:
            f.write(await file.read())

        model = get_embed_model()
        if model is None:
            return{"return: Embedding model failed to load"}
            

        result = ingest_file(path, model, vectorstore)

        return {
            "file_name": file.filename,
            "chunks_created": result.get("chunks", 0),
            "vectors_added": result.get("vectors", 0),
            "total_vectors": result.get("total_index", vectorstore.index.ntotal)
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

    print(f"[API DELETE] Requested doc_id: {doc_id}")

    success = vectorstore.delete_document(doc_id)

    if not success:
        return {
            "success": False,
            "message": "Document not found"
        }
    
    print("\n[DOCUMENT DEBUG]")
    print(f"Total Chunks: {len(vectorstore.texts)}")

    if vectorstore.texts:
        print("Sample Metadata:")
        print(vectorstore.texts[0])
    else:
        print("Vectorstore is empty")

    return {
        "success": True,
        "message": "Document deleted successfully"
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
            headers={"Content-Disposition": "attachment; filename=answer.docx"}
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