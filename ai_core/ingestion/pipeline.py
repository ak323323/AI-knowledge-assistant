from ingestion.loader import load_file
from processing.chunker import chunk_text
from storage.document_registry import DocumentRegistry
from typing import Dict, Any

import hashlib
import numpy as np
import re
import os
from datetime import datetime



def get_file_hash(path: str) -> str:
    """
    Generate stable document hash.

    Used as unique document ID.
    """

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while chunk := f.read(8192):
            md5.update(chunk)

    return md5.hexdigest()

def build_document_metadata(path: str, doc_id: str) -> Dict[str, Any]:
    """
    Build document-level metadata.
    """

    file_name = os.path.basename(path)

    return {

        "doc_id": doc_id,

        "file_name": file_name,

        "document_title": (
            file_name
            .replace(".pdf", "")
            .replace(".docx", "")
            .replace("_", " ")
        ),

        "source_path": path,

        "uploaded_at": datetime.now().isoformat()
    }
    


def ingest_file(path, embed_model, vectorstore):
    try:
        
        doc_id = get_file_hash(path)
        registry = DocumentRegistry()

        document_metadata = build_document_metadata(path, doc_id)

        file_name = document_metadata["file_name"]

        #  Prevent duplicate ingestion
        if registry.document_exists(doc_id):
            print("[SKIP] Document already indexed")
            return {
                "chunks": 0,
                "vectors": 0,
                "message": "Document already exists"
            }
        

        print(f"[INGEST] File: {path}")


        # 1. Load
        text = load_file(path)

        if not text:
            print("[ERROR] No text loaded")
            return {"chunks": 0, "vectors": 0}
        #Debug Metadata Persistence
        print("\n[LOAD DEBUG]")
        print(text[:500])

        # 2. Chunk
        chunks = chunk_text(text=text, file_name=file_name, doc_id=doc_id)

        # =========================================================
        # CLEAN & VALIDATE STRUCTURED CHUNKS
        # =========================================================

        cleaned_chunks = []

        for chunk in chunks:

            # -----------------------------------------------------
            # Extract actual text from ChunkData
            # -----------------------------------------------------

            cleaned_text = clean_chunk(
                chunk["text"]
            )

            # Skip weak chunks
            if len(cleaned_text.strip()) < 50:
                continue

            # -----------------------------------------------------
            # Preserve metadata structure
            # -----------------------------------------------------

            cleaned_chunk = {

                # Main text
                "text": cleaned_text,

                # Document metadata
                "file_name": chunk["file_name"],
                "document_title": chunk["document_title"],
                "doc_id": chunk["doc_id"],

                # Structural metadata
                "section": chunk["section"],
                "subsection": chunk["subsection"],

                # Chunk metadata
                "chunk_id": chunk["chunk_id"],
                "chunk_length": len(cleaned_text),
                "word_count": len(cleaned_text.split()),

                # Semantic metadata
                "keywords": chunk["keywords"],
                "tags": chunk["tags"]
            }
            cleaned_chunks.append(cleaned_chunk)

        # Replace original chunks
        chunks = cleaned_chunks

        print("\n[CHUNK DEBUG]")

        for i, chunk in enumerate(chunks[:5]):

            print(f"\nChunk {i}")

            print("-" * 60)

            print("Section:")
            print(chunk["section"])

            print("\nPreview:")
            print(chunk["text"][:200])

        # =========================================================
        # REMOVE DUPLICATE CHUNKS
        # =========================================================

        unique_chunks = []

        seen_hashes = set()

        for chunk in chunks:

            chunk_hash = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()

            if chunk_hash in seen_hashes:
                continue

            seen_hashes.add(chunk_hash)

            unique_chunks.append(chunk)

        chunks = unique_chunks

        if not chunks:
            print("[ERROR] No valid chunks created")
            return {"chunks": 0, "vectors": 0}

        print(f"[INGEST] Chunks: {len(chunks)}")

        # 3. Embed
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Generate embeddings
        vectors = embed_model.encode(chunk_texts)

        vectors = np.array(vectors, dtype=np.float32)

        # Safety checks
        if np.isnan(vectors).any():
            raise ValueError("NaN embeddings detected")

        if np.isinf(vectors).any():
            raise ValueError("Infinite embeddings detected")

        # Avoid division by zero
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1

        vectors = vectors / norms

        print("\n[EMBED DEBUG]")
        print("Vector shape:", vectors.shape)
        print("Embedding Dimension:", vectors.shape[1])
        print("Sample Vector:", vectors[0][:5])
        print("Total Embeddings:", len(vectors))

        if vectors is None:
            print("[ERROR] Embedding failed")
            print(f"[INGEST] Chunks created: {len(chunks)}")
            print(f"[INGEST] Sample chunk length: {len(chunks[0])}")

        # 4. Store
        vectorstore.add(vectors=vectors, texts=chunk_texts,metadata=chunks, source=path, doc_id=doc_id)
        document_metadata["chunk_count"] = len(chunks)

        document_metadata["embedding_model"] = (
            "BAAI/bge-base-en-v1.5"
        )

        registry.add_document(
            doc_id,
            document_metadata
        )

        print(f"[INGEST] Vectors added: {len(vectors)}")
        print(f"[INGEST] Total index: {vectorstore.index.ntotal}")

        print("\n" + "=" * 80)
        print("[INGESTION COMPLETE]")
        print("=" * 80)

        print(f"Document      : {file_name}")
        print(f"Doc ID        : {doc_id}")
        print(f"Chunks Stored : {len(chunks)}")
        print(f"Vector Count  : {len(vectors)}")
        print(f"FAISS Total   : {vectorstore.index.ntotal}")

        return {
            "chunks": len(chunks),
            "vectors": len(vectors),
            "total_index": vectorstore.index.ntotal
        }

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"chunks": 0, "vectors": 0, "error": str(e)}
    
# =========================================================
# CLEAN INDIVIDUAL CHUNK
# =========================================================

def clean_chunk(text: str) -> str:
    """
    Clean semantic chunks before embedding.
    """

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove weird unicode artifacts
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)

    # Normalize spacing
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    return text.strip()