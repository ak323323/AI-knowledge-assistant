from ingestion.loader import load_file
from processing.chunker import chunk_text
import hashlib
import numpy as np
import re



def get_file_hash(path: str) -> str:
    """Generate unique hash for a file (used as doc_id)"""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    


def ingest_file(path, embed_model, vectorstore):
    try:
        
        doc_id = get_file_hash(path)

        #  Prevent duplicate ingestion
        if vectorstore.document_exists(doc_id):
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
        print(text[0])

        # 2. Chunk
        chunks = chunk_text(text)

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

                "text": cleaned_text,

                "section": chunk["section"],

                "chunk_length": len(cleaned_text)
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

        seen_texts = set()

        for chunk in chunks:

            text = chunk["text"]

            if text in seen_texts:
                continue

            seen_texts.add(text)

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

        # Avoid division by zero
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1

        vectors = vectors / norms

        print("\n[EMBED DEBUG]")
        print("Vector shape:", vectors.shape)
        print("Sample vector:", vectors[0][:5])

        if vectors is None:
            print("[ERROR] Embedding failed")
            print(f"[INGEST] Chunks created: {len(chunks)}")
            print(f"[INGEST] Sample chunk length: {len(chunks[0])}")

        # 4. Store
        vectorstore.add(vectors=vectors, texts=chunk_texts,metadata=chunks, source=path, doc_id=doc_id)

        print(f"[INGEST] Vectors added: {len(vectors)}")
        print(f"[INGEST] Total index: {vectorstore.index.ntotal}")

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
    Cleans individual semantic chunks.

    Fixes:
    -------
    - excessive whitespace
    - broken line spacing
    - repeated spaces
    """

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()