import faiss
import numpy as np
import os
import pickle
import tempfile
from datetime import datetime


class FAISSStore:
    def __init__(self, dim, embed_model, storage_path="storage"):
        self.dim = dim
        self.embed_model = embed_model
        self.storage_path = storage_path
        self.metadata = []
        self.sources = []
        self.doc_ids = []

        os.makedirs(storage_path, exist_ok=True)

        self.index_file = os.path.join(storage_path, "faiss.index")
        self.meta_file = os.path.join(storage_path, "metadata.pkl") 

        # Load or create
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.load()
        else:
            self._create_new()

        print(f"[INIT] FAISS dim = {self.dim}")
        print(f"[INIT] Index size = {self.index.ntotal}")

    # -------------------------
    # CREATE NEW INDEX
    # -------------------------
    def _create_new(self):
        print("[INIT] Creating new FAISS index")
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts = []

    # -------------------------
    # ADD DATA
    # -------------------------
    def add(self, vectors, texts, metadata=None, source="unknown", doc_id=None):
        
        # Ensure 2D
        vectors = np.array(vectors).astype("float32")

        # Validate vectors
        if np.isnan(vectors).any() or np.isinf(vectors).any():
            raise ValueError("Invalid embeddings detected (NaN or Inf)")

        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {vectors.shape[1]}")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        base_index = self.index.ntotal

        for i, text in enumerate(texts):
            # Extracting metadata for this chunk
            meta = metadata[i] if metadata else {}
            
            # Store chunk + metadata
            self.texts.append({
                 # Main retrievable content
                "content": text,

                # File metadata
                "source": source,

                "file_name": os.path.basename(source),

                # Chunk tracking
                "chunk_id": base_index + i,

                "doc_id": doc_id,

                # Semantic metadata
                "section": meta.get("section", "Unknown"),

                "chunk_length": meta.get(
                    "chunk_length",
                    len(text)
                ),

                # Timestamp
                "uploaded_at": datetime.now().isoformat(),

                # Grouping/filtering
                "category": "general",

                # Optional future tagging
                "tags": []
            })
        # Debug metadata
        print("\n[METADATA DEBUG]")
        print(self.texts[-1])

        self.index.add(vectors) #type: ignore

        print(f"[ADD] Index size: {self.index.ntotal}")

        self.save()

    # ---------------------------------------------------
    # SEARCH
    # ---------------------------------------------------
    def search(
        self,
        query_vector,
        k=5,
        filters=None
    ):
        """
        Search FAISS index with optional metadata filtering.

        Flow:
        1. Normalize query
        2. Run FAISS similarity search
        3. Apply metadata filters
        4. Return ranked chunks
        """

        print("\n[SEARCH] Starting vector search")

        # ---------------------------------------------------
        # SAFETY: Empty index
        # ---------------------------------------------------
        if self.index.ntotal == 0:

            print("[SEARCH] Empty FAISS index")

            return []

        # ---------------------------------------------------
        # PREPARE QUERY VECTOR
        # ---------------------------------------------------
        query_vector = np.array(query_vector).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        # ---------------------------------------------------
        # SEARCH MORE THAN K
        # Important:
        # If filtering removes chunks,
        # we still want enough remaining results.
        # ---------------------------------------------------
        search_k = min(
            max(k * 5, 20),
            self.index.ntotal
        )

        print(f"[SEARCH] Searching top {search_k}")

        # ---------------------------------------------------
        # FAISS SEARCH
        # ---------------------------------------------------
        distances, indices = self.index.search(
            query_vector,
            search_k
        ) # type: ignore

        results = []

        # ---------------------------------------------------
        # PROCESS RESULTS
        # ---------------------------------------------------
        for i, idx in enumerate(indices[0]):

            # Invalid index safety
            if idx == -1:
                continue

            # Out-of-range safety
            if idx >= len(self.texts):
                continue

            chunk = self.texts[idx].copy()

            # Add similarity score
            chunk["score"] = float(distances[0][i])

            # ---------------------------------------------------
            # APPLY METADATA FILTERS
            # ---------------------------------------------------
            if filters:

                matched = all(
                    str(chunk.get(key)).lower().strip()
                    ==
                    str(value).lower().strip()
                    for key, value in filters.items()
                )

                if not matched:
                    continue

            results.append(chunk)

        # ---------------------------------------------------
        # SORT RESULTS
        # ---------------------------------------------------
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        # Limit final output
        results = results[:k]

        # ---------------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------------
        print("\n[SEARCH DEBUG]")

        print(f"Returned Results: {len(results)}")

        for r in results[:5]:

            print(
                f"""
    Score      : {r['score']:.4f}
    File       : {r.get('file_name')}
    Section    : {r.get('section')}
    Category   : {r.get('category')}
    Chunk ID   : {r.get('chunk_id')}
    Preview    : {r['content'][:120]}
    """
            )

        return results
    # -------------------------
    # SAFE SAVE (CRITICAL FIX)
    # -------------------------
    def save(self):
        try:
            temp_index = self.index_file + ".tmp"
            temp_meta = self.meta_file + ".tmp"

            # Write to temp files first
            faiss.write_index(self.index, temp_index)

            with open(temp_meta, "wb") as f:
                pickle.dump(self.texts, f)

            # Atomic replace (safe)
            os.replace(temp_index, self.index_file)
            os.replace(temp_meta, self.meta_file)

            print("[SAVE] FAISS index and metadata saved safely")

        except Exception as e:
            print("[ERROR] Save failed:", str(e))

    # -------------------------
    # SAFE LOAD (CRITICAL FIX)
    # -------------------------
    def load(self):
        try:
            # Validate files
            if (
                not os.path.exists(self.index_file)
                or os.path.getsize(self.index_file) == 0
                or not os.path.exists(self.meta_file)
                or os.path.getsize(self.meta_file) == 0
            ):
                print("[LOAD] Invalid storage files → resetting")
                self._create_new()
                return

            self.index = faiss.read_index(self.index_file)

            with open(self.meta_file, "rb") as f:
                self.texts = pickle.load(f)

            # Consistency check
            if self.index.ntotal != len(self.texts):
                print("[WARNING] Index/Text mismatch → resetting")
                self._create_new()
                return

            print("[LOAD] FAISS index and metadata loaded")

        except Exception as e:
            print("[ERROR] Load failed:", e)
            print("[RECOVERY] Resetting storage")
            self._create_new()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        print("[RESET] Clearing storage")

        self._create_new()

        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

    # -------------------------
    # DELETE DOCUMENT
    # -------------------------
    def delete_document(self, doc_id):
        """
        Delete all chunks belonging to a document
        and rebuild the FAISS index.
        """

        print(f"[DELETE] Removing doc_id: {doc_id}")

        # Safety check
        if self.embed_model is None:
            raise ValueError("Embedding model is not initialized")

        # Keep all OTHER documents
        remaining = [
            t for t in self.texts
            if str(t.get("doc_id")).strip() != str(doc_id).strip()
        ]

        print(f"[DELETE] Remaining chunks: {len(remaining)}")

        # If no docs remain → reset everything
        if not remaining:
            self.reset()
            return True

        # Extract text content
        contents = [t["content"] for t in remaining]

        # Re-embed remaining chunks
        vectors = self.embed_model.encode(contents)

        vectors = np.array(vectors).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        # Create fresh index
        self.index = faiss.IndexFlatIP(self.dim)

        # Add vectors
        self.index.add(vectors) #type: ignore

        # Save remaining metadata
        self.texts = remaining

        # Persist
        self.save()

        print("[DELETE] Index rebuilt successfully")

        return True

    # -------------------------
    # CHECK DOC
    # -------------------------
    def document_exists(self, doc_id):
        return any(t.get("doc_id") == doc_id for t in self.texts)