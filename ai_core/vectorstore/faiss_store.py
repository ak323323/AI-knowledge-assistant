import faiss
import numpy as np
import os
import pickle
import uuid
from datetime import datetime
from storage.document_registry import DocumentRegistry


class FAISSStore:

    def __init__(
        self,
        dim,
        embed_model,
        storage_path="storage"
    ):

        self.dim = dim
        self.embed_model = embed_model
        self.storage_path = storage_path

        # -----------------------------------
        # Chunk storage
        # -----------------------------------
        self.texts = []

        # -----------------------------------
        # Document registry
        # Tracks uploaded files
        # -----------------------------------
        self.documents = {}

        os.makedirs(storage_path, exist_ok=True)

        # -----------------------------------
        # Storage files
        # -----------------------------------
        self.index_file = os.path.join(
            storage_path,
            "faiss.index"
        )

        self.meta_file = os.path.join(
            storage_path,
            "metadata.pkl"
        )

        self.docs_file = os.path.join(
            storage_path,
            "documents.pkl"
        )

        # -----------------------------------
        # Load existing index
        # -----------------------------------
        if (
            os.path.exists(self.index_file)
            and os.path.exists(self.meta_file)
        ):
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

        # =====================================================
        # BASE INDEX
        # =====================================================

        base_index = faiss.IndexFlatIP(self.dim)

        # =====================================================
        # WRAP WITH ID MAP
        # =====================================================

        self.index = faiss.IndexIDMap(base_index)

        # =====================================================
        # METADATA STORAGE
        # =====================================================

        self.texts = []

    # -------------------------
    # ADD DATA
    # -------------------------
    def add(self,vectors,texts,metadata=None,source="unknown",doc_id=None):

        print("\n[ADD] Adding document chunks")

        # -----------------------------------
        # Generate doc_id if missing
        # -----------------------------------
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # -----------------------------------
        # Convert vectors
        # -----------------------------------
        vectors = np.array(vectors).astype("float32")

        # -----------------------------------
        # Validate vectors
        # -----------------------------------
        if np.isnan(vectors).any():
            raise ValueError("NaN embeddings detected")

        if np.isinf(vectors).any():
            raise ValueError("Inf embeddings detected")

        # Ensure 2D
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # Dimension check
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim {self.dim}, got {vectors.shape[1]}"
            )

        # -----------------------------------
        # Normalize for cosine similarity
        # -----------------------------------
        faiss.normalize_L2(vectors)

        # -----------------------------------
        # Register document
        # -----------------------------------
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "file_name": os.path.basename(source),
            "source": source,
            "uploaded_at": datetime.now().isoformat(),
            "total_chunks": len(texts)
        }

        # -----------------------------------
        # Base chunk index
        # -----------------------------------
        base_index = self.index.ntotal

        # -----------------------------------
        # Store chunk metadata
        # -----------------------------------
        for i, text in enumerate(texts):

            meta = metadata[i] if metadata else {}

            chunk_data = {

                # =====================================================
                # CORE CONTENT
                # =====================================================

                "content": text,

                # =====================================================
                # VECTOR STORAGE (CRITICAL UPGRADE)
                # =====================================================

                # Store embedding vector directly
                "embedding": vectors[i].tolist(),

                # =====================================================
                # DOCUMENT METADATA
                # =====================================================

                "source": source,

                "file_name": meta.get(
                    "file_name",
                    os.path.basename(source)
                ),

                "document_title": meta.get(
                    "document_title",
                    os.path.basename(source)
                ),

                "doc_id": doc_id,

                # =====================================================
                # CHUNK METADATA
                # =====================================================

                "chunk_id": base_index + i,

                "section": meta.get(
                    "section",
                    "General"
                ),

                "subsection": meta.get(
                    "subsection",
                    "General"
                ),

                "chunk_length": meta.get(
                    "chunk_length",
                    len(text)
                ),

                "word_count": meta.get(
                    "word_count",
                    len(text.split())
                ),

                # =====================================================
                # SEARCH METADATA
                # =====================================================

                "keywords": meta.get(
                    "keywords",
                    []
                ),

                "tags": meta.get(
                    "tags",
                    []
                ),

                "category": meta.get(
                    "category",
                    "general"
                ),

                # =====================================================
                # TIMESTAMP
                # =====================================================

                "uploaded_at": datetime.now().isoformat()
            }

            self.texts.append(chunk_data)

        # -----------------------------------
        # Add vectors to FAISS
        # -----------------------------------
        vector_ids = np.arange(base_index, base_index + len(vectors)).astype("int64")

        self.index.add_with_ids(vectors, vector_ids) #type: ignore

        print(f"[ADD] Added {len(texts)} chunks")
        print(f"[ADD] Index size = {self.index.ntotal}")

        self.save()

        return doc_id

    # ---------------------------------------------------
    # SEARCH
    # ---------------------------------------------------
    def search(self,query_vector,k=5,filters=None):
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

            # Optional document filter
            if filters:

                matched = True

                for key, value in filters.items():

                    if str(chunk.get(key)).lower().strip() != str(value).lower().strip():

                        matched = False
                        break

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
            temp_docs = self.docs_file + ".tmp"

            # Save FAISS index
            faiss.write_index(self.index, temp_index)

            # Save chunk metadata
            with open(temp_meta, "wb") as f:
                pickle.dump(self.texts, f)

            # Save document registry
            with open(temp_docs, "wb") as f:
                pickle.dump(self.documents, f)

            # Atomic replace
            os.replace(temp_index, self.index_file)
            os.replace(temp_meta, self.meta_file)
            os.replace(temp_docs, self.docs_file)

            print("[SAVE] Storage saved successfully")

        except Exception as e:
            print("[SAVE ERROR]", e)

    # -------------------------
    # SAFE LOAD (CRITICAL FIX)
    # -------------------------
    def load(self):

        try:

            print("[LOAD] Loading FAISS storage")

            self.index = faiss.read_index(
                self.index_file
            )

            with open(self.meta_file, "rb") as f:
                self.texts = pickle.load(f)

            # Load document registry
            if os.path.exists(self.docs_file):

                with open(self.docs_file, "rb") as f:
                    self.documents = pickle.load(f)

            else:
                self.documents = {}

            # Consistency check
            if self.index.ntotal != len(self.texts):

                print("[WARNING] Index mismatch")

                self._create_new()

                return

            print("[LOAD] Storage loaded successfully")
            print(f"[LOAD] Chunks: {len(self.texts)}")
            print(f"[LOAD] Documents: {len(self.documents)}")

        except Exception as e:

            print("[LOAD ERROR]", e)

            self._create_new()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):

        print("[RESET] Clearing storage")

        self._create_new()

        for file in [
            self.index_file,
            self.meta_file,
            self.docs_file
        ]:

            if os.path.exists(file):
                os.remove(file)

        registry = DocumentRegistry()

        registry.clear()

    # -------------------------
    # DELETE DOCUMENT
    # -------------------------
    def delete_document(self, doc_id):
        """
        Delete all chunks belonging to a document
        and rebuild FAISS WITHOUT re-embedding.
        """

        print(f"\n[DELETE] Removing document: {doc_id}")

        # =====================================================
        # STEP 1: KEEP ALL OTHER DOCUMENTS
        # =====================================================

        remaining_chunks = [

            chunk
            for chunk in self.texts
            if str(chunk.get("doc_id")).strip()
            != str(doc_id).strip()
        ]

        removed_count = len(self.texts) - len(remaining_chunks)

        print(f"[DELETE] Removed chunks: {removed_count}")

        print(f"[DELETE] Remaining chunks: {len(remaining_chunks)}")

        # =====================================================
        # STEP 2: RESET IF EMPTY
        # =====================================================

        if not remaining_chunks:

            print("[DELETE] No documents remain")

            self.reset()

            return True

        # =====================================================
        # STEP 3: REBUILD FAISS USING STORED VECTORS
        # =====================================================

        print("[DELETE] Rebuilding FAISS index...")

        vectors = np.array(

            [
                chunk["embedding"]
                for chunk in remaining_chunks
            ],

            dtype=np.float32
        )

        # Safety normalization
        faiss.normalize_L2(vectors)

        # =====================================================
        # CREATE NEW BASE INDEX
        # =====================================================

        base_index = faiss.IndexFlatIP(self.dim)

        # =====================================================
        # WRAP WITH INDEX ID MAP
        # =====================================================

        self.index = faiss.IndexIDMap(base_index)

        # =====================================================
        # REBUILD VECTOR IDS
        # =====================================================

        vector_ids = np.arange(len(vectors)).astype("int64")

        # =====================================================
        # ADD VECTORS WITH IDS
        # =====================================================

        self.index.add_with_ids(vectors, vector_ids) #type: ignore

        # =====================================================
        # STEP 4: SAVE METADATA
        # =====================================================

        self.texts = remaining_chunks

        # =====================================================
        # STEP 5: PERSIST STORAGE
        # =====================================================

        self.save()

        print("[DELETE] Document removed successfully")

        print(f"[DELETE] New index size: {self.index.ntotal}")

        return True

    # -------------------------
    # CHECK DOC
    # -------------------------
    def document_exists(self, doc_id):
        return any(t.get("doc_id") == doc_id for t in self.texts)
    

    def list_documents(self):

        """
        Return uploaded document registry.
        """

        return list(self.documents.values())