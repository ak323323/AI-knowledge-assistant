import numpy as np
from sentence_transformers import CrossEncoder
import torch


# =========================================================
# GLOBAL MODEL CACHE
# =========================================================
# We load the reranker only ONCE.
#
# Without this:
# - model reloads repeatedly
# - HuggingFace HEAD requests repeat
# - slower responses
# - possible WinError 10054 issues
#
# With lazy loading:
# - first request loads model
# - later requests reuse memory
# =========================================================

_reranker_model = None


def get_reranker_model():
    """
    Lazy-load and cache the CrossEncoder model.

    Returns:
        CrossEncoder: Loaded reranker model
    """

    global _reranker_model

    # Load only once
    if _reranker_model is None:

        print("[MODEL] Loading reranker model...")

        # GPU Support
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-base",
            device=device
        )

        print("[MODEL] Reranker loaded")

    return _reranker_model


class Reranker:
    """
    Cross-encoder reranker.

    Purpose:
    Improve retrieval quality by re-scoring
    retrieved chunks against the user query.

    Flow:
    Query + Chunk
           ↓
    CrossEncoder predicts relevance
           ↓
    Scores normalized with sigmoid
           ↓
    Results sorted by relevance
    """

    def __init__(self):

        # Reuse cached model
        self.model = get_reranker_model()

    def rerank(self, query, results):
        """
        Rerank retrieved chunks.

        Args:
            query (str):
                User question

            results (list):
                Retrieved chunks from FAISS/BM25

        Returns:
            list:
                Sorted chunks with rerank_score
        """

        # =====================================================
        # SAFETY CHECK
        # =====================================================

        if not results:
            print("[RERANK] No results to rerank")
            return []

        print(f"[RERANK] Reranking {len(results)} chunks")

        # =====================================================
        # BUILD QUERY-CHUNK PAIRS
        # =====================================================
        #
        # CrossEncoder expects:
        #
        # [
        #   ("query", "chunk1"),
        #   ("query", "chunk2")
        # ]
        #
        # =====================================================

        pairs = []

        for r in results:

            enriched_content = f"""
        Section: {r.get('section', '')}

        Document: {r.get('file_name', '')}

        Content:
        {r['content']}
        """

            pairs.append(
                (query, enriched_content)
            )

        # =====================================================
        # MODEL INFERENCE
        # =====================================================
        #
        # Output:
        # raw logits
        #
        # Example:
        # [2.1, -1.3, 5.7]
        #
        # Higher = more relevant
        # =====================================================

        scores = self.model.predict(pairs)

        # =====================================================
        # CONVERT TO NUMPY ARRAY
        # =====================================================
        #
        # Important:
        # - fixes Pylance operator issues
        # - enables vectorized math
        #
        # =====================================================

        scores = np.array(scores, dtype=np.float32)

        print("\n[RERANK SCORES]")

        for i, score in enumerate(scores):

            preview = results[i]["content"][:120].replace("\n", " ")

            print(f"""
            Score   : {score:.4f}
            Preview : {preview}
            """)

        # =====================================================
        # ATTACH SCORES
        # =====================================================

        for i, r in enumerate(results):

            r["rerank_score"] = float(scores[i])
        
        print("\n[RERANK DEBUG SAMPLE]")
        print(results[0])

        # =====================================================
        # SORT RESULTS
        # =====================================================

        ranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        print("\n[RERANK DISTRIBUTION]")

        for i, r in enumerate(ranked[:10]):

            print(
                f"{i+1}. "
                f"{r['rerank_score']:.4f} | "
                f"{r.get('section', 'Unknown')}"
            )

        return ranked