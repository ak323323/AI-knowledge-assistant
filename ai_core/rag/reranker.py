import numpy as np
from sentence_transformers import CrossEncoder
import torch
from scipy.special import expit


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
                Document Name:
                {r.get('file_name', '')}

                Section:
                {r.get('section', '')}

                Category:
                {r.get('category', '')}

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

        # Raw logits
        scores = np.array(scores, dtype=np.float32)

        # Normalize to 0-1 probability
        normalized_scores = expit(scores)

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

            # 0-1 normalized confidence
            r["rerank_confidence"] = float(normalized_scores[i])    
        
        print("\n[RERANK DEBUG SAMPLE]")
        print(results[0])

        filtered = [
            r for r in results
            if r["rerank_confidence"] >= 0.55
        ]

        if not filtered:
            print("[RERANK] No confident matches")
            return []
        

        #======================================================
        # Duplicate chunk removal
        #======================================================
        seen = set()
        unique_results = []

        for r in results:

            key = r["content"][:200]

            if key in seen:
                continue

            seen.add(key)
            unique_results.append(r)

        results = unique_results

        # =====================================================
        # SORT RESULTS
        # =====================================================

        ranked = sorted(
            results,
            key=lambda x: x["rerank_confidence"],
            reverse=True
        )

        print("\n[RERANK RESULTS]")

        for i, r in enumerate(ranked[:10]):

            preview = r["content"][:100].replace("\n", " ")

            print(
                f"""
                Rank        : {i+1}
                Confidence  : {r['rerank_confidence']:.4f}
                Raw Score   : {r['rerank_score']:.4f}
                File        : {r.get('file_name')}
                Section     : {r.get('section')}
                Preview     : {preview}
                """
            )

        return ranked