import numpy as np
import re


class Retriever:

    def __init__(
        self,
        embed_model,
        vectorstore
    ):

        """
        Advanced Hybrid Retriever

        Responsibilities:
        - Query enhancement
        - Embedding generation
        - Vector search
        - Hybrid scoring
        - Metadata filtering
        - Retrieval debugging
        """

        self.embed_model = embed_model
        self.vectorstore = vectorstore

    # =====================================================
    # MAIN RETRIEVAL
    # =====================================================
    def retrieve(
        self,
        query,
        k=10,
        filters=None
    ):

        print("\n" + "=" * 60)
        print("[RETRIEVE] Starting Retrieval")
        print("=" * 60)

        print(f"[QUERY] {query}")

        # -------------------------------------------------
        # STEP 1: Enhance Query
        # -------------------------------------------------
        enhanced_query = self.expand_query(query)

        print(f"[ENHANCED QUERY] {enhanced_query}")

        # -------------------------------------------------
        # STEP 2: Encode Query
        # -------------------------------------------------
        query_vector = self.embed_model.encode(
            [enhanced_query]
        )

        query_vector = np.array(
            query_vector,
            dtype="float32"
        )

        # -------------------------------------------------
        # STEP 3: Dynamic Search Depth
        # -------------------------------------------------
        #
        # Search deeper than final k
        # so reranker gets better candidates
        #
        # -------------------------------------------------

        search_k = max(k * 4, 20)

        # -------------------------------------------------
        # STEP 4: Vector Search
        # -------------------------------------------------
        results = self.vectorstore.search(
            query_vector=query_vector,
            k=search_k,
            filters=filters
        )

        # -------------------------------------------------
        # STEP 5: Empty Safety
        # -------------------------------------------------
        if not results:

            print("[RETRIEVE] No results found")

            return []

        print(f"[RETRIEVE] Raw results: {len(results)}")

        # -------------------------------------------------
        # STEP 6: Hybrid Scoring
        # -------------------------------------------------
        for r in results:

            # -----------------------------
            # Semantic score from FAISS
            # -----------------------------
            semantic_score = float(
                r.get("score", 0)
            )

            # -----------------------------
            # Keyword overlap score
            # -----------------------------
            keyword_score = self.keyword_score(
                query=query,
                text=r["content"]
            )

            # -----------------------------
            # Exact phrase bonus
            # -----------------------------
            phrase_bonus = self.phrase_match_score(
                query=query,
                text=r["content"]
            )

            # -----------------------------
            # Final Hybrid Score
            # -----------------------------
            #
            # Weighted fusion:
            #
            # semantic = strongest signal
            # keyword = grounding signal
            # phrase = precision boost
            #
            # -----------------------------
            hybrid_score = (
                semantic_score * 0.75
                + keyword_score * 0.20
                + phrase_bonus * 0.05
            )

            # Store scores
            r["semantic_score"] = semantic_score
            r["keyword_score"] = keyword_score
            r["phrase_score"] = phrase_bonus
            r["hybrid_score"] = hybrid_score

        # -------------------------------------------------
        # STEP 7: Sort by Hybrid Score
        # -------------------------------------------------
        results = sorted(
            results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        # -------------------------------------------------
        # STEP 8: Adaptive Threshold Filtering
        # -------------------------------------------------
        filtered = []

        for r in results:

            semantic = r["semantic_score"]
            hybrid = r["hybrid_score"]

            # Strong semantic match
            if semantic >= 0.55:
                filtered.append(r)
                continue

            # Medium semantic + strong hybrid
            if semantic >= 0.40 and hybrid >= 0.45:
                filtered.append(r)
                continue

            # Strong keyword grounding
            if r["keyword_score"] >= 0.20:
                filtered.append(r)
                continue

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------
        #
        # If everything filtered out,
        # keep top few results.
        #
        # -------------------------------------------------

        if not filtered:

            print("[RETRIEVE] Using fallback results")

            filtered = results[:3]

        # Final top-k
        filtered = filtered[:k]

        # -------------------------------------------------
        # DEBUG OUTPUT
        # -------------------------------------------------
        self.debug_results(filtered)

        print(
            f"[RETRIEVE] Final chunks: {len(filtered)}"
        )

        return filtered

    # =====================================================
    # QUERY EXPANSION
    # =====================================================
    def expand_query(self, query):

        """
        Improve vague queries for retrieval.

        Especially important for:
        - short questions
        - technical PDFs
        - enterprise documents
        """

        query_lower = query.lower().strip()

        # -------------------------------------------------
        # Generic summary requests
        # -------------------------------------------------
        generic_patterns = [
            "summarize pdf",
            "summarize document",
            "explain pdf",
            "explain document",
            "what is this about",
            "document overview"
        ]

        if query_lower in generic_patterns:

            return (
                query
                + " project overview architecture "
                  "implementation technologies "
                  "summary key concepts"
            )

        # -------------------------------------------------
        # Very short queries
        # -------------------------------------------------
        if len(query_lower.split()) <= 3:

            return (
                query
                + " detailed explanation "
                  "technical context"
            )

        return query

    # =====================================================
    # KEYWORD SCORE
    # =====================================================
    def keyword_score(
        self,
        query,
        text
    ):

        """
        Better keyword overlap scoring.
        """

        # Clean words
        q_words = set(
            re.findall(r"\w+", query.lower())
        )

        t_words = set(
            re.findall(r"\w+", text.lower())
        )

        if not q_words:
            return 0.0

        overlap = q_words.intersection(t_words)

        return len(overlap) / len(q_words)

    # =====================================================
    # PHRASE MATCH BOOST
    # =====================================================
    def phrase_match_score(
        self,
        query,
        text
    ):

        """
        Boost exact phrase matches.
        """

        query = query.lower().strip()
        text = text.lower()

        if query in text:
            return 1.0

        return 0.0

    # =====================================================
    # DEBUGGING
    # =====================================================
    def debug_results(self, results):

        print("\n" + "=" * 60)
        print("[RETRIEVAL DEBUG]")
        print("=" * 60)

        for i, r in enumerate(results[:10], start=1):

            print(
                f"""
                Result #{i}

                File            : {r.get('file_name')}
                Doc ID          : {r.get('doc_id')}
                Chunk ID        : {r.get('chunk_id')}

                Semantic Score  : {r['semantic_score']:.4f}
                Keyword Score   : {r['keyword_score']:.4f}
                Phrase Score    : {r['phrase_score']:.4f}
                Hybrid Score    : {r['hybrid_score']:.4f}

                Preview:
                {r['content'][:200]}
                """
            )