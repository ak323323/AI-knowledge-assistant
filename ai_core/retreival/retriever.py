import numpy as np


class Retriever:
    def __init__(self, embed_model, vectorstore):
        """
        Retriever orchestrates:
        - query embedding
        - vector search
        - optional reranking
        """

        self.embed_model = embed_model
        self.vectorstore = vectorstore

    # ---------------------------------------------------
    # MAIN RETRIEVAL FUNCTION
    # ---------------------------------------------------
    def retrieve(
        self,
        query,
        k=10,
        source_filter=None,
        filters=None
    ):
        """
        Retrieve relevant chunks from vector DB.

        Flow:
        1. Encode query
        2. Search FAISS
        3. Apply metadata filters
        4. Add keyword scores
        5. Return ranked chunks
        """

        print("\n[RETRIEVE] Incoming query:")
        print(query)

        # ---------------------------------------------------
        # STEP 1: Encode query
        # ---------------------------------------------------
        query_vector = self.embed_model.encode([query])

        query_vector = np.array(query_vector).astype("float32")

        # ---------------------------------------------------
        # STEP 2: Vector search
        # Filtering handled INSIDE vectorstore
        # ---------------------------------------------------
        results = self.vectorstore.search(
            query_vector=query_vector,
            k=k,
            filters=filters
        )

        # ---------------------------------------------------
        # STEP 3: Optional source filtering
        # ---------------------------------------------------
        if source_filter:
            results = [
                r for r in results
                if source_filter.lower() in r["source"].lower()
            ]

        # ---------------------------------------------------
        # STEP 4: No results safety
        # ---------------------------------------------------
        if not results:
            print("[RETRIEVE] No matching results")
            return []

        print(f"[RETRIEVE] Retrieved {len(results)} chunks")

        # ---------------------------------------------------
        # STEP 5: Attach hybrid scores
        # ---------------------------------------------------
        for r in results:

            # FAISS semantic score
            r["semantic_score"] = float(r["score"])

            # Simple keyword overlap score
            r["keyword_score"] = self.keyword_score(
                query,
                r["content"]
            )

        # ---------------------------------------------------
        # STEP 6: Sort by semantic score
        # ---------------------------------------------------
        results = sorted(
            results,
            key=lambda x: x["semantic_score"],
            reverse=True
        )

        # ---------------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------------
        print("\n[RETRIEVAL DEBUG]")

        for r in results[:5]:

            print(
                f"""
                Semantic Score : {r['semantic_score']:.4f}
                Keyword Score  : {r['keyword_score']:.4f}
                File           : {r.get('file_name')}
                Category       : {r.get('category')}
                Preview        : {r['content'][:120]}
                """
            )

        return results

    # ---------------------------------------------------
    # KEYWORD SCORE
    # ---------------------------------------------------
    def keyword_score(self, query, text):
        """
        Simple normalized keyword overlap.

        Helps hybrid retrieval.
        """

        q_words = set(query.lower().split())
        t_words = set(text.lower().split())

        if not q_words:
            return 0.0

        overlap = q_words & t_words

        return len(overlap) / (len(q_words) + len(t_words))