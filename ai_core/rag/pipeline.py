from rag.reranker import Reranker


MIN_SCORE = 3.0      # strong match
WEAK_SCORE = -2.0     # borderline

class RAGPipeline:
    def __init__(self, retriever, llm):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Handles vector search
            llm: Ollama client
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = Reranker()

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        """
        Build final prompt using retrieved context + user question.

        Args:
            question (str): User query
            contexts (list[str]): Retrieved chunk texts

        Returns:
            str: Final prompt for LLM
        """

        #  Join the retieved chunks to make a complete context block
        context_text = "\n\n".join(contexts)

        #  Construct instruction-based prompt
        prompt = f"""
            You are a document-grounded AI assistant.

            You MUST answer ONLY using the provided context.

            STRICT RULES:
            - Do NOT use outside knowledge
            - Do NOT guess
            - Do NOT infer missing information
            - If the answer is not explicitly present in context,
            say exactly:

            "I could not find this in the document."

            Context:
            {context_text}

            Question:
            {query}

            Answer:
            """ 
        return prompt

    def run(self, question: str,filters=None,top_k=5):
        """
        Execute full RAG flow:
        1. Retrieve relevant chunks
        2. Build prompt
        3. Call LLM
        4. Return answer + sources
        """

        # Retrieve relevant chunks
        enhanced_query = self.enhance_query(question)
        print("\n[QUERY ENHANCEMENT]")
        print("Original :", question)
        print("Enhanced:", enhanced_query)
        results = self.retriever.retrieve(enhanced_query,
                                          k=top_k,
                                          filters=filters)
        
        # PRE-FILTER WEAK SEMANTIC MATCHES
        results = [
            r for r in results
            if r["semantic_score"] >= 0.30
        ]

        print(f"[RAG] After semantic filtering: {len(results)}")

         #  HARD STOP: no relevant chunks
        if not results:
            return {
            "answer": "This question is not related to the uploaded document.",
            "sources": []
        }

        #   RERANK
        results = self.reranker.rerank(question, results)

        if not self.is_relevant(results):

            print("[RAG] Query rejected as irrelevant")

            return {
                "answer": (
                    "This question is not related "
                    "to the uploaded document."
                ),
                "sources": []
            }

        #  Compute Confidence
        top_score = results[0]["rerank_score"]
        avg_score = sum(r["rerank_score"] for r in results[:3]) / min(3, len(results))

        print(f"[DEBUG] Top: {top_score:.3f} | Avg: {avg_score:.3f}")

        if top_score < 0.10:   #  tune this (0.3–0.5)
            print("[WARNING] Weak retrieval — using fallback top chunks")
        
        # detect generic queries
        is_generic = len(question.split()) <= 4

        #  FALLBACK: If generic query OR no strong matches → summarize whole doc
        if not self.is_relevant(results):

            return {
                "answer": (
                    "This question is not related "
                    "to the uploaded document."
                ),
                "sources": []
            }


        # Extract ONLY text content for LLM(only top 3 Chunks)
        top_results = results[:3]
        contexts = [r["content"] for r in top_results]

        # Build prompt
        prompt = self.build_prompt(question, contexts)

        # Call LLM 
        answer = self.llm.generate(prompt)
        
        #debug block for checking bad embeddings, irrelevant chunks, threshold strictness
        for r in results:
            print(f"[DEBUG] Score: {r['score']:.3f} | {r['content'][:100]}")

        print(type(results))
        print(results)

        # Return structured response for UI
        return {
            "answer": answer,
            "sources": self.format_sources(top_results)   # send full metadata to UI
        }
    

    def enhance_query(self, query: str):
        """
        Expand vague user queries into
        retrieval-friendly semantic queries.
        """

        query_lower = query.lower().strip()

        # -----------------------------------
        # Generic document summary requests
        # -----------------------------------
        generic_patterns = [
            "explain the pdf",
            "summarize the pdf",
            "explain document",
            "summarize document",
            "what is this pdf about",
            "explain this file",
            "what is this about"
        ]

        if query_lower in generic_patterns:

            return """
            document summary
            project overview
            main topics
            technologies used
            implementation details
            architecture
            key concepts
            """
        
        # ------------------------------------------------
        # SHORT QUERIES
        # ------------------------------------------------

        if len(query_lower.split()) <= 3:

            return (
                query
                + " detailed explanation technical context"
            )

        # -----------------------------------
        # API related queries
        # -----------------------------------
        if "api" in query_lower:
            return query + """
            endpoints backend routes swagger json response
            """

        # -----------------------------------
        # Database related queries
        # -----------------------------------
        if "database" in query_lower:
            return query + """
            mysql storage dapper orm retrieval
            """

        # default
        return query
    
    def format_sources(self, results):
        formatted = []

        for r in results:

            formatted.append({
                "content": r.get("content", ""),
                "source": r.get("source", "unknown"),
                "file_name": r.get("file_name", "unknown"),
                "category": r.get("category", "unknown"),
                "chunk_id": r.get("chunk_id", -1),
                "score": float(
                    r.get(
                        "rerank_score",
                        r.get("score", 0)
                    )
                )
            })
        return formatted
    
    def is_relevant(self, results):
        """
        Decide whether retrieved chunks are
        actually relevant to the user query.
        """

        if not results:
            return False

        top_semantic = results[0].get("semantic_score", 0)
        top_rerank = results[0].get("rerank_score", -999)

        avg_semantic = sum(
            r.get("semantic_score", 0)
            for r in results[:3]
        ) / min(3, len(results))

        avg_rerank = sum(
            r.get("rerank_score", -999)
            for r in results[:3]
        ) / min(3, len(results))

        print("\n[RELEVANCE CHECK]")
        print(f"Top Semantic : {top_semantic:.3f}")
        print(f"Avg Semantic : {avg_semantic:.3f}")
        print(f"Top Rerank   : {top_rerank:.3f}")
        print(f"Avg Rerank   : {avg_rerank:.3f}")

        # STRICT RELEVANCE RULES
        if top_semantic < 0.45:
            return False

        if avg_semantic < 0.40:
            return False

        if top_rerank < 2.0:
            return False

        return True
        