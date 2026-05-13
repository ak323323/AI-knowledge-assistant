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
            You are a highly knowledgeable AI assistant.

            Your task:
            - Give a clear, detailed explanation
            - Use simple language first, then go deeper
            - Include examples where helpful
            - Structure the answer (bullets or paragraphs)

            IMPORTANT:
            - Answer ONLY using the provided context
            - If the answer is not clearly present, say:
            "I could not find this in the document"
            - Do NOT use general knowledge

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
        results = self.retriever.retrieve(question,
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

        #  Compute Confidence
        top_score = results[0]["rerank_score"]
        avg_score = sum(r["rerank_score"] for r in results[:3]) / min(3, len(results))

        print(f"[DEBUG] Top: {top_score:.3f} | Avg: {avg_score:.3f}")

        if top_score < 0.15:   #  tune this (0.3–0.5)
            print("[WARNING] Weak retrieval — using fallback top chunks")
        
        # detect generic queries
        is_generic = len(question.split()) <= 4

        #  FALLBACK: If generic query OR no strong matches → summarize whole doc
        if (
            top_score < WEAK_SCORE
            and avg_score < WEAK_SCORE
            and not is_generic):

            print("[RAG] Using fallback summary mode")

            #  Use ALL chunks (not just top-k)
            all_chunks = [t["content"] for t in self.retriever.vectorstore.texts]

            # limit to avoid overload
            context_text = "\n\n".join(all_chunks[:5])

            query = self.enhance_query(question)
            
            #prompt for summary mode
            prompt = f"""
                You are an AI Knowledge assistant.

                Summarize the following document clearly.
                If the context is partially relevant:
                Try your best to answer using available information.

                If completely unrelated:
                Say: "This question is not related to the document."

                Context:
                {context_text}

                Query:
                {query}

                Provide:
                - Overview
                - Key points
                - Technologies / concepts used
                """

            answer = self.llm.generate(prompt)

            return {
                "answer": answer,
                "sources": self.format_sources(results)
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

        return query

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
        