from rag.reranker import Reranker



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
            You are a highly intelligent document-grounded AI assistant.

            Your job is to answer the user's question ONLY using the provided context.

            =====================================================
            STRICT GROUNDING RULES
            =====================================================

            - Use ONLY information from the context
            - Do NOT use outside knowledge
            - Do NOT invent facts
            - Do NOT guess missing information
            - If information is missing, say:

            "I could not find this in the document."

            =====================================================
            ANSWER QUALITY RULES
            =====================================================

            Provide:

            1. A clear direct answer first
            2. Detailed explanation
            3. Technical breakdown when applicable
            4. Step-by-step explanation if relevant
            5. Examples from the document if available
            6. Bullet points where useful
            7. Structured formatting
            8. Mention important technologies/concepts
            9. Explain relationships between components
            10. Be comprehensive but grounded

            =====================================================
            CONTEXT
            =====================================================

            {context_text}

            =====================================================
            QUESTION
            =====================================================

            {query}

            =====================================================
            ANSWER
            =====================================================
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

        # =====================================================
        # STEP 1: QUERY CLASSIFICATION
        # =====================================================

        query_type = self.classify_query(question)

        print(f"\n[QUERY TYPE] {query_type}")

        # =====================================================
        # STEP 2: ENHANCE QUERY
        # =====================================================

        enhanced_query = self.enhance_query(question)

        print("\n[QUERY ENHANCEMENT]")
        print("Original :", question)
        print("Enhanced:", enhanced_query)

        # =====================================================
        # STEP 3: RETRIEVE CHUNKS
        # =====================================================

        dynamic_k = self.adaptive_top_k(query_type)

        results = self.retriever.retrieve(
            enhanced_query,
            k=dynamic_k,
            filters=filters
        )

        # =====================================================
        # STEP 4: REMOVE VERY WEAK SEMANTIC MATCHES
        # =====================================================

        results = [
            r for r in results
            if r["semantic_score"] >= 0.15
        ]

        print(f"[RAG] After semantic filtering: {len(results)}")

        # =====================================================
        # STEP 5: HARD STOP IF NOTHING FOUND
        # =====================================================

        if not results:
            return {
                "answer": (
                    "This question is not related "
                    "to the uploaded document."
                ),
                "sources": []
            }

        # =====================================================
        # STEP 6: RERANK RESULTS
        # =====================================================

        results = self.reranker.rerank(question, results)
        results = self.diversify_results(results)

        # =====================================================
        # STEP 7: STRICT RELEVANCE CHECK
        # =====================================================

        if not self.is_relevant(results):

            return {
                "answer": (
                    "This question is not related "
                    "to the uploaded document."
                ),
                "sources": []
            }

        # =====================================================
        # STEP 8: SUMMARY FALLBACK MODE
        # =====================================================

        if query_type == "summary":

            print("[RAG] Using document summary mode")

            # Take more chunks for overall understanding
            top_results = results[:12]

            contexts = []

            seen = set()

            for r in top_results:

                content = r["content"].strip()

                key = content[:200]

                if key in seen:
                    continue

                seen.add(key)

                contexts.append(content)

            context_text = "\n\n".join(contexts)

            prompt = f"""
            You are a document summarization assistant.

            STRICT RULES:
            - Use ONLY the provided context
            - Do NOT use outside knowledge
            - Summarize clearly and accurately

            Context:
            {context_text}

            Provide:
            - Document overview
            - Main topics
            - Technologies used
            - Important concepts
            """

            answer = self.llm.generate(prompt)

            return {
                "answer": answer,
                "sources": self.format_sources(top_results)
            }
        #======================================================
        # Extract ONLY text content for LLM
        # DYNAMIC CONTEXT WINDOW
        # =====================================================

        query_type = self.classify_query(question)

        if query_type == "summary":

            top_results = results[:8]

        elif query_type == "specific":

            top_results = results[:5]

        else:

            top_results = results[:4]
        
        contexts = [r["content"] for r in top_results]

        # Build prompt
        prompt = self.build_prompt(question, contexts)

        # Call LLM 
        answer = self.llm.generate(prompt)

        if ("not mentioned" in answer.lower() or "not provided" in answer.lower()):
            answer = "I could not find this in the document."
        
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
    
    def classify_query(self, query: str):
        """
        Classify query intent.

        Returns:
            summary -> asks about whole document
            specific -> asks document-specific question
            vague -> short unclear question
        """

        q = query.lower().strip()

        summary_patterns = [
            "summarize",
            "summary",
            "explain the pdf",
            "explain document",
            "what is this about",
            "overview",
            "describe document"
        ]

        # SUMMARY REQUESTS
        if any(p in q for p in summary_patterns):
            return "summary"

        # VERY SHORT QUERIES
        if len(q.split()) <= 2:
            return "vague"

        return "specific"
    
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
        Decide whether retrieved chunks
        are actually relevant to the query.
        """

        if not results:
            return False

        top_semantic = results[0].get(
            "semantic_score",
            0
        )

        top_confidence = results[0].get(
            "rerank_confidence",
            0
        )

        avg_semantic = sum(
            r.get("semantic_score", 0)
            for r in results[:3]
        ) / min(3, len(results))

        avg_confidence = sum(
            r.get("rerank_confidence", 0)
            for r in results[:3]
        ) / min(3, len(results))

        print("\n[RELEVANCE CHECK]")
        print(f"Top Semantic   : {top_semantic:.4f}")
        print(f"Avg Semantic   : {avg_semantic:.4f}")
        print(f"Top Confidence : {top_confidence:.4f}")
        print(f"Avg Confidence : {avg_confidence:.4f}")

        # ----------------------------------------
        # STRICT REJECTION RULES
        # ----------------------------------------

        # Completely unrelated
        if top_semantic < 0.20:
            return False

        # Reranker rejects relevance
        if top_confidence < 0.55:
            return False

        # Overall retrieval weak
        if avg_confidence < 0.50:
            return False

        return True
    
    def adaptive_top_k(self, query_type):
        """
        Dynamically adjust retrieval depth.
        """

        if query_type == "summary":
            return 10

        if query_type == "vague":
            return 8

        return 5
    
    def diversify_results(self, results, max_per_file=2):
        """
        Prevent one document from dominating retrieval.
        """

        diversified = []
        file_counts = {}

        for r in results:

            file_name = r.get("file_name", "unknown")

            count = file_counts.get(file_name, 0)

            if count >= max_per_file:
                continue

            diversified.append(r)

            file_counts[file_name] = count + 1

        return diversified
        