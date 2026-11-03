
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import Retriever
from src.generation.llm import LLMGenerator
from src.generation.prompt_templates import build_rag_prompt


class RAGPipeline:
    """
    Connect query rewriting, document retrieval,
    prompt creation and LLM answer generation.
    """

    def __init__(
        self,
        retriever,
        query_rewriter,
        llm_generator
    ):

        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.llm_generator = llm_generator

        print("RAG Pipeline initialized successfully!")

    def ask(
        self,
        question: str,
        chat_history: str = "",
        top_k: int = 5,
        min_score: float = 0.20
    ) -> dict:
        """
        Process a user question and return an answer
        along with retrieved source information.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Rewrite the user question
        rewritten_query = self.query_rewriter.rewrite_query(
            query=question,
            chat_history=chat_history
        )

        # Step 2: Retrieve relevant document chunks
        retrieved_documents = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            min_score=min_score
        )

        # Step 3: Handle empty retrieval results
        if not retrieved_documents:
            return {
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": (
                    "I could not find sufficient information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "retrieved_chunks": 0
            }

        # Step 4: Prepare context from retrieved chunks
        context_parts = []

        sources = []

        for document in retrieved_documents:

            metadata = document["metadata"]

            source_name = metadata.get(
                "file_name",
                "Unknown"
            )

            page_number = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
Source: {source_name}
Page: {page_number}

Content:
{document["page_content"]}
"""
            )

            sources.append({
                "file_name": source_name,
                "page": page_number,
                "score": round(document["score"], 4)
            })

        context = "\n".join(context_parts)

        # Step 5: Build the RAG prompt
        prompt = build_rag_prompt(
            question=question,
            context=context,
            chat_history=chat_history
        )

        # Step 6: Generate answer using Gemini
        answer = self.llm_generator.generate(prompt)

        # Step 7: Return answer and source details
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_documents)
        }
