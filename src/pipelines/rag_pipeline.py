
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import Retriever
from src.generation.llm import LLMGenerator
from src.generation.prompt_templates import build_rag_prompt
from src.memory.conversation import ConversationMemory


class RAGPipeline:
    """
    Connect query rewriting, document retrieval,
    prompt creation, conversation memory and
    LLM answer generation.
    """

    def __init__(
        self,
        retriever,
        query_rewriter,
        llm_generator,
        conversation_memory=None
    ):

        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.llm_generator = llm_generator

        self.conversation_memory = (
            conversation_memory
            if conversation_memory is not None
            else ConversationMemory()
        )

        # Stores metadata from the latest streaming request
        self.last_stream_metadata = {}

        print("RAG Pipeline initialized successfully!")

    def ask(
        self,
        question: str,
        chat_history: str = "",
        top_k: int = 5,
        min_score: float = 0.20
    ) -> dict:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Get conversation history
        history = (
            chat_history
            if chat_history
            else self.conversation_memory.get_history()
        )

        # Step 2: Rewrite the question
        rewritten_query = self.query_rewriter.rewrite_query(
            query=question,
            chat_history=history
        )

        # Step 3: Retrieve relevant documents
        retrieved_documents = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            min_score=min_score
        )

        # Step 4: Handle empty retrieval
        if not retrieved_documents:

            answer = (
                "I could not find sufficient information "
                "in the uploaded documents."
            )

            self.conversation_memory.add_message(
                user_message=question,
                assistant_message=answer
            )

            return {
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": answer,
                "sources": [],
                "retrieved_chunks": 0
            }

        # Step 5: Prepare context and sources
        context_parts = []
        sources = []

        for document in retrieved_documents:

            metadata = document["metadata"]

            source_name = metadata.get("file_name", "Unknown")
            page_number = metadata.get("page", "Unknown")
            score = document["score"]

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
                "score": round(score, 4)
            })

        context = "\n".join(context_parts)

        # Step 6: Build RAG prompt
        prompt = build_rag_prompt(
            question=question,
            context=context,
            chat_history=history
        )

        # Step 7: Generate complete answer
        answer = self.llm_generator.generate(prompt)

        # Step 8: Save conversation
        self.conversation_memory.add_message(
            user_message=question,
            assistant_message=answer
        )

        # Step 9: Return response
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_documents)
        }

    def ask_stream(
        self,
        question: str,
        chat_history: str = "",
        top_k: int = 5,
        min_score: float = 0.20
    ):
        """
        Generate a streaming RAG answer.

        Answer chunks are yielded as strings.
        Source information is stored separately
        in self.last_stream_metadata.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Reset metadata for the new request
        self.last_stream_metadata = {}

        # Step 1: Get conversation history
        history = (
            chat_history
            if chat_history
            else self.conversation_memory.get_history()
        )

        # Step 2: Rewrite the question
        rewritten_query = self.query_rewriter.rewrite_query(
            query=question,
            chat_history=history
        )

        # Step 3: Retrieve relevant documents
        retrieved_documents = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            min_score=min_score
        )

        # Step 4: Handle empty retrieval
        if not retrieved_documents:

            answer = (
                "I could not find sufficient information "
                "in the uploaded documents."
            )

            self.last_stream_metadata = {
                "question": question,
                "rewritten_query": rewritten_query,
                "sources": [],
                "retrieved_chunks": 0,
                "answer": answer
            }

            self.conversation_memory.add_message(
                user_message=question,
                assistant_message=answer
            )

            yield answer
            return

        # Step 5: Prepare context and source metadata
        context_parts = []
        sources = []

        for document in retrieved_documents:

            metadata = document["metadata"]

            source_name = metadata.get("file_name", "Unknown")
            page_number = metadata.get("page", "Unknown")
            score = document["score"]

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
                "score": round(score, 4)
            })

        context = "\n".join(context_parts)

        # Step 6: Build RAG prompt
        prompt = build_rag_prompt(
            question=question,
            context=context,
            chat_history=history
        )

        # Step 7: Store metadata before streaming
        self.last_stream_metadata = {
            "question": question,
            "rewritten_query": rewritten_query,
            "sources": sources,
            "retrieved_chunks": len(retrieved_documents),
            "answer": ""
        }

        # Step 8: Generate streaming answer
        complete_answer = ""

        for chunk in self.llm_generator.generate_stream(prompt):

            complete_answer += chunk

            yield chunk

        # Step 9: Update metadata with complete answer
        self.last_stream_metadata["answer"] = complete_answer

        # Step 10: Save complete answer in memory
        if complete_answer.strip():

            self.conversation_memory.add_message(
                user_message=question,
                assistant_message=complete_answer
            )
