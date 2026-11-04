
import streamlit as st
import hashlib
import shutil
from pathlib import Path

from config.settings import UPLOAD_DIR

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from src.ingestion.ingestion_service import IngestionService

from src.retrieval.retriever import Retriever
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator

from src.memory.conversation import ConversationMemory
from src.pipelines.rag_pipeline import RAGPipeline


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Modular RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Modular RAG Assistant")

st.write(
    "Upload your PDF documents and ask questions "
    "using an AI-powered Retrieval Augmented Generation system."
)


# ==========================================
# 2. INITIALIZE BACKEND COMPONENTS
# ==========================================

@st.cache_resource
def initialize_backend():

    embedding_model = EmbeddingModel()

    vector_store = ChromaStore()

    ingestion_service = IngestionService(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model
    )

    query_rewriter = QueryRewriter()

    llm_generator = LLMGenerator()

    return (
        ingestion_service,
        vector_store,
        retriever,
        query_rewriter,
        llm_generator
    )


# ==========================================
# 3. INITIALIZE SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )

    st.divider()

    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    min_score = st.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        if st.session_state.rag_pipeline is not None:
            st.session_state.rag_pipeline.conversation_memory.clear_history()

        st.session_state.messages = []

        st.rerun()


# ==========================================
# 5. DOCUMENT INGESTION
# ==========================================

if process_button:

    try:

        ingestion_service, vector_store, _, _, _ = (
            initialize_backend()
        )

        results = []

        with st.spinner("Processing PDF documents..."):

            for uploaded_file in uploaded_files:

                try:

                    file_bytes = uploaded_file.getvalue()

                    # Create a unique folder using file content hash.
                    # This avoids collisions between same-named PDFs.
                    file_hash = hashlib.sha256(
                        file_bytes
                    ).hexdigest()

                    file_folder = UPLOAD_DIR / file_hash

                    file_folder.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    file_path = file_folder / uploaded_file.name

                    with open(file_path, "wb") as file:
                        file.write(file_bytes)

                    result = ingestion_service.ingest_pdf(
                        file_path
                    )

                    results.append(result)

                except Exception as error:

                    results.append({
                        "file_name": uploaded_file.name,
                        "status": "failed",
                        "error": str(error)
                    })

        st.session_state.ingestion_results = results

        st.rerun()

    except Exception as error:

        st.error(f"Backend initialization failed: {error}")


# ==========================================
# 6. DISPLAY DOCUMENT RESULTS
# ==========================================

if "ingestion_results" in st.session_state:

    for result in st.session_state.ingestion_results:

        if result["status"] == "success":

            st.success(
                f"Processed: {result['file_name']} | "
                f"Pages: {result['pages_processed']} | "
                f"Chunks: {result['chunks_created']}"
            )

        else:

            st.error(
                f"Failed: {result['file_name']} - "
                f"{result.get('error', 'Unknown error')}"
            )



# ==========================================
# 7. DOCUMENT MANAGEMENT
# ==========================================

st.divider()

st.subheader("📂 Indexed Documents")

try:

    if "vector_store" not in st.session_state:

        _, vector_store, _, _, _ = initialize_backend()

        st.session_state.vector_store = vector_store

    vector_store = st.session_state.vector_store

    document_summaries = vector_store.get_document_summary()

    st.sidebar.metric(
        "Total Stored Chunks",
        vector_store.get_collection_count()
    )

    if not document_summaries:

        st.info("No indexed documents found.")

    else:

        st.write(
            f"Total indexed entries: {len(document_summaries)}"
        )

        for document in document_summaries:

            source = Path(document["source"])

            document_key = hashlib.sha256(
                str(source).encode("utf-8")
            ).hexdigest()[:16]

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(
                        f"**📄 {document['file_name']}**"
                    )

                    st.caption(
                        f"Pages: {document['total_pages']} | "
                        f"Chunks: {document['total_chunks']}"
                    )

                with col2:

                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{document_key}"
                    ):

                        try:

                            # Delete vectors from ChromaDB
                            deleted_chunks = (
                                vector_store.delete_document(str(source))
                            )

                            # Delete physical PDF only if it is
                            # inside a hash-based upload folder
                            upload_root = UPLOAD_DIR.resolve()
                            resolved_source = source.resolve()

                            is_inside_uploads = (
                                resolved_source.is_relative_to(upload_root)
                            )

                            is_hash_folder = (
                                resolved_source.parent.parent == upload_root
                                and len(resolved_source.parent.name) == 64
                                and all(
                                    char in "0123456789abcdef"
                                    for char in resolved_source.parent.name.lower()
                                )
                            )

                            if is_inside_uploads and is_hash_folder:

                                if resolved_source.exists():
                                    resolved_source.unlink()

                                # Remove the hash folder if empty
                                if resolved_source.parent.exists():
                                    shutil.rmtree(
                                        resolved_source.parent,
                                        ignore_errors=True
                                    )

                            st.success(
                                f"Deleted {deleted_chunks} chunks "
                                f"from {document['file_name']}."
                            )

                            st.rerun()

                        except Exception as delete_error:

                            st.error(
                                f"Failed to delete document: {delete_error}"
                            )

except Exception as error:

    st.error(
        f"Document management error: {error}"
    )
# ==========================================
# 8. INITIALIZE RAG PIPELINE FOR THIS SESSION
# ==========================================

def get_rag_pipeline():

    if st.session_state.rag_pipeline is None:

        (
            _,
            _,
            retriever,
            query_rewriter,
            llm_generator
        ) = initialize_backend()

        conversation_memory = ConversationMemory()

        st.session_state.rag_pipeline = RAGPipeline(
            retriever=retriever,
            query_rewriter=query_rewriter,
            llm_generator=llm_generator,
            conversation_memory=conversation_memory
        )

    return st.session_state.rag_pipeline


# ==========================================
# 9. DISPLAY PREVIOUS CHAT MESSAGES
# ==========================================

st.divider()

st.subheader("💬 Ask Your Documents")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 View Sources"):

                    for index, source in enumerate(sources, start=1):

                        st.markdown(
                            f"**Source {index}:** "
                            f"{source['file_name']}"
                        )

                        st.write(
                            f"Page: {source['page']}"
                        )

                        st.write(
                            f"Similarity Score: {source['score']}"
                        )

                        st.divider()


# ==========================================
# 10. CHAT INPUT AND STREAMING RAG ANSWER
# ==========================================

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:

    if st.session_state.vector_store.get_collection_count() == 0:

        st.warning(
            "Please upload and process at least one PDF first."
        )

    else:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate RAG answer
        with st.chat_message("assistant"):

            try:

                rag = get_rag_pipeline()

                with st.spinner("Searching your documents..."):

                    answer = st.write_stream(
                        rag.ask_stream(
                            question=question,
                            top_k=top_k,
                            min_score=min_score
                        )
                    )

                # Read metadata after streaming completes
                metadata = rag.last_stream_metadata

                sources = metadata.get("sources", [])

                # Display source information
                if sources:

                    with st.expander("📚 View Sources"):

                        for index, source in enumerate(sources, start=1):

                            st.markdown(
                                f"**Source {index}:** "
                                f"{source['file_name']}"
                            )

                            st.write(
                                f"Page: {source['page']}"
                            )

                            st.write(
                                f"Similarity Score: {source['score']}"
                            )

                            st.divider()

                else:

                    st.info("No relevant document sources were found.")

                # Save answer and sources in session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(f"Something went wrong: {error}")
