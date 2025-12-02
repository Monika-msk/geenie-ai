import os
import shutil
from datetime import datetime
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader



APP_TITLE = "⚡ Knowledge Base RAG Agent (LlamaIndex + Groq + BGE Embeddings)"
CHROMA_PERSIST_DIR = "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant over a private document collection. "
    "Use ONLY the provided context from the documents to answer. "
    "If the answer is not fully supported by the context, respond exactly with: "
    '"Information not found in uploaded documents." '
    "Do not use outside knowledge, do not guess, and do not change this fallback wording."
)




def load_env() -> None:
    """Load environment variables from .env if present."""

    load_dotenv()


def get_groq_api_key() -> str:
    """Return GROQ_API_KEY or raise a clear error for the user."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Please create a .env file with GROQ_API_KEY=... or set it in your environment."
        )
    return api_key


def init_llm() -> Groq:
    """Initialize the Groq LLM used for generation."""

    api_key = get_groq_api_key()
    llm = Groq(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
    )
    return llm


def init_embedding_model() -> HuggingFaceEmbedding:
    """Initialize the HuggingFace BGE embedding model."""

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    return embed_model


def init_chroma_vector_store(collection_name: str = "kb_collection") -> ChromaVectorStore:
    """Create a persistent Chroma vector store wrapped for LlamaIndex."""

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store


def clear_chroma_db() -> None:
    """Delete the local ChromaDB folder to reset the knowledge base."""

    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)


def pdf_files_to_documents(uploaded_files) -> list[Document]:
    """Convert uploaded PDF files into LlamaIndex Documents.

    Each page is stored as a separate document with metadata for filename and page number.
    """

    docs: list[Document] = []

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        num_pages = len(reader.pages)
        for page_idx in range(num_pages):
            page = reader.pages[page_idx]
            # pypdf text extraction – quality depends on the underlying PDF structure
            text = page.extract_text() or ""
            if not text.strip():
                # Skip purely empty pages
                continue

            metadata = {
                "file_name": uploaded_file.name,
                "page_number": page_idx + 1,
            }
            docs.append(Document(text=text, metadata=metadata))

    return docs


def build_or_update_index(documents: list[Document]) -> VectorStoreIndex:
    """Build a new index from documents and persist it to ChromaDB.

    For simplicity and determinism, this recreates the collection on each build.
    """

    vector_store = init_chroma_vector_store()
    # Recreate collection cleanly while keeping the on-disk DB folder
    chroma_collection = vector_store._collection
    client = chroma_collection._client  # type: ignore[attr-defined]
    name = chroma_collection.name
    client.delete_collection(name)
    fresh_collection = client.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=fresh_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use local HuggingFace BGE embeddings explicitly (no OpenAI required)
    embed_model = init_embedding_model()
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    Settings.embed_model = embed_model
    Settings.node_parser = splitter

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    return index


def load_index_from_chroma() -> VectorStoreIndex:
    """Load an index from an existing ChromaDB collection.

    If the collection is empty, this will create an index view over an empty store.
    """

    vector_store = init_chroma_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    return index


def get_chat_engine(index: VectorStoreIndex, llm: Groq) -> CondensePlusContextChatEngine:
    """Create a streaming chat engine over the index using the Groq LLM."""

    embed_model = init_embedding_model()

    Settings.llm = llm
    Settings.embed_model = embed_model

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        verbose=False,
        streaming=True,
        system_prompt=SYSTEM_PROMPT,
    )

    return chat_engine


def hash_uploaded_files(uploaded_files) -> str:
    """Create a simple hash based on file names and sizes to detect changes."""

    parts: list[str] = []
    for f in uploaded_files:
        size = getattr(f, "size", 0)
        parts.append(f"{f.name}:{size}")
    return "|".join(sorted(parts))


# ---------- Streamlit UI Helpers ----------

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {role, content, timestamp}
    if "index" not in st.session_state:
        st.session_state.index = None
    if "files_hash" not in st.session_state:
        st.session_state.files_hash = None


def add_message(role: str, content: str) -> None:
    """Append a message with timestamp to the in-memory history."""

    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def render_chat_history() -> None:
    """Render all previous messages with timestamps."""

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"*{msg['timestamp']}*")
            st.write(msg["content"])



def main() -> None:
    load_env()

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚡",
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption(
        "Upload PDFs, build a private knowledge base, and query it via a Groq-powered RAG agent. "
        "All embeddings are stored locally in ChromaDB (chroma_db/)."
    )

    init_session_state()

    with st.sidebar:
        st.header("Configuration")

        if st.button("Clear local ChromaDB (reset KB)", type="secondary"):
            clear_chroma_db()
            st.session_state.index = None
            st.session_state.files_hash = None
            st.session_state.messages = []
            st.success("chroma_db/ cleared. Upload documents again to rebuild the index.")

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            current_hash = hash_uploaded_files(uploaded_files)
            if st.session_state.files_hash and st.session_state.files_hash != current_hash:
                st.info(
                    "Detected change in uploaded files. The knowledge base will be rebuilt "
                    "with the new set of documents."
                )

            if st.button("Build / Rebuild Knowledge Base", type="primary"):
                with st.spinner("Reading PDFs and building index (this may take a moment)..."):
                    docs = pdf_files_to_documents(uploaded_files)
                    if not docs:
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        try:
                            index = build_or_update_index(docs)
                            st.session_state.index = index
                            st.session_state.files_hash = current_hash
                            st.success(
                                f"Indexed {len(docs)} document chunks from {len(uploaded_files)} PDF file(s)."
                            )
                        except Exception as e:
                            st.error(f"Failed to build index: {e}")

        st.markdown("---")
        st.subheader("Status")
        chroma_exists = os.path.exists(CHROMA_PERSIST_DIR)
        st.write(f"ChromaDB folder: {'✅ found' if chroma_exists else '❌ not found'} ({CHROMA_PERSIST_DIR})")
        if st.session_state.index is not None:
            st.write("Index status: ✅ loaded in memory")
        else:
            st.write("Index status: ⚠️ not loaded")

        st.markdown("---")
        st.subheader("Notes")
        st.markdown(
            "- Answers are **strictly grounded** in uploaded documents.\n"
            "- If information is missing, the assistant will answer: \n"
            "  `Information not found in uploaded documents.`"
        )

    # Main chat area
    st.subheader("Chat with your knowledge base")

    try:
        llm = init_llm()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # Load index from session or from existing ChromaDB store
    if st.session_state.index is None and os.path.exists(CHROMA_PERSIST_DIR):
        try:
            st.session_state.index = load_index_from_chroma()
        except Exception:
            # If anything goes wrong, ask user to rebuild
            st.warning(
                "Existing ChromaDB store found but could not be loaded as an index. "
                "Please rebuild the knowledge base from the sidebar."
            )

    index = st.session_state.index

    if index is None:
        st.info("Upload documents and build the knowledge base from the sidebar to start chatting.")
        return

    chat_engine = get_chat_engine(index, llm)

    render_chat_history()

    user_input = st.chat_input("Ask a question about your uploaded documents...")

    if user_input:
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(f"*{st.session_state.messages[-1]['timestamp']}*")
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            sources_container = st.container()

            try:
                response = chat_engine.stream_chat(user_input)

                accumulated_text = ""
                for token in response.response_gen:
                    accumulated_text += token
                    placeholder.markdown(accumulated_text)

                # After streaming completes, show sources
                source_nodes = getattr(response, "source_nodes", []) or []
                if source_nodes:
                    with sources_container.expander("Show referenced document snippets"):
                        for i, node in enumerate(source_nodes, start=1):
                            meta = node.node.metadata or {}
                            file_name = meta.get("file_name", "Unknown file")
                            page_number = meta.get("page_number", "?")
                            st.markdown(f"**Source {i}: {file_name} (page {page_number})**")
                            st.write(node.node.get_content()[:1000])
                            st.markdown("---")

                assistant_text = accumulated_text.strip()
                if not assistant_text:
                    assistant_text = "Information not found in uploaded documents."

                add_message("assistant", assistant_text)

            except Exception as e:
                error_msg = f"Error during chat: {e}"
                placeholder.error(error_msg)
                add_message("assistant", error_msg)


if __name__ == "__main__":
    main()
