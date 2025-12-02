# ⚡ Knowledge Base RAG Agent (LlamaIndex + Groq + BGE Embeddings)

---

## 1. Overview of the Agent

This project is a private, document-grounded Retrieval-Augmented Generation (RAG) agent. You upload PDFs, the app indexes them into a local vector database (ChromaDB), and then you chat with a Groq-hosted LLM that answers **strictly from your documents**.

- **LLM**: Groq (Llama 3.1 via `llama-index-llms-groq`)
- **Embeddings**: `BAAI/bge-small-en-v1.5` via `llama-index-embeddings-huggingface`
- **RAG Framework**: LlamaIndex core
- **Vector DB**: ChromaDB (local persistent folder: `chroma_db/`)
- **UI**: Streamlit chat interface with history & streaming
- **PDF/Text**: `pypdf`-based PDF parsing

All embeddings and vector data are stored **locally**.

---

## 2. Folder Structure

```text
kb_rag_agent/
├─ app.py                # Main Streamlit application
├─ requirements.txt      # Python dependencies
├─ README.md             # This documentation
└─ chroma_db/            # (Auto-created) Local ChromaDB persistence
```

`chroma_db/` is created automatically on first run / first index build.

---

## 3. Features & Limitations

- **Multiple PDF uploads** from the sidebar
- **Automatic chunking, embedding, and indexing** into ChromaDB via LlamaIndex
- **Local persistent vector store** in `chroma_db/`
- **Groq-powered chat** using a Llama 3.1 model
- **Streaming responses** in the chat UI
- **Strict document grounding**
  - If the answer cannot be derived from the retrieved context, the assistant responds with:
    
    ```text
    Information not found in uploaded documents.
    ```
- **Chat history with timestamps**
- **Document snippet sources** displayed in an expandable section
- **Index rebuild detection** when the set of uploaded files changes (within the active session)

---

### 3.1 Limitations

- **No OCR** for scanned/handwritten PDFs (image-only pages will not yield text).
- **Retrieval quality depends on extracted text** and PDF structure.
- **No web browsing or external tools** – the agent is limited to uploaded documents.
- **Knowledge base refresh** currently requires clearing `chroma_db/`.
- **Session-scoped chat history** – not persisted across Streamlit restarts.

---

## 4. Tech Stack & APIs Used

- **Backend & RAG**
  - `llama-index-core` for index and query orchestration.
  - `llama-index-llms-groq` to talk to Groq-hosted LLMs.
  - `llama-index-embeddings-huggingface` with `BAAI/bge-small-en-v1.5` for dense embeddings.
  - `llama-index-vector-stores-chroma` to integrate ChromaDB as the vector store.
- **Vector Database**
  - `chromadb` (local persistent store under `chroma_db/`).
- **LLM Hosting**
  - Groq API (`GROQ_API_KEY`) – ultra-fast inference for Llama 3.1.
- **UI Layer**
  - `streamlit` for the web UI and chat interface.
- **PDF / Text Processing**
  - `pypdf` to extract text from PDFs.
  - `sentence-transformers` + `torch` as underlying HF stack.

---

## 5. Setup & Run Instructions (Run Locally)

- **Python** 3.10+ recommended
- **Virtual environment** (venv, Conda, etc.)
- A **Groq API key** from: https://console.groq.com/

---

From your project root (where `kb_rag_agent/` lives):

```bash
cd kb_rag_agent
```

### 5.1 Create and activate a virtual environment

```bash
# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# On macOS / Linux
# python -m venv .venv
# source .venv/bin/activate
```

### 5.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: Installing `torch` and `sentence-transformers` can take a bit of time, especially on first install.

### 5.3 Configure environment variables

Create a file named `.env` in the `kb_rag_agent/` folder with:

```env
GROQ_API_KEY="gsk_XXXXXXXXXXXXXXXXXXXX"
```

Alternatively, export `GROQ_API_KEY` directly in your shell environment.

---

### 5.4 Run the application locally

From inside the `kb_rag_agent/` folder with your virtual environment activated:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the app in your browser (typically at `http://localhost:8501`).

---

## 6. Architecture Diagram (End-to-End Flow)

High-level architecture of how the agent works from upload to answer:

```text
User Browser (Streamlit UI)
    |
    | 1. Upload PDFs / Ask question
    v
Streamlit app (app.py)
    |
    | 2. Parse PDFs -> LlamaIndex Documents (with metadata)
    v
PDF Parser (pypdf)
    |
    | 3. Chunk text (SentenceSplitter)
    v
LlamaIndex Core
    |
    | 4. Embed chunks (BAAI/bge-small-en-v1.5 via HF)
    v
HuggingFace Embeddings
    |
    | 5. Store / query vectors
    v
ChromaDB (local, chroma_db/)
    |
    | 6. Retrieve relevant chunks
    v
LlamaIndex Retriever / Chat Engine
    |
    | 7. Send context + question to Groq LLM
    v
Groq LLM (Llama 3.1)
    |
    | 8. Stream grounded answer back to UI
    v
User Browser (Chat with history + sources)
```

---

## 7. How It Works (Detailed)

### 7.1 Document Ingestion Flow

1. **Upload PDFs** via the sidebar file uploader.
2. Each uploaded PDF is processed page-by-page using `pypdf`.
3. Each page becomes a LlamaIndex `Document` with metadata:
   - `file_name`
   - `page_number`
4. Documents are **chunked** using LlamaIndex's `SentenceSplitter`.
5. Text chunks are embedded using **BAAI/bge-small-en-v1.5** via `HuggingFaceEmbedding`.
6. Embeddings are stored in a **ChromaDB** collection persisted under `chroma_db/`.

### 7.2 Query + RAG Flow

1. User asks a **natural language question** in the chat box.
2. LlamaIndex performs **vector search** over ChromaDB.
3. Retrieved chunks are passed as **context** to the Groq LLM.
4. Groq (Mixtral) generates a **streaming answer** in the UI.
5. The assistant follows a strict system prompt:
   - Uses **only retrieved context** from the documents.
   - If the answer is not fully supported, it replies:

     ```text
     Information not found in uploaded documents.
     ```

6. The chat history (with timestamps) is kept in the current Streamlit session.
7. Referenced document snippets (file + page) are displayed below each answer in an expandable section.

### 7.3 Index Rebuild Detection (Session-Scoped)

- Each time you upload files, a hash (name + size) is computed.
- If this hash changes compared to the previous upload set, the app:
  - Shows an **informational message** that the KB will be rebuilt.
  - On pressing **"Build / Rebuild Knowledge Base"**, the existing collection is cleared and rebuilt
    from the new documents.

> Note: This detection is **within the current Streamlit session**. If you restart the app, you'll
> need to upload and rebuild again.

---

## 8. UI Overview

- **Sidebar**
  - `Clear local ChromaDB (reset KB)`
  - PDF file uploader (multi-file)
  - `Build / Rebuild Knowledge Base` button
  - Status indicators for `chroma_db/` presence and in-memory index
  - Notes about grounding and fallback behavior

- **Main Area**
  - Chat interface powered by Streamlit's `st.chat_message` and `st.chat_input`
  - Timestamps per message
  - Streaming assistant responses
  - Expandable panel showing referenced document snippets with file name + page

---

## 9. Cleaning Up / Resetting

### 9.1 Clear local vector database from the UI

1. Open the app in your browser.
2. In the sidebar, click **"Clear local ChromaDB (reset KB)"**.
3. The app deletes `chroma_db/`, clears the in-memory index, and wipes the chat history.
4. Upload documents again and click **"Build / Rebuild Knowledge Base"** to create a fresh index.

### 9.2 Manually delete ChromaDB folder

From the `kb_rag_agent/` directory (with the app stopped):

```bash
# On Windows (PowerShell)
Remove-Item -Recurse -Force .\chroma_db

# On macOS / Linux (for reference)
# rm -rf ./chroma_db
```

---

## 10. Potential Improvements & Production Considerations

- **Secrets management**: Use environment variables or a secrets manager rather than committing `.env`.
- **GPU acceleration**: For heavy loads, consider running embeddings on GPU.
- **Concurrency**: Streamlit handles multiple sessions, but for high traffic you may want a separate
  backend API for indexing and querying.
- **Monitoring**: Add logging (e.g., `logging` module) for query traces and errors.
- **Authentication & multi-user isolation**: Add auth and per-user namespaces in ChromaDB.
- **Better file change tracking**: Persist a manifest of files + hashes alongside the DB to allow
  automatic index invalidation across restarts.
- **More controls in UI**: Expose top-k, temperature, and max-tokens as configurable sliders.
- **Additional file formats**: Extend to support DOCX, TXT, and HTML with appropriate readers.


