"""Core RAG pipeline using LangChain, OpenRouter, and ChromaDB."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def _ensure_chroma_otel_safe() -> None:
    """Guard ChromaDB import when Windows App Control blocks optional gRPC OTel DLLs."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    module_name = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
    except ImportError:
        stub = types.ModuleType(module_name)
        stub.OTLPSpanExporter = object  # type: ignore[attr-defined]
        sys.modules[module_name] = stub


_ensure_chroma_otel_safe()

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = "http://localhost:8501"
APP_TITLE = "OpenRouter RAG Chatbot"

DEFAULT_PERSIST_DIR = Path("./chroma_db")
COLLECTION_NAME = "rag_documents"
META_FILENAME = "kb_meta.json"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".csv")
EMBED_BATCH_SIZE = 100


@dataclass
class IndexingResult:
    """Summary returned after indexing uploaded documents into ChromaDB."""

    files_processed: int
    raw_documents: int
    total_chunks: int
    skipped_files: list[str]
    sample_chunks: list[Document]


def _meta_path(persist_dir: Path) -> Path:
    return persist_dir / META_FILENAME


def load_kb_metadata(persist_dir: Path = DEFAULT_PERSIST_DIR) -> dict[str, Any]:
    """Load persisted knowledge-base metadata without requiring an API key."""
    path = _meta_path(persist_dir)
    if not path.exists():
        return {"embedding_model": None, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"embedding_model": None, "files": {}}
        data.setdefault("embedding_model", None)
        data.setdefault("files", {})
        return data
    except (OSError, json.JSONDecodeError):
        return {"embedding_model": None, "files": {}}


def save_kb_metadata(
    metadata: dict[str, Any], persist_dir: Path = DEFAULT_PERSIST_DIR
) -> None:
    """Persist knowledge-base metadata to disk."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(persist_dir).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def get_kb_stats(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    """Return current chunk count, indexed files, and active embedding model."""
    meta = load_kb_metadata(persist_dir)
    chunk_count = 0

    if (persist_dir / "chroma.sqlite3").exists():
        try:
            client = chromadb.PersistentClient(path=str(persist_dir))
            existing_names = {col.name for col in client.list_collections()}
            if collection_name in existing_names:
                collection = client.get_collection(name=collection_name)
                chunk_count = collection.count()
        except Exception:
            chunk_count = sum(
                info.get("chunks", 0) for info in meta.get("files", {}).values()
            )
    elif meta.get("files"):
        chunk_count = sum(
            info.get("chunks", 0) for info in meta.get("files", {}).values()
        )

    return {
        "chunk_count": chunk_count,
        "files": meta.get("files", {}),
        "embedding_model": meta.get("embedding_model"),
    }


def _load_single_file(file_path: Path, original_name: str) -> list[Document]:
    """Load one file into LangChain Documents and preserve original filename metadata."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), autodetect_encoding=True)
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix == ".csv":
        loader = CSVLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    cleaned: list[Document] = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        doc.metadata["source"] = original_name
        cleaned.append(doc)
    return cleaned


def load_uploaded_files(
    uploaded_files: Iterable[Any],
) -> tuple[list[Document], list[str]]:
    """Persist Streamlit UploadedFile objects temporarily and load them as Documents."""
    all_docs: list[Document] = []
    skipped_files: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, uploaded in enumerate(uploaded_files):
            original_name = Path(uploaded.name).name
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped_files.append(f"{original_name} (unsupported extension)")
                continue

            # Prefix index prevents collisions if two files share a name.
            temp_path = tmp_root / f"{idx}_{original_name}"
            temp_path.write_bytes(uploaded.getbuffer())

            loaded = _load_single_file(temp_path, original_name)
            if not loaded:
                skipped_files.append(f"{original_name} (no extractable text)")
                continue
            all_docs.extend(loaded)

    return all_docs, skipped_files


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding and retrieval."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    valid_chunks: list[Document] = []
    for i, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            continue
        chunk.metadata["chunk_index"] = i
        valid_chunks.append(chunk)
    return valid_chunks


def get_embeddings(api_key: str, embedding_model: str) -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings client routed through OpenRouter."""
    if not api_key.strip():
        raise ValueError("OpenRouter API key is required.")
    return OpenAIEmbeddings(
        model=embedding_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        check_embedding_ctx_length=False,
        default_headers={
            "HTTP-Referer": APP_URL,
            "X-Title": APP_TITLE,
        },
    )


def get_vector_store(
    api_key: str,
    embedding_model: str,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Open or create the persistent ChromaDB vector store."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(api_key=api_key, embedding_model=embedding_model)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def validate_embedding_model_compatibility(
    embedding_model: str,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Ensure the selected embedding model matches existing ChromaDB vectors."""
    stats = get_kb_stats(persist_dir=persist_dir, collection_name=collection_name)
    active_model = stats.get("embedding_model")
    if stats["chunk_count"] > 0 and active_model and active_model != embedding_model:
        raise ValueError(
            f"Existing ChromaDB vectors were created with '{active_model}'. "
            f"Switch back to '{active_model}' or click 'Reset ChromaDB' before using '{embedding_model}'."
        )


def index_uploaded_files(
    uploaded_files: list[Any],
    api_key: str,
    embedding_model: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    progress_callback: Callable[[str], None] | None = None,
) -> IndexingResult:
    """Run the full ingestion pipeline: load -> chunk -> embed -> store in ChromaDB."""
    if not uploaded_files:
        raise ValueError("Please upload at least one document before indexing.")

    validate_embedding_model_compatibility(
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    if progress_callback:
        progress_callback("Extracting text from uploaded documents...")
    raw_docs, skipped_files = load_uploaded_files(uploaded_files)
    if not raw_docs:
        skipped_note = "; ".join(skipped_files) if skipped_files else "No content found."
        raise ValueError(f"Could not extract any text from the uploaded files ({skipped_note}).")

    if progress_callback:
        progress_callback(
            f"Chunking {len(raw_docs)} document section(s) (size={chunk_size}, overlap={chunk_overlap})..."
        )
    chunks = chunk_documents(
        raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not chunks:
        raise ValueError("Document splitting produced 0 non-empty chunks.")

    if progress_callback:
        progress_callback(
            f"Generating embeddings via OpenRouter ({embedding_model}) and storing {len(chunks)} chunk(s) in ChromaDB..."
        )
    vector_store = get_vector_store(
        api_key=api_key,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        vector_store.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end],
        )

    # Record per-file chunk counts in metadata for the UI.
    meta = load_kb_metadata(persist_dir)
    meta["embedding_model"] = embedding_model
    file_counts: dict[str, int] = {}
    for chunk in chunks:
        src = str(chunk.metadata.get("source", "Unknown"))
        file_counts[src] = file_counts.get(src, 0) + 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for src, count in file_counts.items():
        prev_count = meta["files"].get(src, {}).get("chunks", 0)
        meta["files"][src] = {
            "chunks": prev_count + count,
            "indexed_at": timestamp,
        }
    save_kb_metadata(meta, persist_dir=persist_dir)

    return IndexingResult(
        files_processed=len(file_counts),
        raw_documents=len(raw_docs),
        total_chunks=len(chunks),
        skipped_files=skipped_files,
        sample_chunks=chunks[:3],
    )


def reset_knowledge_base(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Clear all documents from ChromaDB without removing open SQLite files on Windows."""
    if persist_dir.exists():
        client = chromadb.PersistentClient(path=str(persist_dir))
        existing_names = {col.name for col in client.list_collections()}
        if collection_name in existing_names:
            client.delete_collection(name=collection_name)
    save_kb_metadata({"embedding_model": None, "files": {}}, persist_dir=persist_dir)


def get_llm(
    api_key: str,
    chat_model: str,
    temperature: float = 0.2,
) -> ChatOpenRouter:
    """Instantiate LangChain's native ChatOpenRouter model."""
    if not api_key.strip():
        raise ValueError("OpenRouter API key is required.")
    return ChatOpenRouter(
        model=chat_model,
        api_key=api_key,
        temperature=temperature,
        app_url=APP_URL,
        app_title=APP_TITLE,
    )


def _to_langchain_history(
    messages: list[dict[str, Any]], max_turns: int = 6
) -> list[BaseMessage]:
    """Convert recent session chat messages into LangChain message objects."""
    recent = messages[-(max_turns * 2) :]
    history: list[BaseMessage] = []
    for msg in recent:
        content = msg.get("content", "")
        if not content:
            continue
        if msg.get("role") == "user":
            history.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            history.append(AIMessage(content=content))
    return history


def _extract_text(content: Any) -> str:
    """Normalize string or content-block outputs from ChatOpenRouter."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def rewrite_query_with_history(
    llm: ChatOpenRouter,
    question: str,
    chat_history: list[dict[str, Any]],
) -> str:
    """Condense a follow-up question into a standalone search query when history exists."""
    lc_history = _to_langchain_history(chat_history)
    if not lc_history:
        return question

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the conversation history and a follow-up question, rewrite the "
                "follow-up question into a concise standalone search query for retrieving "
                "relevant document chunks. Do not answer the question; return only the "
                "rewritten query.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"chat_history": lc_history, "question": question})
    rewritten = _extract_text(result.content).strip()
    return rewritten or question


def retrieve_context(
    vector_store: Chroma,
    query: str,
    top_k: int = 4,
) -> list[tuple[Document, float | None]]:
    """Retrieve top-k chunks and L2 distance scores from ChromaDB."""
    try:
        return vector_store.similarity_search_with_score(query, k=top_k)
    except Exception:
        docs = vector_store.similarity_search(query, k=top_k)
        return [(doc, None) for doc in docs]


def format_context(retrieved: list[tuple[Document, float | None]]) -> str:
    """Format retrieved chunks with numbered source headers for the LLM prompt."""
    blocks: list[str] = []
    for idx, (doc, _) in enumerate(retrieved, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(f"[{idx}] Source: {source}{page_label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def stream_rag_answer(
    llm: ChatOpenRouter,
    question: str,
    chat_history: list[dict[str, Any]],
    retrieved: list[tuple[Document, float | None]],
) -> Iterator[str]:
    """Stream a grounded answer from ChatOpenRouter given retrieved ChromaDB context."""
    context_text = format_context(retrieved)
    lc_history = _to_langchain_history(chat_history)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful document assistant. Answer the user's question using "
                "ONLY the retrieved document context below. Cite supporting sources inline "
                "using bracketed numbers like [1], [2]. If the context does not contain "
                "enough information to answer reliably, clearly say what is missing rather "
                "than guessing.\n\n"
                "Retrieved Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm

    for chunk in chain.stream(
        {
            "context": context_text,
            "chat_history": lc_history,
            "question": question,
        }
    ):
        text = _extract_text(chunk.content)
        if text:
            yield text
