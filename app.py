"""Streamlit RAG Chatbot powered by LangChain, OpenRouter, and ChromaDB."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    DEFAULT_PERSIST_DIR,
    SUPPORTED_EXTENSIONS,
    get_kb_stats,
    get_llm,
    get_vector_store,
    index_uploaded_files,
    reset_knowledge_base,
    retrieve_context,
    rewrite_query_with_history,
    stream_rag_answer,
    validate_embedding_model_compatibility,
)

load_dotenv()

CHAT_MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "Custom...",
]

EMBEDDING_MODEL_OPTIONS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "qwen/qwen3-embedding-8b",
    "Custom...",
]


def _init_session_state(kb_stats: dict[str, Any]) -> None:
    """Initialize Streamlit session state keys once per session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = os.getenv("OPENROUTER_API_KEY", "")
    if "upload_expanded" not in st.session_state:
        st.session_state.upload_expanded = kb_stats["chunk_count"] == 0
    if "last_index_result" not in st.session_state:
        st.session_state.last_index_result = None


def _serialize_sources(
    retrieved: list[tuple[Any, float | None]],
) -> list[dict[str, Any]]:
    """Convert retrieved LangChain Documents into lightweight dicts for session state."""
    sources: list[dict[str, Any]] = []
    for idx, (doc, score) in enumerate(retrieved, start=1):
        page = doc.metadata.get("page")
        sources.append(
            {
                "rank": idx,
                "source": doc.metadata.get("source", "Unknown"),
                "page": page + 1 if isinstance(page, int) else None,
                "score": score,
                "content": doc.page_content,
            }
        )
    return sources


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved source chunks in a collapsible section under an answer."""
    if not sources:
        return
    with st.expander(f"📚 Sources Used ({len(sources)} chunks)", expanded=False):
        for item in sources:
            page_info = f" • Page {item['page']}" if item.get("page") else ""
            score_info = (
                f" • Distance: {item['score']:.3f}"
                if item.get("score") is not None
                else ""
            )
            st.markdown(
                f"**[{item['rank']}] {item['source']}**{page_info}{score_info}"
            )
            st.caption(item["content"])
            if item["rank"] < len(sources):
                st.divider()


def _render_sidebar(kb_stats: dict[str, Any]) -> dict[str, Any]:
    """Render the left sidebar for OpenRouter API key, models, and RAG controls."""
    with st.sidebar:
        st.header("🔑 OpenRouter Configuration")
        api_key = st.text_input(
            "OpenRouter API Key",
            key="api_key_input",
            type="password",
            placeholder="sk-or-v1-...",
            help="Enter your OpenRouter API key. It stays in your current session.",
        ).strip()

        if api_key:
            st.success("API key provided", icon="✅")
        else:
            st.warning("Enter your OpenRouter API key to upload or chat.")
        st.caption("Get a key at [openrouter.ai/keys](https://openrouter.ai/keys)")

        st.divider()
        st.subheader("🤖 Models")
        selected_chat = st.selectbox("Chat Model", CHAT_MODEL_OPTIONS, index=0)
        if selected_chat == "Custom...":
            chat_model = st.text_input(
                "Custom Chat Model ID",
                value="openai/gpt-4o-mini",
                help="Any OpenRouter model slug, e.g. mistralai/mistral-large",
            ).strip()
        else:
            chat_model = selected_chat

        selected_embed = st.selectbox(
            "Embedding Model", EMBEDDING_MODEL_OPTIONS, index=0
        )
        if selected_embed == "Custom...":
            embedding_model = st.text_input(
                "Custom Embedding Model ID",
                value="openai/text-embedding-3-small",
                help="Any OpenRouter embedding model slug",
            ).strip()
        else:
            embedding_model = selected_embed

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )

        st.divider()
        st.subheader("⚙️ Chunking & Retrieval")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=3000,
            value=1000,
            step=100,
        )
        max_overlap = max(0, chunk_size - 50)
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=min(500, max_overlap),
            value=min(200, max_overlap),
            step=50,
        )
        top_k = st.slider(
            "Retrieved Chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
        )

        st.divider()
        st.subheader("🗄️ ChromaDB Knowledge Base")
        col1, col2 = st.columns(2)
        col1.metric("Files", len(kb_stats["files"]))
        col2.metric("Chunks", kb_stats["chunk_count"])

        if kb_stats.get("embedding_model"):
            st.caption(f"Indexed with: `{kb_stats['embedding_model']}`")

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if btn_col2.button("Reset ChromaDB", use_container_width=True):
            reset_knowledge_base(persist_dir=DEFAULT_PERSIST_DIR)
            st.session_state.last_index_result = None
            st.session_state.upload_expanded = True
            st.toast("ChromaDB knowledge base cleared.", icon="🗑️")
            st.rerun()

    return {
        "api_key": api_key,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


def _render_upload_section(config: dict[str, Any], kb_stats: dict[str, Any]) -> None:
    """Render Section 1: Document Upload, Chunking, Embeddings, and ChromaDB indexing."""
    with st.expander(
        "📄 1. Document Upload & Processing (Chunking • Embeddings • ChromaDB)",
        expanded=st.session_state.upload_expanded,
    ):
        allowed_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=allowed_types,
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, MD, DOCX, CSV",
        )

        if uploaded_files:
            file_summary = ", ".join(
                f"{f.name} ({f.size / 1024:.1f} KB)" for f in uploaded_files
            )
            st.caption(f"Selected ({len(uploaded_files)}): {file_summary}")

        process_clicked = st.button(
            "⚙️ Process & Index into ChromaDB",
            type="primary",
            disabled=not uploaded_files,
        )

        if process_clicked:
            if not config["api_key"]:
                st.error(
                    "Please enter your OpenRouter API key in the left sidebar first."
                )
            elif not config["embedding_model"]:
                st.error("Please enter a valid embedding model ID in the left sidebar.")
            else:
                with st.status(
                    "Processing documents into ChromaDB...", expanded=True
                ) as status:
                    try:
                        def _update_status(msg: str) -> None:
                            st.write(msg)

                        result = index_uploaded_files(
                            uploaded_files=uploaded_files,
                            api_key=config["api_key"],
                            embedding_model=config["embedding_model"],
                            chunk_size=config["chunk_size"],
                            chunk_overlap=config["chunk_overlap"],
                            persist_dir=DEFAULT_PERSIST_DIR,
                            progress_callback=_update_status,
                        )
                        st.session_state.last_index_result = result
                        st.session_state.upload_expanded = False
                        status.update(
                            label=(
                                f"Indexed {result.total_chunks} chunks from "
                                f"{result.files_processed} file(s) into ChromaDB!"
                            ),
                            state="complete",
                            expanded=False,
                        )
                        st.rerun()
                    except Exception as exc:
                        status.update(label="Indexing failed", state="error")
                        st.error(f"Error during indexing: {exc}")

        last_result = st.session_state.last_index_result
        if last_result is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Files Processed", last_result.files_processed)
            m2.metric("Extracted Sections", last_result.raw_documents)
            m3.metric("Chunks Stored", last_result.total_chunks)

            if last_result.skipped_files:
                st.warning(
                    "Skipped files: " + "; ".join(last_result.skipped_files)
                )

            if last_result.sample_chunks:
                with st.expander("🔍 Preview Sample Chunks", expanded=False):
                    for idx, chunk in enumerate(last_result.sample_chunks, start=1):
                        src = chunk.metadata.get("source", "Unknown")
                        st.markdown(f"**Chunk {idx}** (`{src}`)")
                        st.code(chunk.page_content[:500], language="text")

        if kb_stats["files"]:
            st.markdown("**Indexed Documents in ChromaDB**")
            table_lines = [
                "| File | Chunks | Last Indexed |",
                "| :--- | :---: | :--- |",
            ]
            for name, info in kb_stats["files"].items():
                safe_name = str(name).replace("|", "\\|")
                chunks_count = info.get("chunks", 0)
                indexed_at = info.get("indexed_at", "")
                table_lines.append(f"| `{safe_name}` | {chunks_count} | {indexed_at} |")
            st.markdown("\n".join(table_lines))


def _render_chat_section(config: dict[str, Any], kb_stats: dict[str, Any]) -> None:
    """Render Section 2: Interactive RAG Q&A Chat with ChromaDB and ChatOpenRouter."""
    st.subheader("💬 2. Ask Questions")

    if not st.session_state.messages:
        if kb_stats["chunk_count"] > 0:
            st.info(
                f"Knowledge base is ready with **{kb_stats['chunk_count']}** chunk(s) "
                f"across **{len(kb_stats['files'])}** file(s). Ask a question below!"
            )
        else:
            st.info(
                "Upload and process your documents in **Section 1** above to populate "
                "ChromaDB, then ask your questions here."
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        if not config["api_key"]:
            st.error(
                "Please enter your OpenRouter API key in the left sidebar to ask questions."
            )
            return
        if kb_stats["chunk_count"] == 0:
            st.warning(
                "Your ChromaDB knowledge base is empty. Upload and index at least one document first."
            )
            return
        if not config["chat_model"] or not config["embedding_model"]:
            st.error("Please select both a chat model and an embedding model in the sidebar.")
            return

        try:
            validate_embedding_model_compatibility(
                embedding_model=config["embedding_model"],
                persist_dir=DEFAULT_PERSIST_DIR,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        prior_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                llm = get_llm(
                    api_key=config["api_key"],
                    chat_model=config["chat_model"],
                    temperature=config["temperature"],
                )
                vector_store = get_vector_store(
                    api_key=config["api_key"],
                    embedding_model=config["embedding_model"],
                    persist_dir=DEFAULT_PERSIST_DIR,
                )

                with st.spinner("Searching ChromaDB for relevant context..."):
                    search_query = rewrite_query_with_history(
                        llm=llm,
                        question=prompt,
                        chat_history=prior_history,
                    )
                    retrieved = retrieve_context(
                        vector_store=vector_store,
                        query=search_query,
                        top_k=config["top_k"],
                    )

                if not retrieved:
                    answer = (
                        "I could not find any matching passages in ChromaDB for that question."
                    )
                    st.markdown(answer)
                    sources: list[dict[str, Any]] = []
                else:
                    stream_box = st.empty()
                    answer = ""
                    for token in stream_rag_answer(
                        llm=llm,
                        question=prompt,
                        chat_history=prior_history,
                        retrieved=retrieved,
                    ):
                        answer += token
                        stream_box.markdown(answer + "▌")
                    stream_box.markdown(answer)
                    sources = _serialize_sources(retrieved)
                    _render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                st.error(f"Failed to generate answer: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot • OpenRouter & ChromaDB",
        page_icon="🤖",
        layout="wide",
    )

    kb_stats = get_kb_stats(persist_dir=DEFAULT_PERSIST_DIR)
    _init_session_state(kb_stats)

    config = _render_sidebar(kb_stats)

    st.title("🤖 Document RAG Chatbot")
    st.caption(
        "Powered by **Streamlit**, **LangChain (`ChatOpenRouter`)**, "
        "**OpenRouter Embeddings**, and **ChromaDB**."
    )

    _render_upload_section(config, kb_stats)
    st.divider()
    _render_chat_section(config, kb_stats)


if __name__ == "__main__":
    main()
