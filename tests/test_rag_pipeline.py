"""Unit tests for the offline components of rag_pipeline.py."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag_pipeline import (
    chunk_documents,
    format_context,
    get_kb_stats,
    load_kb_metadata,
    load_uploaded_files,
    reset_knowledge_base,
    save_kb_metadata,
    validate_embedding_model_compatibility,
)


class _DummyUpload(io.BytesIO):
    """Minimal mock mimicking Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name


class RagPipelineTests(unittest.TestCase):
    def test_load_and_chunk_txt_and_csv(self) -> None:
        txt_file = _DummyUpload(
            "notes.txt",
            b"LangChain and ChromaDB power retrieval augmented generation.\n" * 15,
        )
        csv_file = _DummyUpload(
            "items.csv",
            b"name,role\nOpenRouter,LLM Gateway\nChromaDB,Vector Store\n",
        )
        bad_file = _DummyUpload("image.png", b"\x89PNG\r\n\x1a\n")

        docs, skipped = load_uploaded_files([txt_file, csv_file, bad_file])

        self.assertGreaterEqual(len(docs), 2)
        self.assertTrue(any("image.png" in item for item in skipped))
        self.assertTrue(
            all(doc.metadata["source"] in {"notes.txt", "items.csv"} for doc in docs)
        )

        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=40)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertEqual(
            [c.metadata["chunk_index"] for c in chunks], list(range(len(chunks)))
        )

    def test_chunk_overlap_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than chunk size"):
            chunk_documents([], chunk_size=200, chunk_overlap=200)

    def test_kb_metadata_and_model_compatibility(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            persist_dir = Path(tmp) / "chroma_db"
            save_kb_metadata(
                {
                    "embedding_model": "openai/text-embedding-3-small",
                    "files": {
                        "guide.pdf": {
                            "chunks": 12,
                            "indexed_at": "2026-11-26 11:00 UTC",
                        }
                    },
                },
                persist_dir=persist_dir,
            )
            loaded = load_kb_metadata(persist_dir=persist_dir)
            self.assertEqual(
                loaded["embedding_model"], "openai/text-embedding-3-small"
            )
            self.assertEqual(loaded["files"]["guide.pdf"]["chunks"], 12)

            stats = get_kb_stats(persist_dir=persist_dir)
            self.assertEqual(stats["chunk_count"], 12)

            validate_embedding_model_compatibility(
                "openai/text-embedding-3-small", persist_dir=persist_dir
            )

            with self.assertRaisesRegex(ValueError, "Reset ChromaDB"):
                validate_embedding_model_compatibility(
                    "openai/text-embedding-3-large", persist_dir=persist_dir
                )

            reset_knowledge_base(persist_dir=persist_dir)
            self.assertEqual(get_kb_stats(persist_dir=persist_dir)["chunk_count"], 0)

    def test_format_context_includes_source_and_page(self) -> None:
        docs = [
            (
                Document(
                    page_content="Fact A", metadata={"source": "doc.pdf", "page": 0}
                ),
                0.1,
            ),
            (Document(page_content="Fact B", metadata={"source": "notes.md"}), 0.2),
        ]
        formatted = format_context(docs)
        self.assertIn("[1] Source: doc.pdf, page 1\nFact A", formatted)
        self.assertIn("[2] Source: notes.md\nFact B", formatted)


if __name__ == "__main__":
    unittest.main()
