"""
Example retriever -- RAG over verified Catala examples.

Embeds .catala_en files from the catala_examples/ directory using a local
embedding model via Ollama, stores them in a ChromaDB index, and retrieves
the most relevant examples given a legal text query.

Uses the OpenAI-compatible /v1/embeddings endpoint, the same pattern as
llm_pipeline.py, to avoid Ollama API version ambiguity.

The index is built once and reused. Rebuild it by calling build_index().

Usage:
    retriever = ExampleRetriever(
        examples_dir=Path("catala_examples/"),
        index_dir=Path("catala_pipeline/index/"),
        embed_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )
    retriever.build_index()
    examples = retriever.retrieve("payment deadline within 30 days", top_k=2)
    # examples: list of .catala_en file contents, most relevant first
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Override sqlite3 with pysqlite3-binary if available.
# Required on systems with sqlite3 < 3.35.0 (e.g. older university clusters).
try:
    import pysqlite3 as sqlite3_override
    import sys
    sys.modules["sqlite3"] = sqlite3_override
except ImportError:
    pass

import chromadb
from openai import OpenAI


# ---------------------------------------------------------------------------
# Embedding via Ollama OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def _embed(text: str, model: str, client: OpenAI) -> list[float]:
    """
    Get an embedding vector using the OpenAI-compatible /v1/embeddings endpoint.
    This is the same pattern used by llm_pipeline.py for LLM calls.
    """
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ExampleRetriever:
    """
    Retrieves relevant Catala examples for a given legal text query.

    Examples are loaded from .catala_en files in examples_dir.
    The index is persisted in index_dir and rebuilt only when needed.
    """

    def __init__(
        self,
        examples_dir: Path,
        index_dir: Path,
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.examples_dir = examples_dir
        self.index_dir = index_dir
        self.embed_model = embed_model

        # OpenAI-compatible client pointing at Ollama
        self._embed_client = OpenAI(
            api_key="ollama",
            base_url=f"{ollama_base_url}/v1",
        )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(self.index_dir))
        self._collection = self._chroma.get_or_create_collection(
            name="catala_examples",
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Embed all .catala_en files in examples_dir and store in the index.

        Skips files already indexed (by content hash) unless force_rebuild=True.
        Returns the number of files newly indexed.
        """
        if force_rebuild:
            self._chroma.delete_collection("catala_examples")
            self._collection = self._chroma.get_or_create_collection(
                name="catala_examples",
                metadata={"hnsw:space": "cosine"},
            )

        example_files = sorted(self.examples_dir.glob("*.catala_en"))
        if not example_files:
            raise FileNotFoundError(
                f"No .catala_en files found in {self.examples_dir}"
            )

        indexed = 0
        for path in example_files:
            content = path.read_text(encoding="utf-8")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            doc_id = path.stem

            # Skip if already indexed with same content
            try:
                existing = self._collection.get(ids=[doc_id])
                if (
                    existing["ids"]
                    and not force_rebuild
                    and existing["metadatas"][0].get("content_hash") == content_hash
                ):
                    continue
            except Exception:
                pass

            print(f"  Embedding {path.name}...")
            embedding = _embed(content, self.embed_model, self._embed_client)
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"filename": path.name, "content_hash": content_hash}],
            )
            indexed += 1

        return indexed

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """
        Retrieve the top_k most relevant Catala examples for the given query.

        Returns a list of .catala_en file contents, most relevant first.
        Returns empty list if the index is empty.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = _embed(query, self.embed_model, self._embed_client)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )
        return results["documents"][0]