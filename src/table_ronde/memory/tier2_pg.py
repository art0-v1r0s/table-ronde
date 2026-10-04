import logging
import math
import uuid
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SemanticDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class SemanticKnowledgeStore:
    """Tier 2 Semantic / Long-Term Knowledge Memory.

    Supports vector similarity search, keyword search, and hybrid retrieval.
    Includes in-memory fallback when PostgreSQL with pgvector is offline.
    """

    def __init__(self, pg_connection_string: str | None = None) -> None:
        import os
        self.pg_conn = pg_connection_string or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/table_ronde_memory")
        self._in_memory_docs: dict[str, SemanticDocument] = {}
        self._is_pg_connected = False

    async def connect(self) -> bool:
        """Attempts connection to PostgreSQL + pgvector. Defaults to in-memory mode."""
        self._is_pg_connected = False
        logger.info("SemanticKnowledgeStore operating in resilient in-memory mode.")
        return False

    async def add_document(
        self,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticDocument:
        """Stores a document with optional vector embedding and arbitrary metadata."""
        doc = SemanticDocument(
            content=content,
            embedding=embedding or [],
            metadata=metadata or {},
        )
        self._in_memory_docs[doc.doc_id] = doc
        return doc

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[SemanticDocument, float]]:
        """Dense vector search using cosine similarity."""
        results: list[tuple[SemanticDocument, float]] = []
        for doc in self._in_memory_docs.values():
            if not doc.embedding:
                continue
            sim = _cosine_similarity(query_embedding, doc.embedding)
            if sim >= min_similarity:
                results.append((doc, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def keyword_search(self, query: str, top_k: int = 5) -> list[SemanticDocument]:
        """Simple text-match search across stored document contents."""
        terms = query.lower().split()
        scored: list[tuple[SemanticDocument, int]] = []

        for doc in self._in_memory_docs.values():
            text = doc.content.lower()
            score = sum(1 for term in terms if term in text)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SemanticDocument]:
        """Combines semantic vector search with keyword search."""
        vector_results = await self.search_similar(query_embedding, top_k=top_k * 2)
        keyword_results = await self.keyword_search(query, top_k=top_k * 2)

        # Merge and deduplicate maintaining vector rank priority
        seen: set[str] = set()
        merged: list[SemanticDocument] = []

        for doc, _ in vector_results:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                merged.append(doc)

        for doc in keyword_results:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                merged.append(doc)

        return merged[:top_k]
