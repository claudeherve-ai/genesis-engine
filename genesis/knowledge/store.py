"""SQLite-backed knowledge store with a deterministic TF-IDF retriever."""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from genesis.storage.database import KnowledgeDocumentRecord

# A compact English stop-word list. Kept inline so retrieval stays
# dependency-free and deterministic across environments.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its
    of on or that the their them then there these they this to was were what
    when where which who will with you your we our us can could should would
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case, split on non-alphanumerics, drop stop-words and 1-char tokens."""
    if not text:
        return []
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


@dataclass
class KnowledgeDocument:
    """A retrievable document with provenance."""

    id: str
    title: str
    text: str
    source: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_record(cls, record: KnowledgeDocumentRecord) -> "KnowledgeDocument":
        return cls(
            id=record.id,
            title=record.title or "",
            text=record.text,
            source=record.source,
            metadata=record.metadata_json or {},
            created_at=record.created_at,
        )


class KnowledgeStore:
    """Persistent document store with offline TF-IDF retrieval.

    Parameters
    ----------
    session:
        An active SQLAlchemy session bound to the shared engine.
    embeddings:
        Optional callable ``(List[str]) -> List[List[float]]`` for semantic
        retrieval. When ``None`` (default) a deterministic TF-IDF ranker is
        used. The ranker is fully offline and reproducible.
    """

    def __init__(self, session: Session, embeddings=None):
        self.session = session
        self._embeddings = embeddings

    # ------------------------------------------------------------------ write
    def ingest(
        self,
        title: str,
        text: str,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Persist a document and return its id. Empty text is rejected."""
        if not text or not text.strip():
            raise ValueError("Cannot ingest a document with empty text")
        doc_id = uuid.uuid4().hex
        record = KnowledgeDocumentRecord(
            id=doc_id,
            title=(title or "").strip(),
            text=text.strip(),
            source=source,
            metadata_json=metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(record)
        self.session.commit()
        return doc_id

    def clear(self) -> int:
        """Delete all documents. Returns the number removed."""
        count = self.session.query(KnowledgeDocumentRecord).delete()
        self.session.commit()
        return count

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return self.session.query(KnowledgeDocumentRecord).count()

    def all_documents(self) -> List[KnowledgeDocument]:
        records = self.session.query(KnowledgeDocumentRecord).all()
        return [KnowledgeDocument.from_record(r) for r in records]

    def search(
        self, query: str, k: int = 5
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Return up to ``k`` documents ranked by relevance to ``query``.

        Deterministic TF-IDF cosine similarity. Ties break by score
        (descending) then document id (ascending). An empty or stop-word-only
        query returns an empty list. Documents with zero overlap are excluded.
        """
        query_tokens = tokenize(query)
        if not query_tokens or k <= 0:
            return []

        records = self.session.query(KnowledgeDocumentRecord).all()
        if not records:
            return []

        docs = [KnowledgeDocument.from_record(r) for r in records]

        if self._embeddings is not None:
            try:
                return self._embedding_search(docs, query, k)
            except Exception:
                # Fall back to TF-IDF rather than fail closed.
                pass

        return self._tfidf_search(docs, query_tokens, k)

    # --------------------------------------------------------------- internals
    def _tfidf_search(
        self,
        docs: List[KnowledgeDocument],
        query_tokens: List[str],
        k: int,
    ) -> List[Tuple[KnowledgeDocument, float]]:
        n = len(docs)
        doc_tokens = [tokenize(f"{d.title} {d.text}") for d in docs]

        # Document frequency across the corpus.
        df: Counter = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1

        # Smoothed IDF so a single-document corpus still produces signal.
        def idf(term: str) -> float:
            return math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0

        def tfidf_vector(tokens: List[str]) -> Dict[str, float]:
            if not tokens:
                return {}
            counts = Counter(tokens)
            length = len(tokens)
            return {t: (c / length) * idf(t) for t, c in counts.items()}

        query_vec = tfidf_vector(query_tokens)
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if query_norm == 0:
            return []

        scored: List[Tuple[KnowledgeDocument, float]] = []
        for doc, tokens in zip(docs, doc_tokens):
            doc_vec = tfidf_vector(tokens)
            if not doc_vec:
                continue
            dot = sum(query_vec.get(t, 0.0) * v for t, v in doc_vec.items())
            if dot <= 0:
                continue
            doc_norm = math.sqrt(sum(v * v for v in doc_vec.values()))
            if doc_norm == 0:
                continue
            scored.append((doc, dot / (query_norm * doc_norm)))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]

    def _embedding_search(
        self,
        docs: List[KnowledgeDocument],
        query: str,
        k: int,
    ) -> List[Tuple[KnowledgeDocument, float]]:
        texts = [f"{d.title} {d.text}" for d in docs]
        vectors = self._embeddings(texts)
        query_vec = self._embeddings([query])[0]

        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = [
            (doc, cosine(query_vec, vec)) for doc, vec in zip(docs, vectors)
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]
