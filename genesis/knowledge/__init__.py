"""Persistent knowledge store — grounding memory for the factory.

Documents are persisted in SQLite (sharing the app's SQLAlchemy ``Base``) and
retrieved with a deterministic, dependency-free TF-IDF ranker. When an Azure
OpenAI embedding deployment is configured it can be used instead, but the
default path is fully offline and reproducible — no API keys, no sklearn.
"""

from genesis.knowledge.store import (
    KnowledgeStore,
    KnowledgeDocument,
    tokenize,
)

__all__ = ["KnowledgeStore", "KnowledgeDocument", "tokenize"]
