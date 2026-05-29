"""Genesis Engine test fixtures."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from genesis.storage.database import Base


@pytest.fixture
def db_session() -> Session:
    """In-memory SQLite session for tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine)
    session = test_session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_domain_model() -> dict:
    """Sample ANALYZE stage output for testing."""
    return {
        "domain": "customer_support",
        "actors": ["customer", "support_agent", "billing_system"],
        "intents": [
            {
                "actor": "customer",
                "intent": "report_technical_issue",
                "priority": "high",
            },
            {
                "actor": "customer",
                "intent": "billing_question",
                "priority": "medium",
            },
        ],
        "constraints": ["verify_identity", "log_all_interactions"],
        "edge_cases": ["off_hours_contact", "language_barrier"],
        "success_criteria": [
            "requests_triaged_within_5_seconds",
            "resolution_rate_above_80_percent",
        ],
    }


@pytest.fixture
def sample_architecture() -> dict:
    """Sample ARCHITECT stage output for testing."""
    return {
        "topology": "router",
        "agents": [
            {
                "name": "triage_agent",
                "role": "Classify and route incoming requests",
                "triggers": ["new_request"],
                "tools": ["intent_classifier", "knowledge_base_search"],
            },
            {
                "name": "technical_agent",
                "role": "Resolve technical issues",
                "tools": ["knowledge_base_search", "ticket_system_api"],
                "escalates_to": "human_agent",
            },
            {
                "name": "billing_agent",
                "role": "Handle billing questions and refunds",
                "tools": ["billing_system_api", "refund_processor"],
                "escalates_to": "human_agent",
            },
        ],
        "routing": {"strategy": "intent_based", "confidence_threshold": 0.85},
    }
