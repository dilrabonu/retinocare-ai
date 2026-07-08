"""Unit tests for the RAG agent -- including the non-negotiable disclaimer check.

This test file must never be deleted or weakened: test_agent_response_always_includes_disclaimer
is the single most important safety guarantee in this codebase.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.retinocare.agents.prompts import MANDATORY_DISCLAIMER
from src.retinocare.agents.rag_agent import RAGAgent

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-key-not-real")


@pytest.fixture
def fake_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"text": "Severe DR requires referral within 1 month.", "source": "referral-criteria.md", "score": 0.9}
    ]
    return retriever


def _mock_anthropic_response(text: str):
    mock_block = MagicMock(type="text", text=text)
    return MagicMock(content=[mock_block])


def test_agent_response_always_includes_disclaimer(fake_retriever):
    """Even if the LLM forgets the disclaimer, the agent must append it."""
    with patch("src.retinocare.agents.rag_agent.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _mock_anthropic_response(
            "Severe DR means extensive blocked vessels in the retina."
        )
        agent = RAGAgent("knowledge_base/guidelines", retriever=fake_retriever)
        result = agent.respond("Severe", 0.87)

        assert MANDATORY_DISCLAIMER in result["text"]
        assert result["disclaimer_included"] is True


def test_agent_does_not_duplicate_disclaimer_if_already_present(fake_retriever):
    with patch("src.retinocare.agents.rag_agent.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _mock_anthropic_response(
            f"Severe DR means extensive blocked vessels. {MANDATORY_DISCLAIMER}"
        )
        agent = RAGAgent("knowledge_base/guidelines", retriever=fake_retriever)
        result = agent.respond("Severe", 0.87)

        assert result["text"].count(MANDATORY_DISCLAIMER) == 1


def test_agent_cites_retrieved_sources(fake_retriever):
    with patch("src.retinocare.agents.rag_agent.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = _mock_anthropic_response(
            "Severe DR means extensive blocked vessels."
        )
        agent = RAGAgent("knowledge_base/guidelines", retriever=fake_retriever)
        result = agent.respond("Severe", 0.87)

        assert "referral-criteria.md" in result["sources"]