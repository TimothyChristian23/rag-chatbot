"""Tests for session conversation memory."""
from src.conversation.memory import (
    ChatMessage,
    ConversationMemory,
    build_retrieval_query,
    normalize_session_id,
)


def test_memory_stores_and_trims_recent_messages():
    memory = ConversationMemory(max_messages=4)

    memory.add_exchange("student", "First question", "First answer")
    memory.add_exchange("student", "Second question", "Second answer")
    history = memory.add_exchange("student", "Third question", "Third answer")

    assert [message.content for message in history] == [
        "Second question",
        "Second answer",
        "Third question",
        "Third answer",
    ]


def test_memory_clears_one_session():
    memory = ConversationMemory()
    memory.add_exchange("student", "What is OPT?", "Answer")

    memory.clear_session("student")

    assert memory.get_history("student") == []


def test_normalize_session_id_uses_default_for_blank_values():
    assert normalize_session_id(None) == "default"
    assert normalize_session_id("   ") == "default"
    assert normalize_session_id("  demo  ") == "demo"


def test_build_retrieval_query_includes_recent_user_questions_only():
    history = [
        ChatMessage(role="user", content="What is STEM OPT?"),
        ChatMessage(role="assistant", content="STEM OPT answer"),
        ChatMessage(role="user", content="What about unemployment days?"),
    ]

    query = build_retrieval_query("Does that affect my EAD?", history, max_turns=1)

    assert "What about unemployment days?" in query
    assert "Does that affect my EAD?" in query
    assert "STEM OPT answer" not in query
    assert "What is STEM OPT?" not in query
