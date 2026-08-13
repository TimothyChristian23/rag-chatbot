"""Small in-process conversation memory for demo chat sessions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
DEFAULT_MAX_RETRIEVAL_HISTORY_TURNS = int(os.getenv("MAX_RETRIEVAL_HISTORY_TURNS", "3"))


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    """Store recent chat messages by session ID for a single FastAPI process."""

    def __init__(self, max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES):
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2.")
        self.max_messages = max_messages if max_messages % 2 == 0 else max_messages - 1
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of the current session history."""
        normalized_session_id = normalize_session_id(session_id)
        with self._lock:
            return list(self._sessions.get(normalized_session_id, []))

    def add_exchange(self, session_id: str, question: str, answer: str) -> list[ChatMessage]:
        """Append a user/assistant exchange and return the trimmed history."""
        normalized_session_id = normalize_session_id(session_id)
        messages = [
            ChatMessage(role="user", content=question.strip()),
            ChatMessage(role="assistant", content=answer.strip()),
        ]

        with self._lock:
            history = self._sessions.setdefault(normalized_session_id, [])
            history.extend(message for message in messages if message.content)
            del history[:-self.max_messages]
            return list(history)

    def clear_session(self, session_id: str) -> None:
        """Clear one session."""
        normalized_session_id = normalize_session_id(session_id)
        with self._lock:
            self._sessions.pop(normalized_session_id, None)

    def clear_all(self) -> None:
        """Clear every session."""
        with self._lock:
            self._sessions.clear()

    def session_count(self) -> int:
        """Return the number of sessions with stored history."""
        with self._lock:
            return len(self._sessions)


def normalize_session_id(session_id: str | None) -> str:
    """Normalize empty session IDs to the default demo session."""
    normalized = (session_id or "default").strip()
    return normalized or "default"


def build_retrieval_query(
    question: str,
    history: list[ChatMessage],
    max_turns: int = DEFAULT_MAX_RETRIEVAL_HISTORY_TURNS,
) -> str:
    """Include recent student questions so follow-ups retrieve the right chunks."""
    current_question = question.strip()
    previous_questions = [
        message.content.strip()
        for message in history
        if message.role == "user" and message.content.strip()
    ][-max_turns:]

    if not previous_questions:
        return current_question

    previous = "\n".join(f"- {item}" for item in previous_questions)
    return (
        "Recent student questions:\n"
        f"{previous}\n\n"
        "Current student question:\n"
        f"{current_question}"
    )
