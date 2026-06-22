"""In-memory conversation session store."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SESSION_TTL = timedelta(hours=1)


@dataclass
class Turn:
    query: str
    titles: list[str]


@dataclass
class ConversationSession:
    turns: list[Turn] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, query: str, titles: list[str]) -> None:
        self.turns.append(Turn(query=query, titles=titles))
        self.last_active = datetime.utcnow()

    def is_expired(self) -> bool:
        return datetime.utcnow() - self.last_active > SESSION_TTL


_sessions: dict[str, ConversationSession] = {}


def get_or_create(session_id: str | None) -> tuple[str, ConversationSession]:
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        if not session.is_expired():
            return session_id, session
    new_id = str(uuid.uuid4())
    _sessions[new_id] = ConversationSession()
    return new_id, _sessions[new_id]


def format_history(session: ConversationSession) -> str:
    if not session.turns:
        return ""
    lines = ["Prior conversation:"]
    for i, turn in enumerate(session.turns, 1):
        titles = ", ".join(turn.titles[:5])
        lines.append(f'Turn {i} — User asked: "{turn.query}"')
        lines.append(f"  Recommended: {titles}")
    return "\n".join(lines)
