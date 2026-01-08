import uuid
from typing import Dict, List
from datetime import datetime


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class Session:
    def __init__(self):
        self.session_id = uuid.uuid4().hex[:12]
        self.messages: List[Message] = []
        self.created_at = datetime.now().isoformat()

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role, content))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.messages],
        }


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session()


session_manager = SessionManager()
