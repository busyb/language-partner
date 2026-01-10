from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading


@dataclass
class SessionData:
    """Store per-user session state"""
    session_id: str
    created_at: datetime
    last_accessed: datetime

    # Language and conversation state
    current_lang_pair: str = "en-es"
    difficulty: str = "B1"
    counter: int = 0
    full_transcript: str = ""

    # Pronunciation state
    pronounce_counter: int = 0
    pronounce_lang: str = "en"
    pronunciation_history: list = field(default_factory=list)

    # Conversation state - stores ConversationState from PromptManager
    # Import: from utility.prompt_manager import ConversationState
    conversation_state: Optional[object] = None  # Will be ConversationState

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()

    def is_expired(self, ttl_hours: int = 2) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


class SessionManager:
    """Manage user sessions with automatic cleanup"""

    def __init__(self, ttl_hours: int = 2, cleanup_interval: int = 300):
        self.sessions: Dict[str, SessionData] = {}
        self.ttl_hours = ttl_hours
        self.cleanup_interval = cleanup_interval  # seconds
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def get_or_create_session(self, session_id: str) -> SessionData:
        """Get existing session or create new one"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_expired(self.ttl_hours):
                    # Expired session, create new one
                    self._cleanup_session(session_id)
                    session = self._create_session(session_id)
                else:
                    session.touch()
            else:
                session = self._create_session(session_id)

            return session

    def _create_session(self, session_id: str) -> SessionData:
        """Create a new session"""
        now = datetime.now()
        session = SessionData(
            session_id=session_id,
            created_at=now,
            last_accessed=now
        )
        self.sessions[session_id] = session
        print(f"✅ Created new session: {session_id}")
        return session

    def _cleanup_session(self, session_id: str):
        """Clean up a single session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"🗑️ Cleaned up session: {session_id}")

    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        with self._lock:
            expired = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(self.ttl_hours)
            ]
            for sid in expired:
                self._cleanup_session(sid)

            if expired:
                print(f"🗑️ Cleaned up {len(expired)} expired session(s)")

    def _start_cleanup_thread(self):
        """Start background thread for automatic cleanup"""

        def cleanup_loop():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup_expired_sessions()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self._lock:
            return len(self.sessions)

    def reset_session(self, session_id: str):
        """Reset a session to initial state while keeping the ID"""
        with self._lock:
            if session_id in self.sessions:
                self._cleanup_session(session_id)
                self._create_session(session_id)