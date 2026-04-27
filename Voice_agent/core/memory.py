import threading
from dataclasses import dataclass, field
from datetime import datetime
from config import MAX_HISTORY_TURNS

@dataclass
class Turn:
    role: str
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

class ConversationMemory:
    def __init__(self):
        self._turns = []
        self._lock = threading.Lock()

    def add_turn(self, role, text):
        with self._lock:
            self._turns.append(Turn(role=role, text=text))
            if len(self._turns) > MAX_HISTORY_TURNS * 2:
                self._turns = self._turns[-MAX_HISTORY_TURNS * 2:]

    def get_history(self):
        with self._lock:
            return list(self._turns)

    def clear(self):
        with self._lock:
            self._turns.clear()
