from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from database import MongoDBClient

class SessionMemoryManager:
    """
    Manages LangChain-compatible conversation memory across sessions.
    Uses ChatMessageHistory (langchain_community) and persists to MongoDB.
    """
    def __init__(self, mongo: MongoDBClient, plan_id: str):
        self.mongo     = mongo
        self.plan_id   = plan_id
        self.chat_memory = ChatMessageHistory()
        self._load()

    def _load(self):
        """Restore persisted messages into LangChain ChatMessageHistory."""
        messages = self.mongo.load_memory(self.plan_id)
        for msg in messages:
            if msg.get("role") == "user":
                self.chat_memory.add_user_message(msg["content"])
            elif msg.get("role") in ("assistant", "ai"):
                self.chat_memory.add_ai_message(msg["content"])

    def add_user_message(self, content: str):
        self.chat_memory.add_user_message(content)
        self.mongo.append_memory_message(self.plan_id, "user", content)

    def add_ai_message(self, content: str):
        self.chat_memory.add_ai_message(content)
        self.mongo.append_memory_message(self.plan_id, "assistant", content)

    def get_history(self) -> list:
        """Return messages as plain dicts for AgentState."""
        result = []
        for msg in self.chat_memory.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result

    def get_summary(self) -> str:
        msgs = self.chat_memory.messages
        if not msgs:
            return "No conversation history yet."
        lines = []
        for msg in msgs[-6:]:
            role = "Student" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content[:120]}{'...' if len(msg.content) > 120 else ''}")
        return "\n".join(lines)

