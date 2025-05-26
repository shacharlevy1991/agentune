"""Data models for chat-based SOP generation."""
import attrs
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Represents the role of a message sender."""
    CUSTOMER = "customer"
    AGENT = "agent"


@attrs.frozen
class Message:
    """A single message in a conversation.
    
    Attributes:
        content: The text content of the message
        role: The role of the message sender
        timestamp: When the message was sent (optional)
    """
    content: str
    role: MessageRole
    timestamp: datetime | None = None

    def __str__(self) -> str:
        return f"[{self.role.upper()}] {self.content}"


@attrs.frozen
class Conversation:
    """A complete conversation between participants.
    
    Attributes:
        id: Unique identifier for the conversation
        messages: Tuple of messages in chronological order
        outcome: Optional string-based conversation outcome (e.g., "resolved", "escalated")
        satisfaction: Optional integer (1-10) representing customer satisfaction
    """
    id: str
    messages: tuple[Message, ...]
    outcome: str | None = None
    satisfaction: int | None = None
   
    @property
    def customer_messages(self) -> tuple[Message, ...]:
        """Get all messages from the customer."""
        return tuple(msg for msg in self.messages if msg.role == MessageRole.CUSTOMER)
    
    @property
    def agent_messages(self) -> tuple[Message, ...]:
        """Get all messages from agents."""
        return tuple(msg for msg in self.messages if msg.role == MessageRole.AGENT)
    
    def __str__(self) -> str:
        total = len(self.messages)
        if total == 0:
            return f"Conversation {self.id} (0 messages): <empty>"

        if total <= 20:
            messages_str = "\n".join(str(msg) for msg in self.messages)
        else:
            first = "\n".join(str(msg) for msg in self.messages[:10])
            last = "\n".join(str(msg) for msg in self.messages[-10:])
            messages_str = f"{first}\n...\n{last}"

        return f"Conversation {self.id} ({total} messages):\n{messages_str}"
