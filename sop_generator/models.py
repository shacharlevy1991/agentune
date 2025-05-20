"""Data models for chat-based SOP generation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class MessageRole(str, Enum):
    """Represents the role of a message sender."""
    CUSTOMER = "customer"
    AGENT = "agent"
    SYSTEM = "system"
    BOT = "bot"


@dataclass
class Message:
    """A single message in a conversation.
    
    Attributes:
        content: The text content of the message
        role: The role of the message sender
        timestamp: When the message was sent (optional)
        metadata: Additional metadata about the message
    """
    content: str
    role: MessageRole
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.role.upper()}] {self.content}"


@dataclass
class Conversation:
    """A complete conversation between participants.
    
    Attributes:
        id: Unique identifier for the conversation
        messages: List of messages in chronological order
        metadata: Additional metadata about the conversation
    """
    id: str
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def customer_messages(self) -> List[Message]:
        """Get all messages from the customer."""
        return [msg for msg in self.messages if msg.role == MessageRole.CUSTOMER]
    
    @property
    def agent_messages(self) -> List[Message]:
        """Get all messages from agents."""
        return [msg for msg in self.messages if msg.role == MessageRole.AGENT]
    
    def __str__(self) -> str:
        return f"Conversation {self.id} ({len(self.messages)} messages)"


@dataclass
class ConversationCluster:
    """A cluster of similar conversations.
    
    Attributes:
        id: Unique identifier for the cluster
        conversations: List of conversations in this cluster
        topic: Human-readable description of the cluster topic (optional)
        keywords: List of keywords that characterize this cluster
        metadata: Additional metadata about the cluster
    """
    id: str
    conversations: List[Conversation]
    topic: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Cluster {self.id}: {self.topic or 'Untitled'} ({len(self.conversations)} conversations)"


@dataclass
class SOP:
    """A Standard Operating Procedure generated from conversation clusters.
    
    Attributes:
        id: Unique identifier for the SOP
        title: Human-readable title of the SOP
        content: Full text content of the SOP
        steps: List of steps in the SOP, each with a title and description
        source_cluster_ids: IDs of clusters used to generate this SOP
        metadata: Additional metadata about the SOP
    """
    id: str
    title: str
    content: str
    steps: List[Dict[str, Any]]
    source_cluster_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"SOP: {self.title} ({len(self.steps)} steps)"
