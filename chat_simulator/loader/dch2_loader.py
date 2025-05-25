"""Loader for DCH2 processed chat format."""

from datetime import datetime
from typing import Dict, Any

from ..models import Conversation, Message, MessageRole
from .json_loader import JsonChatLoader

__all__ = ["DCH2JsonLoader"]


class DCH2JsonLoader(JsonChatLoader):
    """Loader for DCH2 processed JSON format.
    
    This loader handles the specific JSON format used in the DCH2 dataset.
    """
    
    def parse_conversation(self, raw: Dict[str, Any]) -> Conversation:
        """Parse a single conversation from DCH2 JSON format.
        
        Args:
            raw: Raw conversation data from JSON
            
        Returns:
            Parsed Conversation object
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            messages = [
                Message(
                    content=msg["text"],
                    role=self._map_role(msg["sender"]),
                    timestamp=datetime.fromisoformat(ts) if (ts := msg.get("timestamp")) else None,
                    metadata={
                        "original_data": msg,
                        "nugget_label": msg.get("consolidated_nugget_label")
                    }
                )
                for msg in raw["turns"]
            ]
            
            return Conversation(
                id=str(raw["dialogue_id"]),
                messages=messages,
                metadata={
                    "source": raw.get("source", "dch2_processed_train.json"),
                    "quality_scores": raw.get("consolidated_quality_scores", {})
                }
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required field in conversation data: {e}") from e
    
    def _map_role(self, sender: str) -> MessageRole:
        """Map raw sender string to MessageRole enum.
        
        Args:
            sender: Raw sender string from the data
            
        Returns:
            Corresponding MessageRole
            
        Raises:
            ValueError: If sender value is unknown
        """
        role_map = {
            "customer": MessageRole.CUSTOMER,
            "helpdesk": MessageRole.AGENT,
            "agent": MessageRole.AGENT,  # Support both 'helpdesk' and 'agent' as valid agent roles
            "system": MessageRole.SYSTEM,
            "bot": MessageRole.BOT
        }
        
        role = role_map.get(sender.lower())
        if role is None:
            raise ValueError(f"Unknown sender role: {sender}")
        return role
