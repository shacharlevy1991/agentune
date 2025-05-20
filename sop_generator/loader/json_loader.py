"""Base class for JSON-based chat loaders."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from .base import ChatLoader
from ..models import Conversation


class JsonChatLoader(ChatLoader):
    """Base class for JSON-based chat loaders."""
    
    def load(self, source: Path) -> List[Conversation]:
        """Load conversations from a JSON file.
        
        Args:
            source: Path to the JSON file
            
        Returns:
            List of parsed conversations
            
        Raises:
            ValueError: If the JSON is invalid
        """
        try:
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [self.parse_conversation(item) for item in data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {source}: {e}") from e
    
    @abstractmethod
    def parse_conversation(self, raw: Dict[str, Any]) -> Conversation:
        """Parse a single conversation from raw JSON data.
        
        Args:
            raw: Raw conversation data from JSON
            
        Returns:
            Parsed Conversation object
            
        Raises:
            ValueError: If required fields are missing
        """
        raise NotImplementedError("Subclasses must implement parse_conversation()")
