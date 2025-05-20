"""Base classes for chat loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import Conversation


class ChatLoader(ABC):
    """Abstract base class for chat loaders."""
    
    @abstractmethod
    def load(self, source: Path) -> List[Conversation]:
        """Load conversations from a source.
        
        Args:
            source: Path to the source file
            
        Returns:
            List of loaded conversations
            
        Raises:
            NotImplementedError: If the loader doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement load()")
