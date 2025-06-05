"""Base runner interface for conversation simulation."""

from __future__ import annotations

import abc

from ..models.simulation import ConversationResult


class BaseRunner(abc.ABC):
    """Base interface for conversation simulation runners.
    
    This abstract base class defines the interface that all conversation
    runners must implement.
    """
    
    @abc.abstractmethod
    async def run(self) -> ConversationResult:
        """Execute the conversation simulation.
        
        Returns:
            ConversationResult with conversation history and metadata
        """
        ...
