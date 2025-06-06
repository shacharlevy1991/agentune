"""Base participant interface for conversation simulation."""

from __future__ import annotations

import abc

from ..models.conversation import Conversation
from ..models.message import Message


class Participant(abc.ABC):
    """Base interface for all simulated conversation participants.
    
    This abstract base class defines the interface that all conversation
    participants (customers and agents) must implement.
    """
    
    @abc.abstractmethod
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate the next message based on conversation history.
        
        Args:
            conversation: The conversation history up to this point
            
        Returns:
            Message with timestamp, or None if participant is finished
        """
        ...
