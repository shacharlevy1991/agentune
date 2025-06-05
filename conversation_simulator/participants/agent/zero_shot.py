"""Zero-shot agent participant implementation."""

from __future__ import annotations

from .base import BaseAgent
from ...models.conversation import Conversation
from ...models.message import Message


class ZeroShotAgent(BaseAgent):
    """Zero-shot LLM-based agent participant.
    
    Uses a language model to generate agent responses without
    fine-tuning or few-shot examples.
    """
    
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Generate next agent message using zero-shot LLM approach.
        
        Args:
            conversation: Current conversation history
            
        Returns:
            Generated message or None if conversation should end
        """
        # TODO: Implement zero-shot agent logic
        # - Use LangChain LLM chain
        # - Generate realistic response timing
        raise NotImplementedError("ZeroShotAgent implementation pending")
