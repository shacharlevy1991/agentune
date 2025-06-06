"""Full simulation runner implementation."""

from __future__ import annotations

import attrs
from datetime import datetime
from typing import Any, Callable

from .base import Runner
from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.message import MessageDraft
from ..models.outcome import Outcomes
from ..models.simulation import ConversationResult
from ..participants.base import Participant


@attrs.define
class FullSimulationRunner(Runner):
    """Runs conversations with both simulated customer and agent.
    
    Single-use runner that manages conversation state internally.
    Provides progress tracking capabilities for conversations.
    """
    
    customer: Participant
    agent: Participant
    initial_message: MessageDraft
    intent: Intent
    outcomes: Outcomes
    max_messages: int = 100
    base_timestamp: datetime | None = None  # If None, use current time when run() starts
    progress_callback: Callable[[Conversation, dict[str, Any]], None] | None = None # todo: should be a class?
    
    # Private state - managed internally
    _conversation: Conversation = attrs.field(init=False)
    _is_complete: bool = attrs.field(init=False, default=False)
    _start_time: datetime | None = attrs.field(init=False, default=None)
    _current_timestamp: datetime = attrs.field(init=False)
    
    def __attrs_post_init__(self) -> None:
        """Initialize conversation with timestamped initial message."""
        # TODO: Complete initialization logic
        # - Set up timestamps
        # - Create initial conversation
        # - Initialize state
        pass
    
    async def run(self) -> ConversationResult:
        """Execute the full simulation conversation.
        
        Returns:
            ConversationResult with conversation history and metadata
        """
        # TODO: Implement full simulation logic
        # - Initialize timing
        # - Run alternating message loop
        # - Handle outcome detection
        # - Call progress callbacks
        # - Return results
        raise NotImplementedError("FullSimulationRunner implementation pending")
    
    @property
    def conversation(self) -> Conversation:
        """Get current conversation state (read-only access)."""
        # TODO: Return current conversation
        raise NotImplementedError("Property implementation pending")
    
    @property
    def is_complete(self) -> bool:
        """Check if the simulation has completed."""
        return self._is_complete
    
    def get_progress(self) -> dict[str, Any]:
        """Get current progress information."""
        # TODO: Implement progress tracking
        # - Message counts
        # - Timing information
        # - Outcome status
        raise NotImplementedError("Progress tracking implementation pending")
