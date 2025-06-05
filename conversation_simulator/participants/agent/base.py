"""Agent participant base class."""

from __future__ import annotations

from ..base import BaseParticipant
from ...models.roles import ParticipantRole


class BaseAgent(BaseParticipant):
    """Base class for agent participants.
    
    Provides common functionality and interface for agent implementations.
    All agent participants should inherit from this class.
    """
    
    @property
    def role(self) -> ParticipantRole:
        """Agent role identifier."""
        return ParticipantRole.AGENT
