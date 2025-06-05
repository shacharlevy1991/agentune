"""Customer participant base class."""

from __future__ import annotations

from ..base import BaseParticipant
from ...models.roles import ParticipantRole


class BaseCustomer(BaseParticipant):
    """Base class for customer participants.
    
    Provides common functionality and interface for customer implementations.
    All customer participants should inherit from this class.
    """
    
    @property
    def role(self) -> ParticipantRole:
        """Customer role identifier."""
        return ParticipantRole.CUSTOMER
