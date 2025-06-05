"""Conversation Simulator - A library for simulating customer-agent conversations."""

from __future__ import annotations

# Public API exports
from .models.conversation import Conversation
from .models.roles import ParticipantRole
from .models.intent import Intent
from .models.message import Message, MessageDraft
from .models.outcome import Outcome, Outcomes
from .models.simulation import ConversationResult
from .participants.base import BaseParticipant
from .runners.base import BaseRunner
from .runners.full_simulation import FullSimulationRunner

__version__ = "0.1.0"

__all__ = [
    # Models
    "Conversation",
    "Message", 
    "MessageDraft",
    "Intent",
    "Outcome",
    "Outcomes", 
    "ConversationResult",
    "ParticipantRole",
    # Base classes
    "BaseParticipant",
    "BaseRunner",
    # Runners
    "FullSimulationRunner",
]
