"""Participant roles for conversation simulation."""

from __future__ import annotations

from enum import Enum


class ParticipantRole(str, Enum):
    """Role of a conversation participant."""
    
    CUSTOMER = "customer"
    AGENT = "agent"
