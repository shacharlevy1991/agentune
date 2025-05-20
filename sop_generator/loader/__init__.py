"""Chat data loaders for different formats.

This package provides loaders for various chat data formats. The main components are:

- `ChatLoader`: Abstract base class for all loaders
- `JsonChatLoader`: Base class for JSON-based loaders
- `DCH2JsonLoader`: Implementation for DCH2 format
"""

from .base import ChatLoader
from .json_loader import JsonChatLoader
from .dch2_loader import DCH2JsonLoader

__all__ = ["ChatLoader", "JsonChatLoader", "DCH2JsonLoader"]
