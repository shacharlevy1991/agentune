"""Tests for the DCH2 chat loader."""

import json
from datetime import datetime

import pytest

from chat_simulator.loader.dch2_loader import DCH2JsonLoader
from chat_simulator.models import Conversation, MessageRole


class TestDCH2JsonLoader:
    """Tests for the DCH2JsonLoader class."""

    def test_parse_conversation_valid(self, sample_chats_data):
        """Test parsing a valid DCH2 conversation."""
        test_data = sample_chats_data[0]  # First conversation
        loader = DCH2JsonLoader()
        result = loader.parse_conversation(test_data)
        
        assert isinstance(result, Conversation)
        assert result.id == "conv_123"
        assert len(result.messages) == 4
        assert result.metadata["source"] == "test_fixture"
        
        # Check first message
        assert result.messages[0].content == "Hello, I need help with my order"
        assert result.messages[0].role == MessageRole.CUSTOMER
        # Check timestamp is timezone-aware and has the correct value
        assert result.messages[0].timestamp is not None
        assert result.messages[0].timestamp.replace(tzinfo=None) == datetime(2025, 5, 21, 10, 0, 0)
        
        # Check second message
        assert result.messages[1].role == MessageRole.AGENT
        assert "order number" in result.messages[1].content.lower()

    def test_parse_conversation_missing_required_field(self, sample_chats_data):
        """Test parsing a conversation with missing required fields."""
        test_data = sample_chats_data[0].copy()
        del test_data["dialogue_id"]
        loader = DCH2JsonLoader()
        
        with pytest.raises(ValueError, match="dialogue_id"):
            loader.parse_conversation(test_data)

    def test_parse_conversation_empty_turns(self, sample_chats_data):
        """Test parsing a conversation with no messages."""
        test_data = sample_chats_data[0].copy()
        test_data["turns"] = []
        loader = DCH2JsonLoader()
        
        result = loader.parse_conversation(test_data)
        assert len(result.messages) == 0

    def test_load_from_file(self, tmp_path, sample_chats_data):
        """Test loading conversations from a file."""
        test_file = tmp_path / "test_chats.json"
        test_file.write_text(json.dumps(sample_chats_data))
        loader = DCH2JsonLoader()
        
        result = loader.load(test_file)
        
        assert len(result) == 2
        assert result[0].id == "conv_123"
        assert result[1].id == "conv_124"
        assert len(result[0].messages) == 4
        assert len(result[1].messages) == 2

    def test_map_role(self):
        """Test the role mapping functionality."""
        loader = DCH2JsonLoader()
        assert loader._map_role("customer") == MessageRole.CUSTOMER
        assert loader._map_role("agent") == MessageRole.AGENT
        assert loader._map_role("system") == MessageRole.SYSTEM
        assert loader._map_role("bot") == MessageRole.BOT
        
        # Test that unknown roles raise an error
        with pytest.raises(ValueError, match="Unknown sender role: unknown"):
            loader._map_role("unknown")
