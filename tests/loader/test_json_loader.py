"""Tests for the JSON chat loader."""

import json

import pytest

from chat_simulator.loader.json_loader import JsonChatLoader
from chat_simulator.models import Conversation


class TestJsonChatLoader:
    """Tests for the JsonChatLoader class."""

    def test_json_loader_is_abstract(self):
        """Test that JsonChatLoader is still abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            JsonChatLoader()  # type: ignore

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises an error."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("{invalid json}")

        class TestLoader(JsonChatLoader):
            def parse_conversation(self, raw):
                return Conversation(id="test", messages=())

        with pytest.raises(ValueError, match="Invalid JSON"):
            TestLoader().load(test_file)

    def test_load_valid_json(self, tmp_path, sample_chats_data):
        """Test loading valid JSON data."""
        test_file = tmp_path / "valid.json"
        test_file.write_text(json.dumps(sample_chats_data))

        class TestLoader(JsonChatLoader):
            def parse_conversation(self, raw):
                return Conversation(
                    id=raw["dialogue_id"],
                    messages=()
                )

        result = TestLoader().load(test_file)
        assert len(result) == 2
        assert result[0].id == "conv_123"
        assert result[1].id == "conv_124"
