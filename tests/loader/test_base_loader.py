"""Tests for the base chat loader."""

import pytest
from pathlib import Path
from typing import List

from chat_simulator.loader.base import ChatLoader
from chat_simulator.models import Conversation


def test_chat_loader_is_abstract():
    """Test that ChatLoader is an abstract base class."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ChatLoader()  # type: ignore


def test_chat_loader_requires_load_method():
    """Test that subclasses must implement the load method."""
    class IncompleteLoader(ChatLoader):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteLoader()  # type: ignore


class TestDummyLoader(ChatLoader):
    """A concrete implementation of ChatLoader for testing."""
    
    def load(self, source: Path) -> List[Conversation]:
        """Load conversations from a source."""
        return [
            Conversation(
                id="dummy_1",
                messages=[],
                metadata={"source": str(source)}
            )
        ]


def test_concrete_loader_implementation():
    """Test that a concrete implementation works as expected."""
    loader = TestDummyLoader()
    test_file = Path("/dummy/path")
    
    result = loader.load(test_file)
    
    assert len(result) == 1
    assert result[0].id == "dummy_1"
    assert result[0].metadata["source"] == str(test_file)
