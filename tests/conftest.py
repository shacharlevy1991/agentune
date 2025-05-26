"""Configuration for pytest.

This file contains fixtures and configurations used by pytest.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chat_simulator.models import Conversation, Message, MessageRole


@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_chats_path(test_data_dir: Path) -> Path:
    """Return the path to the sample chats JSON file."""
    return test_data_dir / "sample_chats.json"


@pytest.fixture(scope="session")
def sample_chats_data(sample_chats_path: Path) -> List[Dict[str, Any]]:
    """Load and return the sample chats data."""
    with open(sample_chats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of conversations")
    return data


@pytest.fixture
def sample_conversation() -> Conversation:
    """Return a sample conversation for testing."""
    return Conversation(
        id="test_conv_1",
        messages=(
            Message(
                content="Hello, I need help with my order",
                role=MessageRole.CUSTOMER
            ),
            Message(
                content="I'd be happy to help. What's your order number?",
                role=MessageRole.AGENT
            )
        )
    )
