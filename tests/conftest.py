"""Configuration for pytest.

This file contains fixtures and configurations used by pytest.
"""
import logging
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
