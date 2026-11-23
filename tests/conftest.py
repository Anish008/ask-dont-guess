"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def mock_config():
    """Minimal mock configuration for tests."""
    return {}
