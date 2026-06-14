"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from football_predictor.data.sample_data import generate_sample_matches


@pytest.fixture(scope="session")
def sample_matches():
    """A small, deterministic synthetic dataset shared across tests."""
    return generate_sample_matches()
