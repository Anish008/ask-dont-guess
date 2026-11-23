"""Tests for seeding utilities."""

import numpy as np
import torch
from adg.core.seeding import derive_seed, get_torch_generator, set_global_seed


def test_set_global_seed():
    """Test global seed sets all RNGs."""
    import random

    set_global_seed(42)

    # Check Python random, numpy, torch produce same values
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    # Reset and verify reproducibility
    set_global_seed(42)
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert abs(t1 - t2) < 1e-6


def test_derive_seed_deterministic():
    """Test seed derivation is deterministic."""
    seed1 = derive_seed(0, "query_123")
    seed2 = derive_seed(0, "query_123")
    assert seed1 == seed2

    # Different keys produce different seeds
    seed3 = derive_seed(0, "query_124")
    assert seed1 != seed3

    # Different global seeds produce different seeds
    seed4 = derive_seed(1, "query_123")
    assert seed1 != seed4


def test_derive_seed_range():
    """Test derived seed is in valid range."""
    seed = derive_seed(0, "test")
    assert 0 <= seed < 2**31


def test_derive_seed_multiple_keys():
    """Test derivation with multiple keys."""
    seed1 = derive_seed(0, "train", "fold_1", "query_123")
    seed2 = derive_seed(0, "train", "fold_1", "query_123")
    assert seed1 == seed2

    seed3 = derive_seed(0, "train", "fold_2", "query_123")
    assert seed1 != seed3


def test_get_torch_generator():
    """Test torch generator creation."""
    gen1 = get_torch_generator(42, "cpu")
    gen2 = get_torch_generator(42, "cpu")

    # Same seed should produce same values
    t1 = torch.rand(1, generator=gen1).item()
    t2 = torch.rand(1, generator=gen2).item()

    assert abs(t1 - t2) < 1e-6
