"""Tests for hashing utilities."""

from adg.core.hashing import cache_key, config_hash, stable_hash


def test_stable_hash_deterministic():
    """Test hash is deterministic for same input."""
    obj = {"model": "qwen", "temp": 0.7, "seed": 0}

    hash1 = stable_hash(obj)
    hash2 = stable_hash(obj)

    assert hash1 == hash2
    assert len(hash1) == 16  # 8 bytes = 16 hex chars


def test_stable_hash_key_order():
    """Test hash is same regardless of key order."""
    obj1 = {"a": 1, "b": 2, "c": 3}
    obj2 = {"c": 3, "a": 1, "b": 2}

    assert stable_hash(obj1) == stable_hash(obj2)


def test_stable_hash_different_values():
    """Test different values produce different hashes."""
    hash1 = stable_hash({"temp": 0.7})
    hash2 = stable_hash({"temp": 0.8})

    assert hash1 != hash2


def test_stable_hash_various_types():
    """Test hashing various JSON types."""
    # Simple types
    assert stable_hash("test") != stable_hash("test2")
    assert stable_hash(42) != stable_hash(43)
    assert stable_hash(True) != stable_hash(False)
    assert stable_hash(None) is not None

    # Complex types
    assert stable_hash([1, 2, 3]) != stable_hash([1, 2, 4])
    assert stable_hash({"a": [1, 2]}) != stable_hash({"a": [1, 3]})


def test_config_hash():
    """Test config hash wrapper."""
    config = {"sampling": {"n_samples": 8}, "llm": {"model": "qwen"}}
    h = config_hash(config)

    assert len(h) == 16
    assert h == stable_hash(config)


def test_cache_key():
    """Test cache key generation from multiple parts."""
    key = cache_key("gen", "qwen2.5-1.5b", {"temp": 0.7}, 0)

    # Should be 4 hashes joined by colons
    parts = key.split(":")
    assert len(parts) == 4
    assert all(len(p) == 16 for p in parts)

    # Same inputs produce same key
    key2 = cache_key("gen", "qwen2.5-1.5b", {"temp": 0.7}, 0)
    assert key == key2

    # Different inputs produce different keys
    key3 = cache_key("gen", "qwen2.5-1.5b", {"temp": 0.8}, 0)
    assert key != key3
