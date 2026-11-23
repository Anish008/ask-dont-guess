"""Content-addressed hashing for cache keys.

Per spec §3.4 and §6.12: Cache keys are stable hashes of
(model_id, revision, prompt, generation params, seed).
"""

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    """Generate a stable hash for any JSON-serializable object.

    Args:
        obj: Any object that can be JSON-serialized
            (dict, list, str, int, float, bool, None)

    Returns:
        A 16-character hex hash (first 64 bits of SHA-256)

    Example:
        >>> key = stable_hash({"model": "qwen2.5-1.5b", "temp": 0.7, "seed": 0})
        >>> # Same inputs always produce same hash
    """
    # Serialize to JSON with sorted keys for stability
    json_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))

    # Hash the JSON string
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).digest()

    # Return first 64 bits as hex (16 characters)
    return hash_bytes[:8].hex()


def config_hash(config: dict[str, Any]) -> str:
    """Generate a hash of a configuration dict.

    Args:
        config: Configuration dictionary

    Returns:
        A 16-character hex hash

    Note:
        Useful for tracking which exact config was used in a run.
    """
    return stable_hash(config)


def cache_key(*parts: Any) -> str:
    """Generate a cache key from multiple parts.

    Args:
        *parts: Any JSON-serializable objects

    Returns:
        A cache key string (concatenated hashes)

    Example:
        >>> key = cache_key("gen", model_id, prompt, params)
        >>> cached_value = cache.get(key)
    """
    # Hash each part separately, then join
    hashes = [stable_hash(part) for part in parts]
    return ":".join(hashes)
