"""Reproducible seeding for experiments.

Per spec §4.3 and §12.2: Seeds are derived deterministically from
a global seed and query ID, so results don't depend on dataset order.
"""

import hashlib
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Args:
        seed: Global seed value

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(global_seed: int, *keys: str | int) -> int:
    """Derive a deterministic seed from global seed and additional keys.

    Per spec §4.3: seed_q = hash64(global_seed, query_id)

    Args:
        global_seed: The base seed
        *keys: Additional keys (e.g., query_id, split_name, iteration)

    Returns:
        A derived 32-bit seed (0 to 2^31 - 1)

    Example:
        >>> seed_q = derive_seed(0, "query_123")
        >>> seed_split = derive_seed(0, "train", "fold_1")
    """
    # Combine all keys into a single string
    combined = f"{global_seed}::{':'.join(str(k) for k in keys)}"

    # Hash to get deterministic value
    hash_bytes = hashlib.sha256(combined.encode("utf-8")).digest()

    # Take first 4 bytes as seed (unsigned 32-bit)
    seed = int.from_bytes(hash_bytes[:4], byteorder="big")

    # Clip to valid seed range for PyTorch/NumPy (2^31 - 1)
    return seed % (2**31)


def get_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Create a torch.Generator with the given seed.

    Args:
        seed: Seed value
        device: Device for the generator ("cpu", "cuda", etc.)

    Returns:
        A seeded torch.Generator

    Example:
        >>> gen = get_torch_generator(42, "cuda")
        >>> model.generate(..., generator=gen)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
