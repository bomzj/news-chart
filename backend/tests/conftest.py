import pytest


@pytest.fixture
def sample_embeddings():
    """Generate sample normalized vectors for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    vecs = rng.random((5, 1536))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


@pytest.fixture
def duplicate_embeddings():
    """Two nearly identical vectors (cosine > 0.95) + one different."""
    import numpy as np

    rng = np.random.default_rng(42)
    base = rng.random(1536)
    base = base / np.linalg.norm(base)

    # Add tiny noise to create near-duplicate
    noise = rng.random(1536) * 0.01
    near_dup = base + noise
    near_dup = near_dup / np.linalg.norm(near_dup)

    # Create a clearly different vector
    different = rng.random(1536)
    different = different / np.linalg.norm(different)

    return [base.tolist(), near_dup.tolist(), different.tolist()]
