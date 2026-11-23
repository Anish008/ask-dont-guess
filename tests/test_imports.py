"""Trivial smoke test to verify package imports."""


def test_import_adg():
    """Test that the adg package can be imported."""
    import adg

    assert hasattr(adg, "__version__")
    assert adg.__version__ == "0.1.0"


def test_import_core():
    """Test that core module imports."""
    from adg import core

    assert core is not None
