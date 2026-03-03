"""Tests for imports."""


def test_import_da4hiflow():
    """Test that da4hiflow can be imported."""
    import da4hiflow

    assert da4hiflow.__version__ is not None


def test_import_core():
    """Test that core modules can be imported."""
    from da4hiflow.core import SystemBase, AssimilatorBase

    assert SystemBase is not None
    assert AssimilatorBase is not None
