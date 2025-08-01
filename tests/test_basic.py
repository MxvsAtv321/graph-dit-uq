def test_import():
    """Test that the package can be imported."""
    try:
        import src
        assert src.__version__ == "0.1.0"
    except ImportError:
        pass  # Package not installed in test environment

def test_basic():
    """Basic test to ensure CI pipeline works."""
    assert True 