"""Tests for core/errors.py - exception hierarchy."""

import pytest
from adg.core.errors import (
    ADGError,
    BackendNotFoundError,
    ConfigError,
    GenerationError,
    InternalError,
    InvalidExampleError,
    InvalidQueryError,
    MethodNotFoundError,
    ModelNotLoadedError,
    NLIModelError,
    NLIOOMError,
    ParsingError,
    QueryTooLongError,
    SessionNotFoundError,
    SessionStateError,
)


def test_adg_error_base():
    """Test base ADGError exception."""
    err = ADGError("Something went wrong", code="test_error", hint="Try again")

    assert str(err) == "Something went wrong"
    assert err.code == "test_error"
    assert err.hint == "Try again"
    assert err.message == "Something went wrong"


def test_adg_error_default_code():
    """Test ADGError auto-generates code from class name."""
    err = ADGError("Test message")

    # Should use class name without "Error" suffix
    assert err.code == "adg"


def test_invalid_query_error():
    """Test InvalidQueryError."""
    err = InvalidQueryError("Empty query", hint="Please provide a question")

    assert err.code == "invalid_query"
    assert err.hint == "Please provide a question"
    assert isinstance(err, ADGError)


def test_query_too_long_error():
    """Test QueryTooLongError."""
    err = QueryTooLongError(length=3000, max_length=2000)

    assert err.code == "invalid_query"  # Inherits from InvalidQueryError
    assert err.length == 3000
    assert err.max_length == 2000
    assert "3000" in str(err)
    assert "2000" in str(err)
    assert err.hint is not None


def test_generation_error():
    """Test GenerationError."""
    err = GenerationError("Model timeout", backend="hf", hint="Reduce batch size")

    assert err.code == "generation_error"
    assert err.backend == "hf"
    assert err.hint == "Reduce batch size"


def test_model_not_loaded_error():
    """Test ModelNotLoadedError."""
    err = ModelNotLoadedError(model_id="qwen2.5-1.5b")

    assert err.code == "model_not_loaded"
    assert err.model_id == "qwen2.5-1.5b"
    assert "qwen2.5-1.5b" in str(err)
    assert err.hint is not None


def test_parsing_error():
    """Test ParsingError."""
    err = ParsingError("Invalid format", raw_text="Bad output")

    assert err.code == "parsing_error"
    assert err.raw_text == "Bad output"


def test_nli_model_error():
    """Test NLIModelError."""
    err = NLIModelError("CUDA OOM")

    assert err.code == "nli_model_error"
    assert "CUDA OOM" in str(err)
    assert err.hint is not None


def test_nli_oom_error():
    """Test NLIOOMError."""
    err = NLIOOMError(batch_size=32)

    assert err.code == "nli_oom"
    assert err.batch_size == 32
    assert "32" in str(err)


def test_config_error():
    """Test ConfigError."""
    err = ConfigError("Invalid config file", hint="Check YAML syntax")

    assert err.code == "config_error"
    assert err.hint == "Check YAML syntax"


def test_backend_not_found_error():
    """Test BackendNotFoundError."""
    err = BackendNotFoundError("vllm", available=["hf", "openai", "mock"])

    assert err.code == "backend_not_found"
    assert err.backend_name == "vllm"
    assert err.available == ["hf", "openai", "mock"]
    assert "hf" in str(err)
    assert err.hint is not None


def test_method_not_found_error():
    """Test MethodNotFoundError."""
    err = MethodNotFoundError("unknown_method", available=["adg_ic", "semantic_entropy"])

    assert err.code == "method_not_found"
    assert err.method_name == "unknown_method"
    assert len(err.available) == 2


def test_invalid_example_error():
    """Test InvalidExampleError."""
    err = InvalidExampleError(example_id="ex_001", reason="missing_question")

    assert err.code == "invalid_example"
    assert err.example_id == "ex_001"
    assert err.reason == "missing_question"


def test_session_not_found_error():
    """Test SessionNotFoundError."""
    err = SessionNotFoundError(session_id="sess_123")

    assert err.code == "session_not_found"
    assert err.session_id == "sess_123"
    assert "sess_123" in str(err)
    assert "expired" in err.hint.lower()


def test_session_state_error():
    """Test SessionStateError."""
    err = SessionStateError(current_state="IDLE", action="resolve_clarification")

    assert err.code == "session_state_error"
    assert err.current_state == "IDLE"
    assert err.action == "resolve_clarification"


def test_internal_error():
    """Test InternalError."""
    err = InternalError("Unexpected null pointer", component="clustering")

    assert err.code == "internal_error"
    assert err.component == "clustering"
    assert "clustering" in str(err)
    assert "abstain" in err.hint.lower()


def test_error_inheritance():
    """Test exception inheritance hierarchy."""
    # All errors inherit from ADGError
    assert issubclass(InvalidQueryError, ADGError)
    assert issubclass(GenerationError, ADGError)
    assert issubclass(NLIModelError, ADGError)
    assert issubclass(ConfigError, ADGError)
    assert issubclass(BackendNotFoundError, ADGError)

    # QueryTooLongError inherits from InvalidQueryError
    assert issubclass(QueryTooLongError, InvalidQueryError)
    assert issubclass(QueryTooLongError, ADGError)


def test_error_catching():
    """Test catching errors in try-except blocks."""

    # Catch specific error
    with pytest.raises(QueryTooLongError):
        raise QueryTooLongError(length=3000, max_length=2000)

    # Catch base class
    with pytest.raises(ADGError):
        raise NLIModelError("Test error")

    # Verify error details available after catching
    try:
        raise ModelNotLoadedError("test-model")
    except ModelNotLoadedError as e:
        assert e.model_id == "test-model"
        assert e.code == "model_not_loaded"


def test_error_attributes():
    """Test that errors have correct attributes."""
    err = BackendNotFoundError("test", ["a", "b"])

    # All ADGErrors should have these
    assert hasattr(err, "message")
    assert hasattr(err, "code")
    assert hasattr(err, "hint")

    # Specific attributes
    assert hasattr(err, "backend_name")
    assert hasattr(err, "available")
