"""Exception hierarchy for ADG.

Per spec §6.1 and §4.12: typed exceptions for error handling with structured hints.
"""


# ==============================================================================
# Base Exception
# ==============================================================================


class ADGError(Exception):
    """Base exception for all ADG errors.

    All exceptions carry a machine-readable code and optional hint for the user.
    """

    def __init__(self, message: str, code: str | None = None, hint: str | None = None):
        """Initialize ADG exception.

        Args:
            message: Error message for logging
            code: Machine-readable error code (e.g., "invalid_query", "nli_error")
            hint: Optional user-facing hint for recovery
        """
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__.lower().replace("error", "")
        self.hint = hint


# ==============================================================================
# Input Validation Errors
# ==============================================================================


class InvalidQueryError(ADGError):
    """Query validation failed.

    Per spec §4.2: empty query, too long, or invalid characters.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message, code="invalid_query", hint=hint)


class QueryTooLongError(InvalidQueryError):
    """Query exceeds maximum length.

    Per spec config: max_query_chars (default 2000).
    """

    def __init__(self, length: int, max_length: int):
        super().__init__(
            f"Query length {length} exceeds maximum {max_length}",
            hint=f"Please shorten your question to under {max_length} characters.",
        )
        self.length = length
        self.max_length = max_length


# ==============================================================================
# LLM Backend Errors
# ==============================================================================


class LLMError(ADGError):
    """Base class for LLM backend errors.

    Per spec §4.12: LLM generation failures.
    """


class GenerationError(LLMError):
    """LLM generation failed.

    Per spec §4.12: retried per tenacity policy, then raised.
    """

    def __init__(self, message: str, backend: str, hint: str | None = None):
        super().__init__(message, code="generation_error", hint=hint)
        self.backend = backend


class ModelNotLoadedError(LLMError):
    """Model not loaded or available."""

    def __init__(self, model_id: str):
        super().__init__(
            f"Model not loaded: {model_id}",
            code="model_not_loaded",
            hint="Check that the model is properly configured and loaded.",
        )
        self.model_id = model_id


class ParsingError(LLMError):
    """Failed to parse structured LLM output.

    Per spec §4.3: invalid interpretation/answer format.
    """

    def __init__(self, message: str, raw_text: str):
        super().__init__(
            message,
            code="parsing_error",
            hint="The model's response format was invalid. This may be retried.",
        )
        self.raw_text = raw_text


# ==============================================================================
# NLI Errors
# ==============================================================================


class NLIError(ADGError):
    """Base class for NLI scoring errors.

    Per spec §4.12: NLI model failures (OOM, runtime errors).
    """


class NLIModelError(NLIError):
    """NLI model inference failed.

    Per spec §4.12: retry with smaller batch, then raise if persistent.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(
            message,
            code="nli_model_error",
            hint=hint or "NLI scoring failed. The system will abstain.",
        )


class NLIOOMError(NLIError):
    """NLI model ran out of memory.

    Per spec §4.12: retry with batch_size / 2.
    """

    def __init__(self, batch_size: int):
        super().__init__(
            f"NLI OOM with batch_size={batch_size}",
            code="nli_oom",
            hint="Reducing batch size for retry.",
        )
        self.batch_size = batch_size


# ==============================================================================
# Configuration Errors
# ==============================================================================


class ConfigError(ADGError):
    """Configuration validation or loading error."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message, code="config_error", hint=hint)


class MissingConfigError(ConfigError):
    """Required configuration missing."""

    def __init__(self, key: str):
        super().__init__(
            f"Missing required config: {key}",
            hint=f"Please provide '{key}' in your configuration.",
        )
        self.key = key


# ==============================================================================
# Backend Registration Errors
# ==============================================================================


class RegistryError(ADGError):
    """Backend or method registration error."""


class BackendNotFoundError(RegistryError):
    """Requested backend not registered."""

    def __init__(self, backend_name: str, available: list[str]):
        super().__init__(
            f"Backend '{backend_name}' not found. Available: {', '.join(available)}",
            code="backend_not_found",
            hint=f"Use one of: {', '.join(available)}",
        )
        self.backend_name = backend_name
        self.available = available


class MethodNotFoundError(RegistryError):
    """Requested method not registered."""

    def __init__(self, method_name: str, available: list[str]):
        super().__init__(
            f"Method '{method_name}' not found. Available: {', '.join(available)}",
            code="method_not_found",
            hint=f"Use one of: {', '.join(available)}",
        )
        self.method_name = method_name
        self.available = available


# ==============================================================================
# Data Loading Errors
# ==============================================================================


class DataError(ADGError):
    """Base class for dataset loading errors."""


class DatasetNotFoundError(DataError):
    """Requested dataset not found."""

    def __init__(self, dataset: str):
        super().__init__(
            f"Dataset not found: {dataset}",
            code="dataset_not_found",
            hint="Check that the dataset is downloaded and in the correct location.",
        )
        self.dataset = dataset


class InvalidExampleError(DataError):
    """Dataset example failed validation.

    Per spec §9.3: skip with logged reason, never crash.
    """

    def __init__(self, example_id: str, reason: str):
        super().__init__(
            f"Invalid example {example_id}: {reason}",
            code="invalid_example",
        )
        self.example_id = example_id
        self.reason = reason


# ==============================================================================
# Conversation Errors
# ==============================================================================


class ConversationError(ADGError):
    """Multi-turn conversation errors."""


class SessionNotFoundError(ConversationError):
    """Session ID not found or expired."""

    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            code="session_not_found",
            hint="The session may have expired. Please start a new conversation.",
        )
        self.session_id = session_id


class SessionStateError(ConversationError):
    """Invalid session state transition."""

    def __init__(self, current_state: str, action: str):
        super().__init__(
            f"Cannot perform '{action}' in state '{current_state}'",
            code="session_state_error",
        )
        self.current_state = current_state
        self.action = action


# ==============================================================================
# Internal Errors (should not reach user)
# ==============================================================================


class InternalError(ADGError):
    """Unexpected internal error.

    Should trigger ABSTAIN with reason 'internal_error'.
    """

    def __init__(self, message: str, component: str):
        super().__init__(
            f"Internal error in {component}: {message}",
            code="internal_error",
            hint="An unexpected error occurred. The system will abstain from answering.",
        )
        self.component = component


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Base
    "ADGError",
    "BackendNotFoundError",
    # Configuration
    "ConfigError",
    # Conversation
    "ConversationError",
    # Data
    "DataError",
    "DatasetNotFoundError",
    "GenerationError",
    # Internal
    "InternalError",
    "InvalidExampleError",
    # Input validation
    "InvalidQueryError",
    # LLM
    "LLMError",
    "MethodNotFoundError",
    "MissingConfigError",
    "ModelNotLoadedError",
    # NLI
    "NLIError",
    "NLIModelError",
    "NLIOOMError",
    "ParsingError",
    "QueryTooLongError",
    # Registry
    "RegistryError",
    "SessionNotFoundError",
    "SessionStateError",
]
