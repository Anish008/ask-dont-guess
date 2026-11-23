"""Registry pattern for backends and methods.

Provides simple registries for pluggable LLM backends, NLI scorers, methods, and simulated users.
"""

from collections.abc import Callable
from typing import Any

from adg.core.errors import BackendNotFoundError, MethodNotFoundError
from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend, Method, NLIScorer, SimulatedUser

log = get_logger(__name__)


# ==============================================================================
# Backend Registry
# ==============================================================================


class BackendRegistry:
    """Registry for LLM backends.

    Usage:
        >>> from adg.core.registry import backend_registry
        >>> backend_registry.register("mock", MockLLMBackend)
        >>> backend = backend_registry.get("mock", config=config)
    """

    def __init__(self) -> None:
        self._backends: dict[str, type[LLMBackend]] = {}

    def register(self, name: str, backend_class: type[LLMBackend]) -> None:
        """Register a backend class.

        Args:
            name: Backend name (e.g., "hf", "vllm", "openai", "mock")
            backend_class: Backend class implementing LLMBackend protocol
        """
        self._backends[name] = backend_class
        log.debug("backend_registered", name=name, class_name=backend_class.__name__)

    def get(self, name: str, **kwargs: Any) -> LLMBackend:
        """Get and instantiate a backend.

        Args:
            name: Backend name
            **kwargs: Constructor arguments for the backend

        Returns:
            Instantiated backend

        Raises:
            BackendNotFoundError: If backend not registered
        """
        if name not in self._backends:
            raise BackendNotFoundError(name, list(self._backends.keys()))

        backend_class = self._backends[name]
        return backend_class(**kwargs)

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())


# Global backend registry instance
backend_registry = BackendRegistry()


# ==============================================================================
# NLI Scorer Registry
# ==============================================================================


class NLIScorerRegistry:
    """Registry for NLI scorers.

    Usage:
        >>> from adg.core.registry import nli_registry
        >>> nli_registry.register("hf", HFNLIScorer)
        >>> scorer = nli_registry.get("hf", config=config)
    """

    def __init__(self) -> None:
        self._scorers: dict[str, type[NLIScorer]] = {}

    def register(self, name: str, scorer_class: type[NLIScorer]) -> None:
        """Register an NLI scorer class.

        Args:
            name: Scorer name (e.g., "hf", "mock")
            scorer_class: Scorer class implementing NLIScorer protocol
        """
        self._scorers[name] = scorer_class
        log.debug("nli_scorer_registered", name=name, class_name=scorer_class.__name__)

    def get(self, name: str, **kwargs: Any) -> NLIScorer:
        """Get and instantiate an NLI scorer.

        Args:
            name: Scorer name
            **kwargs: Constructor arguments

        Returns:
            Instantiated scorer

        Raises:
            BackendNotFoundError: If scorer not registered
        """
        if name not in self._scorers:
            raise BackendNotFoundError(name, list(self._scorers.keys()))

        scorer_class = self._scorers[name]
        return scorer_class(**kwargs)

    def list_scorers(self) -> list[str]:
        """List all registered scorer names."""
        return list(self._scorers.keys())


# Global NLI scorer registry instance
nli_registry = NLIScorerRegistry()


# ==============================================================================
# Method Registry
# ==============================================================================


class MethodRegistry:
    """Registry for decision methods.

    Usage:
        >>> from adg.core.registry import method_registry
        >>> method_registry.register("adg_ic", ADGMethod)
        >>> method = method_registry.get("adg_ic")
    """

    def __init__(self) -> None:
        self._methods: dict[str, type[Method]] = {}

    def register(self, name: str, method_class: type[Method]) -> None:
        """Register a method class.

        Args:
            name: Method name (e.g., "adg_ic", "semantic_entropy")
            method_class: Method class implementing Method protocol
        """
        self._methods[name] = method_class
        log.debug("method_registered", name=name, class_name=method_class.__name__)

    def get(self, name: str, **kwargs: Any) -> Method:
        """Get and instantiate a method.

        Args:
            name: Method name
            **kwargs: Constructor arguments

        Returns:
            Instantiated method

        Raises:
            MethodNotFoundError: If method not registered
        """
        if name not in self._methods:
            raise MethodNotFoundError(name, list(self._methods.keys()))

        method_class = self._methods[name]
        return method_class(**kwargs)

    def list_methods(self) -> list[str]:
        """List all registered method names."""
        return list(self._methods.keys())


# Global method registry instance
method_registry = MethodRegistry()


# ==============================================================================
# Simulated User Registry
# ==============================================================================


class SimUserRegistry:
    """Registry for simulated users.

    Usage:
        >>> from adg.core.registry import simuser_registry
        >>> simuser_registry.register("rule", RuleSimUser)
        >>> user = simuser_registry.get("rule", config=config)
    """

    def __init__(self) -> None:
        self._users: dict[str, type[SimulatedUser]] = {}

    def register(self, name: str, user_class: type[SimulatedUser]) -> None:
        """Register a simulated user class.

        Args:
            name: User type name (e.g., "rule", "llm")
            user_class: User class implementing SimulatedUser protocol
        """
        self._users[name] = user_class
        log.debug("simuser_registered", name=name, class_name=user_class.__name__)

    def get(self, name: str, **kwargs: Any) -> SimulatedUser:
        """Get and instantiate a simulated user.

        Args:
            name: User type name
            **kwargs: Constructor arguments

        Returns:
            Instantiated simulated user

        Raises:
            BackendNotFoundError: If user type not registered
        """
        if name not in self._users:
            raise BackendNotFoundError(name, list(self._users.keys()))

        user_class = self._users[name]
        return user_class(**kwargs)

    def list_users(self) -> list[str]:
        """List all registered simulated user types."""
        return list(self._users.keys())


# Global simulated user registry instance
simuser_registry = SimUserRegistry()


# ==============================================================================
# Decorator-based Registration
# ==============================================================================


def register_backend(name: str) -> Callable[[type[LLMBackend]], type[LLMBackend]]:
    """Decorator to register an LLM backend.

    Usage:
        >>> @register_backend("mock")
        >>> class MockLLMBackend:
        >>>     ...
    """

    def decorator(cls: type[LLMBackend]) -> type[LLMBackend]:
        backend_registry.register(name, cls)
        return cls

    return decorator


def register_nli(name: str) -> Callable[[type[NLIScorer]], type[NLIScorer]]:
    """Decorator to register an NLI scorer.

    Usage:
        >>> @register_nli("mock")
        >>> class MockNLIScorer:
        >>>     ...
    """

    def decorator(cls: type[NLIScorer]) -> type[NLIScorer]:
        nli_registry.register(name, cls)
        return cls

    return decorator


def register_method(name: str) -> Callable[[type[Method]], type[Method]]:
    """Decorator to register a method.

    Usage:
        >>> @register_method("adg_ic")
        >>> class ADGMethod:
        >>>     ...
    """

    def decorator(cls: type[Method]) -> type[Method]:
        method_registry.register(name, cls)
        return cls

    return decorator


def register_simuser(name: str) -> Callable[[type[SimulatedUser]], type[SimulatedUser]]:
    """Decorator to register a simulated user.

    Usage:
        >>> @register_simuser("rule")
        >>> class RuleSimUser:
        >>>     ...
    """

    def decorator(cls: type[SimulatedUser]) -> type[SimulatedUser]:
        simuser_registry.register(name, cls)
        return cls

    return decorator


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Registry classes
    "BackendRegistry",
    "MethodRegistry",
    "NLIScorerRegistry",
    "SimUserRegistry",
    # Registries
    "backend_registry",
    "method_registry",
    "nli_registry",
    # Decorators
    "register_backend",
    "register_method",
    "register_nli",
    "register_simuser",
    "simuser_registry",
]
