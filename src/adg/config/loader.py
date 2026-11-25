"""Configuration loading from YAML files.

Per spec §7: Load base.yaml + referenced configs + CLI overrides.
"""

import contextlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from adg.config.schema import AppConfig
from adg.core.logging import get_logger

log = get_logger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return as dict.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] | None = yaml.safe_load(f)

    if data is None:
        return {}

    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        Merged configuration

    Note:
        Nested dicts are merged; lists and primitives are replaced.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Replace
            result[key] = value

    return result


def resolve_config_reference(
    config_dir: Path, config_type: str, config_name: str
) -> dict[str, Any]:
    """Resolve a config reference like `llm.config: qwen2.5-1.5b`.

    Args:
        config_dir: Base config directory (usually repo_root/configs)
        config_type: Type of config ("models", "nli", "methods", "hardware")
        config_name: Name of the config file (without .yaml)

    Returns:
        Loaded config dict

    Example:
        >>> resolve_config_reference(Path("configs"), "models", "qwen2.5-1.5b")
        # Loads configs/models/qwen2.5-1.5b.yaml
    """
    config_path = config_dir / config_type / f"{config_name}.yaml"

    if not config_path.exists():
        log.warning(
            "config_reference_not_found",
            config_type=config_type,
            config_name=config_name,
            path=str(config_path),
        )
        return {}

    return load_yaml(config_path)


def load_config(
    config_path: Path | str = "configs/base.yaml",
    overrides: dict[str, Any] | None = None,
    resolve_references: bool = True,
) -> AppConfig:
    """Load and validate application configuration.

    Args:
        config_path: Path to base config file
        overrides: Optional dict to override config values (from CLI)
        resolve_references: If True, resolve references like llm.config

    Returns:
        Validated AppConfig

    Raises:
        FileNotFoundError: If config file not found
        ValidationError: If config is invalid

    Example:
        >>> config = load_config("configs/base.yaml")
        >>> config = load_config("configs/base.yaml", overrides={"sampling.n_samples": 16})
    """
    config_path = Path(config_path)
    config_dir = config_path.parent  # configs/ directory

    log.info("loading_config", path=str(config_path))

    # Load base config
    base_data = load_yaml(config_path)

    # Resolve references if requested
    if resolve_references:
        # Resolve LLM config reference
        if "llm" in base_data and "config" in base_data["llm"]:
            llm_config_name = base_data["llm"]["config"]
            llm_config_data = resolve_config_reference(config_dir, "models", llm_config_name)
            # Merge into base, keeping base overrides
            base_data["llm"] = merge_configs(llm_config_data, base_data.get("llm", {}))

        # Resolve NLI config reference
        if "nli" in base_data and "config" in base_data["nli"]:
            nli_config_name = base_data["nli"]["config"]
            nli_config_data = resolve_config_reference(config_dir, "nli", nli_config_name)
            base_data["nli"] = merge_configs(nli_config_data, base_data.get("nli", {}))

        # Resolve hardware config if not "auto"
        if "hardware" in base_data and "config" in base_data["hardware"]:
            hw_config_name = base_data["hardware"]["config"]
            if hw_config_name != "auto":
                hw_config_data = resolve_config_reference(config_dir, "hardware", hw_config_name)
                base_data["hardware"] = merge_configs(hw_config_data, base_data.get("hardware", {}))

    # Apply CLI overrides
    if overrides:
        log.debug("applying_overrides", overrides=overrides)
        base_data = merge_configs(base_data, overrides)

    # Validate with pydantic
    try:
        config = AppConfig(**base_data)
        config._loaded_from = config_path
        log.info(
            "config_loaded",
            config_hash=config._config_hash,
            llm=config.llm.model_id or config.llm.config,
            nli=config.nli.model_id or config.nli.config,
        )
        return config
    except ValidationError as e:
        log.error("config_validation_failed", errors=e.errors())
        raise


def apply_dot_overrides(overrides_list: list[str]) -> dict[str, Any]:
    """Convert CLI dot-notation overrides to nested dict.

    Args:
        overrides_list: List of "key.nested=value" strings

    Returns:
        Nested dict

    Example:
        >>> apply_dot_overrides(["sampling.n_samples=16", "llm.temperature=0.5"])
        {"sampling": {"n_samples": 16}, "llm": {"temperature": 0.5}}
    """
    result: dict[str, Any] = {}

    for override in overrides_list:
        if "=" not in override:
            log.warning("invalid_override_format", override=override)
            continue

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Try to parse value as int, float, bool, or keep as string
        value: Any = value_str
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        elif value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            value = int(value_str)
        else:
            # Try float parsing
            with contextlib.suppress(ValueError):
                value = float(value_str)

        # Build nested dict
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    return result
