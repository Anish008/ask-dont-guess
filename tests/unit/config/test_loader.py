"""Tests for config loading."""


import pytest
from adg.config.loader import apply_dot_overrides, load_config, merge_configs


def test_merge_configs():
    """Test config merging."""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 4}

    result = merge_configs(base, override)

    assert result["a"] == 1
    assert result["b"]["c"] == 99  # Overridden
    assert result["b"]["d"] == 3  # Preserved
    assert result["e"] == 4  # Added


def test_apply_dot_overrides():
    """Test CLI dot-notation overrides."""
    overrides = [
        "sampling.n_samples=16",
        "sampling.temperature=0.5",
        "llm.load_in_4bit=true",
    ]

    result = apply_dot_overrides(overrides)

    assert result["sampling"]["n_samples"] == 16
    assert result["sampling"]["temperature"] == 0.5
    assert result["llm"]["load_in_4bit"] is True


def test_apply_dot_overrides_types():
    """Test type inference in overrides."""
    overrides = [
        "int_val=42",
        "float_val=3.14",
        "bool_true=true",
        "bool_false=false",
        "string_val=hello",
    ]

    result = apply_dot_overrides(overrides)

    assert result["int_val"] == 42
    assert result["float_val"] == 3.14
    assert result["bool_true"] is True
    assert result["bool_false"] is False
    assert result["string_val"] == "hello"


def test_load_config_base():
    """Test loading base config."""
    config = load_config("configs/base.yaml", resolve_references=False)

    assert config.project.name == "ask-dont-guess"
    assert config.project.seed == 0
    assert config.sampling.n_samples == 8
    assert config.sampling.temperature == 0.7
    assert config.decision.thresholds.tau_answer == 0.7


def test_load_config_with_references():
    """Test config loading resolves references."""
    config = load_config("configs/base.yaml", resolve_references=True)

    # LLM config should be resolved from qwen2.5-1.5b.yaml
    assert config.llm.backend == "hf"
    assert "qwen" in config.llm.model_id.lower() if config.llm.model_id else True

    # NLI config should be resolved
    assert config.nli.model_id or config.nli.config


def test_load_config_with_overrides():
    """Test loading config with overrides."""
    overrides = {"sampling": {"n_samples": 16}, "llm": {"load_in_4bit": True}}

    config = load_config("configs/base.yaml", overrides=overrides, resolve_references=False)

    assert config.sampling.n_samples == 16  # Overridden
    assert config.llm.load_in_4bit is True  # Overridden
    assert config.sampling.temperature == 0.7  # Preserved


def test_load_config_missing_file():
    """Test loading nonexistent config raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")


def test_config_hash_computed():
    """Test config hash is computed on load."""
    config = load_config("configs/base.yaml", resolve_references=False)

    assert config._config_hash is not None
    assert len(config._config_hash) == 16
