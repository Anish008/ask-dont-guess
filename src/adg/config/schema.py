"""Configuration schema using pydantic.

Per spec §7: YAML config files validate against these models.
Structure matches configs/base.yaml from the spec.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project-level config
class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    name: str = "ask-dont-guess"
    seed: int = 0
    env: Literal["dev", "prod"] = "dev"


# LLM configuration
class LLMConfig(BaseModel):
    """LLM backend configuration."""

    config: str = "qwen2.5-1.5b"  # Reference to configs/models/{config}.yaml
    backend: Literal["hf", "vllm", "openai", "mock"] = "hf"
    model_id: str | None = None  # Resolved from config file
    revision: str = "main"
    dtype: Literal["auto", "fp16", "fp32", "bf16"] = "auto"
    load_in_4bit: bool = False
    max_context: int = 4096
    trust_remote_code: bool = False
    # OpenAI-specific
    api_key_env: str | None = "OPENAI_API_KEY"
    timeout_s: int = 60
    max_retries: int = 5


# NLI configuration
class NLIConfig(BaseModel):
    """NLI scorer configuration."""

    config: str = "deberta-large-mnli"  # Reference to configs/nli/{config}.yaml
    model_id: str | None = None  # Resolved
    input_format: Literal["qa", "answer_only", "interpretation_answer"] = "qa"
    batch_size: int = 32
    max_length: int = 512


# Sampling configuration
class SamplingConfig(BaseModel):
    """Answer sampling configuration."""

    n_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 96
    prompt: Literal["structured", "plain"] = "structured"
    max_resample_attempts: int = 1
    seed: int = 0  # Will be derived per query


# Relations and clustering configuration
class RelationsConfig(BaseModel):
    """Pairwise NLI relations configuration."""

    equivalence: Literal["strict", "soft"] = "soft"
    tau_eq: float = 0.5
    entity_number_guard: bool = True
    guard_cap: float = 0.2


class ClusteringConfig(BaseModel):
    """Clustering algorithm configuration."""

    algorithm: Literal[
        "greedy_bidirectional", "agglomerative", "connected_components"
    ] = "agglomerative"
    linkage: Literal["average", "complete"] = "average"
    hedge_mode: Literal["split", "normal"] = "split"


class FeaturesConfig(BaseModel):
    """Cluster features configuration."""

    mass: Literal["discrete", "likelihood"] = "discrete"
    significance_rho: float = 0.25
    gamma: float = 0.5  # Balance exponent in ask_score


# Interpretation channel configuration
class InterpretationChannelConfig(BaseModel):
    """Interpretation channel configuration ([ER] addition)."""

    enabled: bool = True
    divergence: Literal["hard", "soft"] = "soft"


# Verification configuration
class VerificationConfig(BaseModel):
    """Conditional consistency verification ([ER] addition, default OFF)."""

    enabled: bool = False
    top_k: int = 2
    m_samples: int = 4


# Decision engine configuration
class ThresholdsConfig(BaseModel):
    """Decision thresholds (uncalibrated defaults)."""

    tau_refuse: float = 0.5
    tau_answer: float = 0.7
    tau_intra: float = 0.5
    k_max: int = 4
    tau_cov: float = 0.6
    tau_sep: float = 0.3
    tau_div: float = 0.2
    tau_cc: float = 0.6
    tau_hedge: float = 0.25


class DecisionConfig(BaseModel):
    """Decision engine configuration."""

    mode: Literal["rules", "learned"] = "rules"
    thresholds_file: Path | None = None  # Path to calibrated thresholds
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)


# Clarification configuration
class ClarifyConfig(BaseModel):
    """Clarifying question generation configuration."""

    mode: Literal["template", "llm"] = "llm"
    max_options: int = 4
    max_words: int = 40


# Conversation configuration
class ConversationConfig(BaseModel):
    """Multi-turn conversation configuration."""

    max_clarification_turns: int = 2
    on_unresolved: Literal["assume_and_answer", "abstain"] = "assume_and_answer"
    session_ttl_s: int = 3600
    reuse_cluster_answer: bool = True


# Cache configuration
class CacheConfig(BaseModel):
    """Disk cache configuration."""

    enabled: bool = True
    dir: Path = Path(".cache/adg")


# Hardware configuration
class HardwareConfig(BaseModel):
    """Hardware configuration (auto-detected or explicit)."""

    config: Literal["auto", "cpu", "gpu_16gb", "gpu_8gb", "gpu_8gb_4bit", "mps"] = "auto"


# API configuration
class APIConfig(BaseModel):
    """FastAPI service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    max_query_chars: int = 2000


# Logging configuration
class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False  # True for production


# Tracking configuration
class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    backend: Literal["none", "wandb", "mlflow"] = "none"
    project: str | None = None
    run_name: str | None = None


# Main application config
class AppConfig(BaseSettings):
    """Complete application configuration.

    Loads from configs/base.yaml + environment overrides.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADG_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    relations: RelationsConfig = Field(default_factory=RelationsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    interpretation_channel: InterpretationChannelConfig = Field(
        default_factory=InterpretationChannelConfig
    )
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    clarify: ClarifyConfig = Field(default_factory=ClarifyConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    # Additional fields for resolved values
    _config_hash: str | None = None
    _loaded_from: Path | None = None

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization: compute config hash."""
        from adg.core.hashing import config_hash

        self._config_hash = config_hash(self.model_dump(mode="json"))
