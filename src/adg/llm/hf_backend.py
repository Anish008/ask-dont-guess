"""HuggingFace Transformers LLM backend.

Per spec §4.3: local model inference with HuggingFace transformers.
"""

from __future__ import annotations

from typing import Any

import torch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

from adg.config.schema import AppConfig
from adg.core.device import detect_hardware, get_torch_dtype
from adg.core.errors import GenerationError, ModelNotLoadedError
from adg.core.logging import get_logger
from adg.core.types import Query, SamplingResult
from adg.llm.parsing import parse_structured_output
from adg.llm.prompts import format_structured_prompt

log = get_logger(__name__)


class HFLLMBackend:
    """HuggingFace Transformers backend for local LLM inference.

    Supports fp16, fp32, int8, and 4-bit quantization modes.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize HF backend and load model.

        Args:
            config: Application configuration

        Raises:
            ModelNotLoadedError: If model loading fails
        """
        self.config = config
        self.model_id = config.llm.model_id or config.llm.config
        self.hw_info = detect_hardware()

        log.info(
            "hf_backend_initializing",
            model_id=self.model_id,
            device=self.hw_info.device,
            dtype=str(self.hw_info.dtype),
            load_in_4bit=config.llm.load_in_4bit,
        )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=config.llm.trust_remote_code,
            )
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise ModelNotLoadedError(self.model_id) from e

        # Load model with device and dtype
        self.model = self._load_model()

        log.info("hf_backend_initialized", model_id=self.model_id)

    def _load_model(self) -> Any:
        """Load model with appropriate device and quantization settings.

        Returns:
            Loaded model

        Raises:
            ModelNotLoadedError: If loading fails
        """
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.llm.trust_remote_code,
        }

        # 4-bit quantization
        if self.config.llm.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            # Use bitsandbytes config for 4-bit
            from transformers import BitsAndBytesConfig

            torch_dtype = get_torch_dtype(self.hw_info.dtype)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            # fp16, fp32, or bf16
            torch_dtype = get_torch_dtype(self.hw_info.dtype)
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["device_map"] = self.hw_info.device

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            model.eval()
            return model
        except Exception as e:
            log.error("model_load_failed", model_id=self.model_id, error=str(e))
            raise ModelNotLoadedError(self.model_id) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(GenerationError),
    )
    def sample(
        self,
        query: Query,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> SamplingResult:
        """Sample N answers from the LLM.

        Args:
            query: Input query
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility

        Returns:
            SamplingResult with N samples

        Raises:
            GenerationError: If generation fails after retries
        """
        log.debug(
            "hf_sampling",
            query_id=query.query_id,
            n_samples=n_samples,
            temperature=temperature,
            seed=seed,
        )

        # Set seed for reproducibility
        set_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Format prompt
        prompt = format_structured_prompt(query.text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.llm.max_context,
        )
        inputs = {k: v.to(self.hw_info.device) for k, v in inputs.items()}

        # Generation config
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Parse each output into Sample
            samples = []
            for i, text in enumerate(generated_texts):
                try:
                    sample = parse_structured_output(text, sample_id=i)
                    samples.append(sample)
                except Exception as e:
                    log.warning("parse_failed", sample_id=i, error=str(e))
                    # Create invalid sample
                    from adg.core.types import Sample

                    samples.append(
                        Sample(
                            interpretation="",
                            answer="",
                            sample_id=i,
                            is_valid=False,
                            raw_text=text,
                        )
                    )

            # Count valid and refusal samples
            n_valid = sum(1 for s in samples if s.is_valid and not s.is_refusal)
            n_refusals = sum(1 for s in samples if s.is_refusal)

            log.info(
                "hf_sampling_complete",
                query_id=query.query_id,
                n_samples=len(samples),
                n_valid=n_valid,
                n_refusals=n_refusals,
            )

            return SamplingResult(
                samples=samples,
                n_samples=n_samples,
                n_valid=n_valid,
                n_refusals=n_refusals,
                seed=seed,
            )

        except Exception as e:
            log.error("generation_failed", query_id=query.query_id, error=str(e))
            raise GenerationError(
                f"HF generation failed: {e}",
                backend="hf",
                hint="Check model availability and GPU memory.",
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(GenerationError),
    )
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 96,
        seed: int | None = None,
    ) -> str:
        """Generate a single response.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 for greedy)
            max_new_tokens: Maximum tokens to generate
            seed: Optional random seed

        Returns:
            Generated text

        Raises:
            GenerationError: If generation fails
        """
        # Set seed if provided
        if seed is not None:
            set_seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.llm.max_context,
        )
        inputs = {k: v.to(self.hw_info.device) for k, v in inputs.items()}

        # Generation config
        GREEDY_THRESHOLD = 0.0
        FALLBACK_TEMP = 1.0
        gen_config = GenerationConfig(
            do_sample=temperature > GREEDY_THRESHOLD,
            temperature=temperature if temperature > GREEDY_THRESHOLD else FALLBACK_TEMP,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Ensure it's a string (decode returns str for single sequence)
            if isinstance(generated_text, list):
                generated_text = generated_text[0] if generated_text else ""

            return generated_text.strip()

        except Exception as e:
            log.error("generation_failed", error=str(e))
            raise GenerationError(
                f"HF generation failed: {e}",
                backend="hf",
            ) from e

    def close(self) -> None:
        """Clean up resources."""
        log.info("hf_backend_closing")
        # Move model to CPU and clear cache
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("hf_backend_closed")


__all__ = ["HFLLMBackend"]
