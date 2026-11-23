"""HuggingFace NLI scorer using DeBERTa.

Per spec §4.4: bidirectional entailment scoring for semantic equivalence.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from adg.config.schema import AppConfig
from adg.core.device import detect_hardware, get_torch_dtype
from adg.core.errors import NLIModelError
from adg.core.logging import get_logger
from adg.core.types import PairwiseRelations, Sample

log = get_logger(__name__)


class HFNLIScorer:
    """HuggingFace NLI scorer using DeBERTa-MNLI.

    Computes bidirectional entailment for semantic equivalence.
    Per spec §4.4: S[i][j] = P(i→j) * P(j→i)
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize NLI scorer and load model.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_id = config.nli.model_id or config.nli.config
        self.threshold = 0.5  # Default entailment threshold for equivalence
        self.batch_size = config.nli.batch_size
        self.hw_info = detect_hardware()

        log.info(
            "hf_nli_initializing",
            model_id=self.model_id,
            device=self.hw_info.device,
            threshold=self.threshold,
        )

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            torch_dtype = get_torch_dtype(self.hw_info.dtype)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
            )
            self.model.to(self.hw_info.device)
            self.model.eval()

            # DeBERTa label mapping: 0=contradiction, 1=neutral, 2=entailment
            self.label_mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}

        except Exception as e:
            log.error("nli_model_load_failed", model_id=self.model_id, error=str(e))
            raise NLIModelError(
                f"Failed to load NLI model {self.model_id}: {e}",
                hint="Check model ID and network connection.",
            ) from e

        log.info("hf_nli_initialized", model_id=self.model_id)

    def score_pairs(
        self,
        samples: list[Sample],
        query_text: str,
        input_format: str = "qa",
    ) -> PairwiseRelations:
        """Compute NLI scores for all pairs of samples.

        Args:
            samples: List of Sample objects to compare
            query_text: Original query (for contextualization)
            input_format: "qa", "answer_only", or "interpretation_answer"

        Returns:
            PairwiseRelations with symmetric scores matrix S[i][j]
        """
        n = len(samples)
        log.debug(
            "hf_nli_scoring",
            n_samples=n,
            input_format=input_format,
            query_text=query_text[:50],
        )

        # Format texts based on input_format
        texts = self._format_texts(samples, query_text, input_format)

        # Initialize score matrix (symmetric)
        scores = [[0.0 for _ in range(n)] for _ in range(n)]
        equivalence = [[False for _ in range(n)] for _ in range(n)]

        # Diagonal is always 1.0 (self-equivalence)
        for i in range(n):
            scores[i][i] = 1.0
            equivalence[i][i] = True

        # Collect all pairs to score
        pairs_to_score: list[tuple[int, int, str, str]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_to_score.append((i, j, texts[i], texts[j]))

        # Score in batches
        pair_scores = self._score_batch_bidirectional(pairs_to_score)

        # Fill matrix
        for (i, j, _, _), score in zip(pairs_to_score, pair_scores, strict=False):
            scores[i][j] = score
            scores[j][i] = score  # Symmetric

            is_equiv = score >= self.threshold
            equivalence[i][j] = is_equiv
            equivalence[j][i] = is_equiv

        n_pairs_total = n * (n - 1) // 2
        n_pairs_computed = len(pairs_to_score)

        log.info(
            "hf_nli_scoring_complete",
            n_samples=n,
            n_pairs_computed=n_pairs_computed,
            n_pairs_total=n_pairs_total,
        )

        return PairwiseRelations(
            scores=scores,
            equivalence=equivalence,
            n_pairs_computed=n_pairs_computed,
            n_pairs_total=n_pairs_total,
            n_guard_applied=0,
            mode="soft",
        )

    def score_single_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, float, float]:
        """Score a single premise-hypothesis pair.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            (entailment_prob, neutral_prob, contradiction_prob)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.hw_info.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Extract probabilities (DeBERTa: 0=contra, 1=neutral, 2=entail)
        contradiction_prob = probs[0].item()
        neutral_prob = probs[1].item()
        entailment_prob = probs[2].item()

        return (entailment_prob, neutral_prob, contradiction_prob)

    def close(self) -> None:
        """Clean up resources."""
        log.info("hf_nli_closing")
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("hf_nli_closed")

    def _format_texts(
        self,
        samples: list[Sample],
        query_text: str,
        input_format: str,
    ) -> list[str]:
        """Format sample texts based on input_format.

        Args:
            samples: List of samples
            query_text: Original query
            input_format: Format specification

        Returns:
            List of formatted texts
        """
        texts = []
        for sample in samples:
            if input_format == "answer_only":
                text = sample.answer
            elif input_format == "interpretation_answer":
                text = f"{sample.interpretation} {sample.answer}"
            else:  # "qa"
                text = f"Q: {query_text} A: {sample.answer}"
            texts.append(text)
        return texts

    def _score_batch_bidirectional(
        self,
        pairs: list[tuple[int, int, str, str]],
    ) -> list[float]:
        """Score pairs with bidirectional entailment.

        Args:
            pairs: List of (i, j, text_i, text_j) tuples

        Returns:
            List of bidirectional scores S[i][j] = P(i→j) * P(j→i)
        """
        # Prepare forward and backward pairs
        forward_pairs = [(text_i, text_j) for _, _, text_i, text_j in pairs]
        backward_pairs = [(text_j, text_i) for _, _, text_i, text_j in pairs]

        # Score forward
        forward_scores = self._score_unidirectional_batch(forward_pairs)

        # Score backward
        backward_scores = self._score_unidirectional_batch(backward_pairs)

        # Bidirectional: S[i][j] = P(i→j) * P(j→i)
        bidirectional_scores = [
            forward * backward
            for forward, backward in zip(forward_scores, backward_scores, strict=False)
        ]

        return bidirectional_scores

    def _score_unidirectional_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Score pairs unidirectionally (premise → hypothesis).

        Args:
            pairs: List of (premise, hypothesis) tuples

        Returns:
            List of entailment probabilities
        """
        all_scores = []

        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]

            # Tokenize batch
            premises = [p for p, _ in batch_pairs]
            hypotheses = [h for _, h in batch_pairs]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.hw_info.device) for k, v in inputs.items()}

            # Forward pass
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)

                    # Extract entailment probabilities (index 2 for DeBERTa)
                    entailment_probs = probs[:, 2].cpu().tolist()
                    all_scores.extend(entailment_probs)

            except Exception as e:
                log.error("nli_batch_failed", batch_size=len(batch_pairs), error=str(e))
                # Fallback: return 0.0 for failed batch
                all_scores.extend([0.0] * len(batch_pairs))

        return all_scores


__all__ = ["HFNLIScorer"]
