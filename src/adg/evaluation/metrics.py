"""Evaluation metrics.

Per spec §12.1: Asking P/R/F1, 3-way, answer correctness, etc.
"""

import re
import string
from typing import Literal

from pydantic import BaseModel
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from adg.core.logging import get_logger

log = get_logger(__name__)


class AskingMetrics(BaseModel):
    """Asking precision/recall/F1 metrics.

    Args:
        precision: Asking precision (correctness of ASK decisions)
        recall: Asking recall (coverage of ambiguous questions)
        f1: F1 score
        support_ask: Number of gold ambiguous examples
        support_total: Total examples
        auroc: AUROC of ask score (if available)
        auprc: AUPRC of ask score (if available)
    """

    precision: float
    recall: float
    f1: float
    support_ask: int
    support_total: int
    auroc: float | None = None
    auprc: float | None = None


class ThreeWayMetrics(BaseModel):
    """3-way classification metrics.

    Args:
        confusion: Confusion matrix (3x3)
        accuracy: Overall accuracy
        macro_f1: Macro-averaged F1
        per_class_f1: F1 per class (ANSWER, ASK, ABSTAIN)
        unnecessary_ask_rate: Rate of ASK on clear questions
        abstain_vs_ask_confusion: Rate of ABSTAIN when should ASK
    """

    confusion: list[list[int]]  # 3x3 matrix
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    unnecessary_ask_rate: float
    abstain_vs_ask_confusion: float


class AnswerMetrics(BaseModel):
    """Answer correctness metrics.

    Args:
        exact_match: Exact match accuracy
        token_f1: Token-level F1
        support: Number of answered examples
    """

    exact_match: float
    token_f1: float
    support: int


def compute_asking_metrics(
    predicted_actions: list[Literal["ANSWER", "ASK", "ABSTAIN"]],
    gold_is_ambiguous: list[bool],
    ask_scores: list[float] | None = None,
) -> AskingMetrics:
    """Compute asking precision/recall/F1.

    Per spec §12.1: Primary metrics for ASK decisions.

    Args:
        predicted_actions: Predicted actions
        gold_is_ambiguous: Gold ambiguity labels
        ask_scores: Optional continuous ask scores for AUROC

    Returns:
        Asking metrics
    """
    if len(predicted_actions) != len(gold_is_ambiguous):
        raise ValueError("Length mismatch between predictions and labels")

    # Binary labels: ASK vs not ASK
    pred_ask = [1 if action == "ASK" else 0 for action in predicted_actions]
    gold_ask = [1 if amb else 0 for amb in gold_is_ambiguous]

    # Precision/recall/F1
    precision = precision_score(gold_ask, pred_ask, zero_division=0.0)
    recall = recall_score(gold_ask, pred_ask, zero_division=0.0)
    f1 = f1_score(gold_ask, pred_ask, zero_division=0.0)

    support_ask = sum(gold_ask)
    support_total = len(gold_ask)

    # AUROC/AUPRC if scores provided
    auroc = None
    auprc = None

    if ask_scores is not None and len(ask_scores) == len(gold_ask):
        try:
            auroc = roc_auc_score(gold_ask, ask_scores)

            # AUPRC: area under precision-recall curve
            # Using sklearn's curve functions
            fpr, tpr, _ = roc_curve(gold_ask, ask_scores)
            # For PR curve, we'd use precision_recall_curve
            # but sklearn doesn't have direct AUPRC
            # ponytail: skip AUPRC for now, can add later
        except ValueError as e:
            log.warning("auroc_computation_failed", error=str(e))

    return AskingMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support_ask=support_ask,
        support_total=support_total,
        auroc=float(auroc) if auroc is not None else None,
        auprc=auprc,
    )


def compute_three_way_metrics(
    predicted_actions: list[Literal["ANSWER", "ASK", "ABSTAIN"]],
    gold_actions: list[Literal["ANSWER", "ASK", "ABSTAIN"]],
    gold_is_ambiguous: list[bool],
) -> ThreeWayMetrics:
    """Compute 3-way classification metrics.

    Per spec §12.1: Confusion matrix, macro F1, unnecessary ask rate.

    Args:
        predicted_actions: Predicted actions
        gold_actions: Gold actions (computed per §9.4)
        gold_is_ambiguous: Gold ambiguity labels

    Returns:
        3-way metrics
    """
    if len(predicted_actions) != len(gold_actions):
        raise ValueError("Length mismatch")

    # Confusion matrix
    labels = ["ANSWER", "ASK", "ABSTAIN"]
    cm = confusion_matrix(
        gold_actions,
        predicted_actions,
        labels=labels,
    )

    # Overall accuracy
    correct = sum(1 for p, g in zip(predicted_actions, gold_actions, strict=False) if p == g)
    accuracy = correct / len(gold_actions) if gold_actions else 0.0

    # Macro F1
    macro_f1 = f1_score(
        gold_actions,
        predicted_actions,
        labels=labels,
        average="macro",
        zero_division=0.0,
    )

    # Per-class F1
    per_class_f1_scores = f1_score(
        gold_actions,
        predicted_actions,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    per_class_f1 = {
        label: float(score) for label, score in zip(labels, per_class_f1_scores, strict=False)
    }

    # Unnecessary ask rate: ASK on clear (not ambiguous) questions
    clear_indices = [i for i, amb in enumerate(gold_is_ambiguous) if not amb]
    unnecessary_asks = sum(1 for i in clear_indices if predicted_actions[i] == "ASK")
    unnecessary_ask_rate = unnecessary_asks / len(clear_indices) if clear_indices else 0.0

    # ABSTAIN vs ASK confusion on clear-but-unknown
    # (should ABSTAIN but decided ASK)
    should_abstain = [
        i
        for i, (g, amb) in enumerate(zip(gold_actions, gold_is_ambiguous, strict=False))
        if g == "ABSTAIN" and not amb
    ]
    abstain_ask_confusion = sum(1 for i in should_abstain if predicted_actions[i] == "ASK")
    abstain_vs_ask_confusion = (
        abstain_ask_confusion / len(should_abstain) if should_abstain else 0.0
    )

    return ThreeWayMetrics(
        confusion=cm.tolist(),
        accuracy=accuracy,
        macro_f1=float(macro_f1),
        per_class_f1=per_class_f1,
        unnecessary_ask_rate=unnecessary_ask_rate,
        abstain_vs_ask_confusion=abstain_vs_ask_confusion,
    )


def normalize_answer(answer: str) -> str:
    """Normalize answer for matching.

    Per spec §12.1: SQuAD-style normalization.

    Args:
        answer: Answer text

    Returns:
        Normalized answer
    """
    # Lowercase
    answer = answer.lower()

    # Remove articles
    for article in ["a", "an", "the"]:
        answer = re.sub(rf"\b{article}\b", " ", answer)

    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))

    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer


def score_answer(
    predicted: str,
    gold_answers: list[str],
) -> tuple[float, float]:
    """Score predicted answer against gold answers.

    Per spec §12.1: EM and token-F1.

    Args:
        predicted: Predicted answer
        gold_answers: List of acceptable gold answers

    Returns:
        Tuple of (exact_match, token_f1)
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = [normalize_answer(g) for g in gold_answers]

    # Exact match: 1 if any gold matches
    exact_match = 1.0 if pred_norm in gold_norm else 0.0

    # Token F1: best F1 against any gold answer
    best_f1 = 0.0
    pred_tokens = set(pred_norm.split())

    for gold in gold_norm:
        gold_tokens = set(gold.split())

        if not pred_tokens and not gold_tokens:
            f1 = 1.0
        elif not pred_tokens or not gold_tokens:
            f1 = 0.0
        else:
            overlap = pred_tokens & gold_tokens
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(gold_tokens)

            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        best_f1 = max(best_f1, f1)

    return exact_match, best_f1


__all__ = [
    "AskingMetrics",
    "ThreeWayMetrics",
    "AnswerMetrics",
    "compute_asking_metrics",
    "compute_three_way_metrics",
    "normalize_answer",
    "score_answer",
]
