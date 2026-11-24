# ADG Algorithm Documentation

This document explains the core algorithm behind **Ask, Don't Guess (ADG)**, a training-free system for ambiguity detection in small instruction-tuned LLMs.

## Table of Contents

1. [Overview](#overview)
2. [The Core Insight](#the-core-insight)
3. [Pipeline Steps](#pipeline-steps)
4. [Decision Rules](#decision-rules)
5. [Variants](#variants)
6. [Implementation Details](#implementation-details)

---

## Overview

ADG wraps a small LLM (1.5B-3B parameters) to decide whether to:
- **ANSWER** directly when the question is clear
- **ASK** a targeted clarifying question when ambiguous
- **ABSTAIN** honestly when the model doesn't have enough information

The system is **training-free** and **inference-time only** — no fine-tuning required, just sampling and semantic clustering.

---

## The Core Insight

**Problem:** High uncertainty can mean two different things:
1. **Ambiguity** — The question has multiple valid interpretations (e.g., "capital of Georgia" → state vs country)
2. **Ignorance** — The model doesn't know the answer to a clear question

**Key Innovation:** Instead of compressing uncertainty into a single number (like semantic entropy), ADG preserves the **structure** of meaning clusters:
- **Few large, coherent, distinct clusters** → Ambiguity → ASK
- **One dominant cluster** → Clear question → ANSWER
- **Many small, scattered clusters** → Ignorance → ABSTAIN

This separation is what distinguishes ADG from prior uncertainty-based methods.

---

## Pipeline Steps

### Step 1: Answer Sampling

Sample ~8 diverse answers from the LLM using:
- **Top-p (nucleus) sampling** with p=0.95
- **Temperature** = 0.7 for diversity
- **Seeded sampling** for reproducibility (seed derived from query ID)

Each sample is parsed to extract:
- The **answer** text
- Detected **refusals** (e.g., "I don't know")
- Detected **hedges** (e.g., "probably", "might be")

```python
# Example samples for "What is the capital of Georgia?"
samples = [
    "The capital of Georgia is Atlanta.",
    "The capital of Georgia is Tbilisi.",
    "Atlanta is the capital of the state of Georgia.",
    "Tbilisi is the capital of the country of Georgia.",
    ...
]
```

### Step 2: Semantic Clustering

Group samples by **meaning**, not by exact text:

1. **Pairwise NLI Scoring:**
   - For each pair of samples (s1, s2), compute bidirectional entailment:
     - `entailment(s1 → s2)` using DeBERTa-MNLI
     - `entailment(s2 → s1)`
   - If both directions show entailment → semantically equivalent
   - **Optimization:** Exact string matches are grouped immediately without NLI

2. **Clustering Algorithm:**
   - **Agglomerative clustering** (default) with NLI scores as similarity
   - Alternative: Greedy bidirectional, Connected components
   - Result: Groups of semantically equivalent answers

```python
# Example clusters for "What is the capital of Georgia?"
clusters = [
    Cluster(
        id=0,
        samples=[sample0, sample2],  # Both say Atlanta
        representative="The capital of Georgia is Atlanta.",
        size=2
    ),
    Cluster(
        id=1,
        samples=[sample1, sample3],  # Both say Tbilisi
        representative="The capital of Georgia is Tbilisi.",
        size=2
    ),
]
```

### Step 3: Feature Extraction

Compute aggregate features from the cluster structure:

| Feature | Definition | Meaning |
|---------|-----------|---------|
| **p1** | Fraction of samples in largest cluster | Consensus strength |
| **p2** | Fraction in second-largest cluster | Competing interpretation |
| **Δ (delta)** | p1 - p2 | Separation between top interpretations |
| **Coverage** | Sum of top-k cluster sizes / total samples | How much is "explained" |
| **n_clusters** | Number of clusters found | Diversity of meanings |
| **refusal_rate** | Fraction of samples that refused | Model confidence |
| **invalid_rate** | Fraction that failed to parse | Generation quality |

```python
# Example features
features = AggregateFeatures(
    p1=0.50,         # 50% say Atlanta
    p2=0.50,         # 50% say Tbilisi
    delta=0.00,      # Perfect tie
    coverage=1.0,    # All samples in top 2 clusters
    n_clusters=2,    # Two distinct interpretations
    refusal_rate=0.0,
    invalid_rate=0.0
)
```

### Step 4: Decision Engine

Apply an **ordered rule list** with calibrated thresholds:

```python
# Ordered decision rules
1. If refusal_rate > 0.3 → ABSTAIN (model refusing)
2. If invalid_rate > 0.3 → ABSTAIN (generation failed)
3. If ask_score > threshold → ASK (ambiguous)
4. If answer_score > threshold → ANSWER (clear)
5. Otherwise → ABSTAIN (default)
```

**Scoring functions:**

```python
# ASK score: Look for multiple strong interpretations
def ask_score(features):
    if features.n_clusters < 2:
        return 0.0
    # High when: p1 and p2 both substantial, delta small (tie)
    return min(features.p1, features.p2) * (1.0 - features.delta)

# ANSWER score: Look for one dominant interpretation
def answer_score(features):
    # High when: p1 large, delta large (clear winner)
    return features.p1 * features.delta
```

Decision also includes:
- **Confidence scores** (normalized) for each action
- **Decision trace** showing which rule fired
- **Rule name** for interpretability

### Step 5: Clarification Generation

When ASK is chosen, generate a targeted question from the clusters:

**Template Mode (default):**
```python
# Extract representative answers from top clusters
interpretations = [
    "Atlanta (the US state)",
    "Tbilisi (the country)"
]

# Build clarifying question
question = "Do you mean Georgia the US state or Georgia the country?"
options = ["the US state", "the country"]
```

**LLM Mode (optional):**
- Use the LLM to generate a natural clarifying question
- Provide cluster representatives as context
- More flexible but requires additional LLM call

### Step 6: Multi-Turn Resolution

After the user replies:

1. **Resolve interpretation** using ReplyResolver:
   - Numeric choice ("1", "2")
   - Exact match ("the US state")
   - Fuzzy matching with similarity threshold
   - Fallback to asking again if ambiguous

2. **Answer for chosen interpretation:**
   - Re-sample conditioned on the chosen interpretation
   - Or return the representative answer from that cluster

---

## Decision Rules

### ADG-Core Decision Rules

The core decision engine uses an ordered rule list:

| Rule | Condition | Action | Rationale |
|------|-----------|--------|-----------|
| **abstain_refusal** | refusal_rate > 0.3 | ABSTAIN | Model explicitly refusing |
| **abstain_invalid** | invalid_rate > 0.3 | ABSTAIN | Generation quality too low |
| **ask** | ask_score > threshold | ASK | Multiple interpretations detected |
| **answer** | answer_score > threshold | ANSWER | One clear interpretation |
| **fallback** | (default) | ABSTAIN | Uncertain case |

**Threshold Calibration:**
- Thresholds are calibrated on a held-out calibration set
- Optimized for F1 score or other specified metrics
- Stored in `results/calibration/` for reproducibility

### Confidence Scores

For interpretability, ADG also computes normalized confidence scores:

```python
# Raw scores for each action
raw_scores = {
    "ANSWER": answer_score(features),
    "ASK": ask_score(features),
    "ABSTAIN": max(refusal_rate, invalid_rate)
}

# Normalize to [0, 1] summing to 1.0
confidence_scores = normalize(raw_scores)

# Example output
{
    "ANSWER": 0.15,
    "ASK": 0.75,    # Highest - ASK chosen
    "ABSTAIN": 0.10
}
```

---

## Variants

### ADG-Core (Proposal-Faithful)

**What:** Decision based solely on answer-cluster structure
- Uses p1, p2, delta, coverage, n_clusters from answer samples
- No explicit interpretation annotation

**When to use:** Baseline comparison, proposal-faithful results

### ADG+IC (with Interpretation Channel)

**What:** Augments each sample with its assumed interpretation
- Samples include both answer AND the interpretation assumed
- Can directly distinguish "different answers for different interpretations" (ambiguity) from "conflicting answers under same interpretation" (ignorance)

**How it works:**
```python
# Enhanced sample format
Sample(
    interpretation="Georgia the US state",
    answer="The capital is Atlanta."
)

Sample(
    interpretation="Georgia the country",
    answer="The capital is Tbilisi."
)
```

**Additional feature:**
```python
# Interpretation divergence
divergence = compute_interpretation_divergence(samples)
# High divergence → multiple interpretations → more likely ASK
```

**When to use:** Best performance, real-world deployment

**Tradeoff:** Requires slightly more complex prompting, minor token overhead

---

## Implementation Details

### NLI Model

**Model:** `microsoft/deberta-large-mnli` (default)
- Trained on Multi-Genre NLI dataset
- 3-way classification: entailment, neutral, contradiction
- Bidirectional scoring for semantic equivalence

**Alternative:** `microsoft/deberta-v3-base` (smaller, faster)

### Clustering Algorithms

Three algorithms implemented:

1. **Agglomerative (default):**
   - Bottom-up hierarchical clustering
   - Linkage: average NLI score
   - Threshold: significance_rho (default 0.25)

2. **Greedy Bidirectional:**
   - Faster, greedy approach
   - Merges pairs with highest bidirectional entailment

3. **Connected Components:**
   - Graph-based: edge if bidirectional entailment > threshold
   - Finds connected components

### Sampling Parameters

Default parameters (configurable):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| n_samples | 8 | Number of answers to sample |
| temperature | 0.7 | Diversity control |
| top_p | 0.95 | Nucleus sampling parameter |
| max_new_tokens | 256 | Maximum answer length |

### Seed Derivation

For reproducibility:
```python
# Deterministic seed from query ID
query_seed = derive_seed(query_id, base_seed=0)

# Each sample gets a unique seed
sample_seeds = [
    derive_seed(query_id, base_seed=i)
    for i in range(n_samples)
]
```

### Caching

**NLI scores** are cached by content hash:
- Avoids redundant inference on identical sample pairs
- Persistent disk cache with diskcache library
- Significantly speeds up repeated evaluations

**LLM samples** can be cached (optional):
- For deterministic models with same seed
- Useful for development/debugging
- Disabled by default in experiments

---

## Performance Characteristics

### Computational Cost

For one query with n=8 samples:

| Component | Operations | Typical Time |
|-----------|-----------|--------------|
| **LLM Sampling** | 8 forward passes | ~1-2s (GPU), ~10-30s (CPU) |
| **NLI Scoring** | O(n²) = 28 pairs (worst case) | ~0.5-1s |
| **Clustering** | O(n² log n) | <0.01s |
| **Decision** | O(1) | <0.001s |
| **Clarification** | 1 LLM call (LLM mode only) | ~0.5s |

**Total:** ~2-3s per query on GPU, ~11-32s on CPU (8-core)

### Memory Requirements

| Component | Memory |
|-----------|--------|
| **LLM (Qwen2.5-1.5B fp16)** | ~3 GB VRAM |
| **NLI (DeBERTa-large)** | ~1.5 GB VRAM |
| **Total (fp16)** | ~5 GB VRAM |
| **Total (4-bit quant)** | ~2 GB VRAM |

**Minimum:** 8 GB VRAM (comfortable), 4 GB VRAM (4-bit quant, tight)

---

## Comparison with Semantic Entropy

| Aspect | Semantic Entropy | ADG |
|--------|------------------|-----|
| **Clustering** | Same (NLI-based) | Same (NLI-based) |
| **Uncertainty measure** | Entropy over clusters (1 scalar) | Cluster structure (p1, p2, Δ, coverage) |
| **Decision** | Threshold on entropy | Rule list on structured features |
| **Actions** | Answer / Abstain (2-way) | Answer / Ask / Abstain (3-way) |
| **Clarification** | None | Targeted questions from clusters |
| **Ambiguity vs Ignorance** | Cannot distinguish | Explicitly distinguishes |

**Key difference:** Semantic entropy compresses cluster structure into entropy. ADG preserves structure to separate ambiguity from ignorance.

---

## References

- **Semantic Entropy:** Kuhn, Gal & Farquhar. "Semantic Uncertainty." ICLR 2023.
- **NLI Model:** He et al. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR 2021.
- **AmbigQA:** Min et al. "AmbigQA: Answering Ambiguous Open-domain Questions." EMNLP 2020.
- **CLAMBER:** Zhang et al. "CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs." ACL 2024.

---

## See Also

- [SPEC.md](SPEC.md) - Full system specification
- [reproduction.md](reproduction.md) - How to reproduce experiments
- [limitations.md](limitations.md) - Known limitations
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - Implementation status
