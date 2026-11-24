# Known Limitations

This document describes the known limitations, failure modes, and boundaries of the Ask, Don't Guess (ADG) system.

## Table of Contents

1. [Scope Limitations](#scope-limitations)
2. [Algorithmic Limitations](#algorithmic-limitations)
3. [Model Limitations](#model-limitations)
4. [Performance Limitations](#performance-limitations)
5. [Dataset Limitations](#dataset-limitations)
6. [Implementation Limitations](#implementation-limitations)
7. [Known Failure Modes](#known-failure-modes)

---

## Scope Limitations

### What ADG Does

ADG is designed for:
- **Short-form factoid questions** (< 100 words)
- **Open-domain questions** where ambiguity might exist
- **Small LLMs** (1.5B-3B parameters) where uncertainty is high
- **Inference-time detection** without training

### What ADG Does NOT Do

**1. Long-form or complex queries**
- Multi-part questions requiring reasoning chains
- Questions requiring multiple paragraphs to answer
- Code generation, creative writing, etc.

**Example failure:**
```
Query: "Explain the causes and consequences of World War I,
        including the political, economic, and social factors."
Problem: Semantic clustering assumes short, factoid answers.
         Long explanations may cluster poorly.
```

**2. Subjective or opinion-based questions**
- Questions with no ground truth
- Preference-based queries
- Creative or open-ended prompts

**Example failure:**
```
Query: "What's the best programming language?"
Problem: Multiple valid answers (Python, Rust, etc.) but
         they're opinions, not disambiguable interpretations.
```

**3. Real-time or personal information**
- Current events (model knowledge cutoff)
- User-specific information (calendar, emails, etc.)
- Dynamic information (stock prices, weather)

**4. Multi-turn conversational context**
- ADG processes each query independently
- No conversation memory beyond one clarification
- Cannot handle "what about X" follow-ups

---

## Algorithmic Limitations

### 1. Sample Efficiency

**Limitation:** ADG requires ~8 LLM samples per query (vs 1 for greedy decoding)

**Impact:**
- 8x computational cost
- 8x latency (if samples are sequential)
- 8x API cost (for paid models)

**Mitigation:**
- Use vLLM or batching for parallel sampling (Linux + CUDA)
- Reduce to 4 samples for latency-critical applications
- Cache samples for repeated queries

### 2. NLI Model Dependency

**Limitation:** Quality depends on NLI model's understanding of semantic equivalence

**Failure cases:**
```python
# NLI model may incorrectly cluster:
Sample 1: "Paris is the capital."
Sample 2: "The capital is Paris."
# → Correctly clustered (entailment)

Sample 1: "Mercury is a planet."
Sample 2: "Mercury is a metal."
# → May incorrectly cluster if NLI model is confused
```

**Mitigation:**
- Use high-quality NLI model (DeBERTa-large-MNLI)
- Fallback to string matching for identical answers
- Manual inspection of clustering on dev set

### 3. Threshold Sensitivity

**Limitation:** Decision thresholds must be calibrated per dataset/domain

**Problem:**
- Thresholds calibrated on AmbigQA may not transfer to medical QA
- Different domains have different ambiguity patterns
- Requires labeled calibration data

**Mitigation:**
- Provide sensible defaults (ask_threshold=0.4, answer_threshold=0.5)
- Document threshold calibration process
- Support per-domain calibration

### 4. Binary Ambiguity Assumption

**Limitation:** ADG works best with 2-way ambiguities (Georgia: state vs country)

**Failure case:**
```
Query: "What is Java?"
Possible interpretations:
  1. Java programming language
  2. Java island in Indonesia
  3. Java coffee
  4. Java man (fossil)

Problem: With 4+ interpretations, clusters may be small and scattered.
         ADG may ABSTAIN instead of ASK.
```

**Mitigation:**
- Increase sample count for multi-way ambiguities
- Tune thresholds to handle more interpretations
- Limit clarifying questions to top 3 interpretations

### 5. Implicit Ambiguity

**Limitation:** ADG only detects ambiguity if the LLM samples reveal it

**Failure case:**
```
Query: "Who is the president?"
Problem: If LLM always assumes "current US president" without
         considering other countries or time periods, ADG
         won't detect the ambiguity.
```

**Mitigation:**
- Use diverse sampling (temperature=0.7, top_p=0.95)
- Interpretation channel (ADG+IC) makes interpretations explicit
- Cannot fully solve this — depends on base LLM

---

## Model Limitations

### 1. Base LLM Quality

**Limitation:** ADG cannot fix a bad base model

**Problem:**
- If base LLM generates nonsense, ADG will ABSTAIN (correct)
- If base LLM is biased, ADG will reflect that bias
- If base LLM lacks knowledge, ADG will abstain

**Example:**
```
Query: "What is the capital of Bhutan?"
Base LLM: Doesn't know → generates 8 different wrong answers
ADG: Correctly ABSTAINs (scattered clusters)
```

**Mitigation:**
- Use high-quality base models (Qwen2.5, Llama-3, etc.)
- ADG helps by being honest about uncertainty
- Cannot create knowledge that isn't in the model

### 2. Model Size Limits

**Tested on:** 1.5B-3B parameter models

**Limitations:**
- Very small models (<1B) may be too weak
- Very large models (>7B) may not need ADG (already confident)
- Optimal range: 1.5B-3B for practical ambiguity detection

### 3. NLI Model Limitations

**Current:** DeBERTa-large-MNLI (304M parameters)

**Limitations:**
- Trained on English text only
- May struggle with technical/domain-specific language
- May miss subtle semantic differences
- Bidirectional entailment is not perfect semantic equivalence

---

## Performance Limitations

### 1. Latency

**Typical latency per query:**
- **Mock mode:** <0.1s
- **GPU (fp16):** 2-3s
- **GPU (4-bit):** 3-5s
- **CPU (8-core):** 15-30s

**Bottlenecks:**
- LLM sampling: ~80% of time
- NLI scoring: ~15% of time
- Clustering/decision: <5% of time

**Not suitable for:**
- Real-time chat (<500ms latency)
- High-throughput APIs (>100 QPS without batching)

### 2. Memory

**Minimum VRAM:**
- **fp16:** 5 GB (LLM + NLI)
- **4-bit:** 2 GB (quantized)
- **CPU:** No GPU required, but slow

**Scaling:**
- Cannot fit models >3B on 8 GB GPUs without quantization
- Multi-GPU requires vLLM (Linux only)

### 3. Cost

**Per query cost (8 samples):**

| Model | Cost (API) | Latency |
|-------|-----------|---------|
| GPT-4o-mini | $0.0024 | ~1-2s |
| Local Qwen2.5-1.5B | $0 (amortized) | ~2-3s |

**For 10,000 queries:**
- API: $24
- Local: $0 (after setup cost)

---

## Dataset Limitations

### 1. Evaluation Datasets

**AmbigQA:**
- English only
- Wikipedia-based (may not reflect real user queries)
- Ambiguity is annotated, but coverage may be incomplete
- Some questions have >2 interpretations (hard for binary detection)

**CLAMBER:**
- Synthetic ambiguity (created by researchers)
- May not reflect natural ambiguity patterns
- Limited size

**NQ-open:**
- Not explicitly annotated for ambiguity
- Used for ABSTAIN evaluation (questions model doesn't know)
- May contain implicit ambiguities

### 2. Language

**Current support:** English only

**Limitations:**
- LLM (Qwen2.5) is multilingual but optimized for English
- NLI model (DeBERTa-MNLI) is English-only
- Evaluation datasets are English-only

**Future work:** Multilingual NLI models exist (XLM-R-based) but not tested

### 3. Domain Coverage

**Tested on:** Open-domain factoid questions

**Untested:**
- Medical/legal/scientific QA
- Code-related questions
- Math/reasoning questions
- Conversational queries

---

## Implementation Limitations

### 1. Platform Support

**Fully supported:**
- Linux (Ubuntu 20.04+, CUDA 11.8+)
- macOS (Intel and Apple Silicon)
- Windows (CPU mode, limited GPU support)

**Limited support:**
- Windows + CUDA (works but less tested)
- vLLM (Linux + CUDA only)

### 2. Hardware Support

**Tested:**
- NVIDIA GPUs (3080, 3090, 4090, A100)
- Apple Silicon (M1/M2 with MPS)
- CPU (Intel x86-64, ARM64)

**Not tested:**
- AMD GPUs (ROCm)
- Intel GPUs (Arc)
- Other accelerators (TPU, etc.)

### 3. Missing Features

**Not yet implemented:**
- Streaming responses (all samples must complete before decision)
- Async API (FastAPI endpoints exist but experiments are synchronous)
- Web frontend (Gradio demo only, no Next.js dashboard yet)
- DPO training extension (spec includes it, not implemented)

---

## Known Failure Modes

### 1. Over-Asking

**Symptom:** ADG asks clarifying questions for clear queries

**Causes:**
- Thresholds too low
- Base LLM generates diverse answers even for clear questions
- NLI model clusters answers too aggressively

**Example:**
```
Query: "What is 2+2?"
Base LLM: Sometimes says "4", sometimes "four", "2+2=4"
ADG: May incorrectly ask for clarification
```

**Solution:**
- Tune thresholds on calibration set
- Use higher answer_threshold
- Add exact match deduplication

### 2. Under-Asking

**Symptom:** ADG answers directly when it should ask

**Causes:**
- Thresholds too high
- Base LLM doesn't explore different interpretations
- NLI model merges different meanings

**Example:**
```
Query: "What is the capital of Georgia?"
Base LLM: Always says "Atlanta" (US bias)
ADG: Answers "Atlanta" without asking
```

**Solution:**
- Use interpretation channel (ADG+IC)
- Increase sample diversity (higher temperature)
- Cannot fully solve LLM bias

### 3. Unhelpful Abstentions

**Symptom:** ADG abstains when it could answer or ask

**Causes:**
- Base LLM generates scattered, low-quality answers
- Too many refusals in samples
- Clusters are too small/incoherent

**Example:**
```
Query: "What is quantum computing?"
Base LLM: Generates 8 different vague explanations
ADG: Abstains (correctly, but user gets no help)
```

**Solution:**
- This is often correct behavior
- Could fallback to "best effort" answer with disclaimer
- Need better base LLM for complex topics

### 4. Poor Clarifying Questions

**Symptom:** Clarifying question is confusing or generic

**Causes:**
- Clusters have vague representatives
- Template-based generation is too rigid
- LLM-based generation fails

**Example:**
```
Query: "Tell me about Mercury"
Clusters: "planet", "metal", "Roman god"
Bad question: "Which Mercury do you mean?"
Better: "Do you mean Mercury the planet, the chemical element, or the Roman god?"
```

**Solution:**
- Use LLM-based clarification generation
- Improve prompt templates
- Extract clearer cluster labels

### 5. Multi-turn Confusion

**Symptom:** After clarification, final answer doesn't match chosen interpretation

**Causes:**
- Reply resolver picks wrong cluster
- LLM doesn't follow instruction to answer for specific interpretation
- Context lost between turns

**Example:**
```
Turn 1: "What is the capital of Georgia?"
ADG asks: "US state or country?"
User: "country"
ADG answers: "Atlanta" (wrong - should be Tbilisi)
```

**Solution:**
- Improve reply resolution logic
- Use more explicit conditioning in final answer prompt
- Verify answer matches interpretation

---

## Mitigation Strategies

### For Users

1. **Be specific:** Add context to avoid ambiguity
   - Bad: "What is Java?"
   - Good: "What is the Java programming language?"

2. **Clarify up front:** If you know a question might be ambiguous
   - Add: "I mean the country, not the state"

3. **Expect abstentions:** If the model doesn't know, honesty is valuable
   - Abstention > confident wrong answer

### For Developers

1. **Calibrate thresholds:** Use domain-specific calibration data
2. **Monitor metrics:** Track ask/abstain rates in production
3. **Use ADG+IC:** Interpretation channel improves ambiguity detection
4. **Tune sampling:** Adjust n_samples, temperature for your use case
5. **Test thoroughly:** Validate on examples from your domain

### For Researchers

1. **Report both ADG-Core and ADG+IC:** Compare proposal-faithful and enhanced versions
2. **Document failures:** Negative results are valuable
3. **Test on new domains:** Current evaluation is limited
4. **Compare fairly:** Use equal sample budgets across baselines
5. **Statistical testing:** Use bootstrap CIs and significance tests

---

## Future Work

**Algorithmic improvements:**
- Better handling of multi-way ambiguities (>2 interpretations)
- Hierarchical clustering for nested ambiguities
- Active learning for threshold calibration
- Confidence calibration for scores

**System improvements:**
- Streaming responses
- Async/parallel processing
- Better clarifying question generation
- Multi-turn conversation context

**Evaluation:**
- More diverse datasets
- Multilingual support
- Domain-specific benchmarks
- Human evaluation at scale

**Extensions:**
- DPO training for better ambiguity detection
- Integration with retrieval (RAG)
- Explanation generation for decisions
- User feedback loop

---

## Reporting Issues

If you encounter a limitation not listed here:

1. **Check if it's a bug:** See [Troubleshooting](reproduction.md#troubleshooting)
2. **Document the issue:** Minimal reproduction example
3. **Report it:** GitHub Issues with label `limitation`
4. **Include:**
   - Query text
   - Expected behavior
   - Actual behavior
   - Configuration used
   - System info (OS, GPU, model versions)

---

## See Also

- [algorithm.md](algorithm.md) - How ADG works
- [reproduction.md](reproduction.md) - How to run experiments
- [SPEC.md](SPEC.md) - Full specification
- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - What's implemented
