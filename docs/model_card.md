# Model Card: Ask, Don't Guess (ADG)

## Model Details

### Basic Information

- **Model Name:** Ask, Don't Guess (ADG)
- **Model Type:** Inference-time ambiguity detection system
- **Version:** 0.1.0
- **Date:** November 2024
- **Organization:** ADG Team (University AI Course Project)
- **License:** Apache 2.0
- **Contact:** GitHub Issues - https://github.com/adg-team/ask-dont-guess/issues

### Model Description

ADG is a training-free system that wraps small instruction-tuned LLMs (1.5B-3B parameters) to detect query ambiguity and decide whether to answer directly, ask a clarifying question, or abstain honestly.

**Components:**
1. **Base LLM:** Small instruction-tuned model for answer generation (default: Qwen2.5-1.5B-Instruct)
2. **NLI Model:** Semantic equivalence scorer for clustering (default: DeBERTa-large-MNLI)
3. **Decision Engine:** Rule-based classifier with calibrated thresholds
4. **Clarification Generator:** Template or LLM-based question generation

**Key Innovation:** Uses semantic cluster structure (not just entropy) to distinguish ambiguity from ignorance.

---

## Intended Use

### Primary Intended Uses

1. **Question-answering systems** where ambiguity might exist
   - Customer support chatbots
   - Educational Q&A platforms
   - Virtual assistants for open-domain queries

2. **Research** on uncertainty quantification and ambiguity detection
   - Benchmarking selective prediction
   - Studying when LLMs should/shouldn't answer
   - Improving human-AI interaction

3. **Development/prototyping** with small LLMs
   - Testing ambiguity handling before deploying large models
   - Understanding model behavior on edge cases
   - Building more reliable AI assistants

### Primary Intended Users

- **Researchers** studying uncertainty, ambiguity, and selective prediction
- **ML Engineers** building Q&A systems with safety requirements
- **Developers** integrating LLMs into production applications
- **Educators** teaching about AI reliability and human-AI interaction

### Out-of-Scope Uses

**NOT intended for:**

1. **Medical, legal, or safety-critical decisions**
   - ADG is a research system, not safety-certified
   - Abstentions may still be overconfident
   - No guarantee of catching all ambiguities

2. **Long-form content generation**
   - Designed for short factoid questions
   - Not suitable for essays, code, creative writing

3. **Multilingual applications** (current version)
   - English only (NLI model limitation)
   - Untested on other languages

4. **Real-time/low-latency applications**
   - 2-30s latency per query (depending on hardware)
   - Not suitable for <500ms requirements

5. **High-stakes decisions without human oversight**
   - Errors can occur (see Limitations)
   - Human review recommended for important queries

---

## Training Data

### Decision Thresholds (Calibration)

**Data:**
- **Source:** Subset of AmbigQA development set
- **Size:** ~500 examples (stratified by gold action)
- **Purpose:** Calibrate decision thresholds (ask_threshold, answer_threshold)

**Process:**
- Grid search over threshold combinations
- Optimize for F1 score (or specified metric)
- 5-fold cross-validation

**Note:** No neural network training involved (training-free system)

### Base Models (Pre-trained, Not Modified)

**LLM (Qwen2.5-1.5B-Instruct):**
- **Training data:** Multilingual web data, books, code
- **Training process:** Pre-training + instruction tuning
- **Provider:** Alibaba Cloud
- **License:** Apache 2.0 (open weights)

**NLI (DeBERTa-large-MNLI):**
- **Training data:** MultiNLI dataset (433k sentence pairs)
- **Training process:** Fine-tuned on NLI task
- **Provider:** Microsoft
- **License:** MIT

---

## Evaluation Data

### Datasets

1. **AmbigQA** (Min et al., EMNLP 2020)
   - **Size:** 10K train, 2K dev
   - **Domain:** Open-domain factoid questions
   - **Ambiguity:** Questions with multiple valid interpretations
   - **Annotations:** Disambiguated question-answer pairs

2. **CLAMBER** (Zhang et al., ACL 2024)
   - **Size:** ~2K examples
   - **Domain:** Synthetic ambiguous information needs
   - **Taxonomy:** 5 ambiguity types

3. **NQ-open** (Kwiatkowski et al., 2019)
   - **Size:** 79K train, 8K dev, 3K test
   - **Domain:** Open-domain questions from Google search
   - **Purpose:** Test ABSTAIN on questions model doesn't know

### Evaluation Protocol

**Metrics:**
- **ASK detection:** Precision, Recall, F1 (on AmbigQA + CLAMBER)
- **3-way classification:** Accuracy (ANSWER/ASK/ABSTAIN)
- **Answer quality:** Exact Match, Token F1 (when answering)
- **Cost:** Average samples, tokens, latency per query

**Baselines:**
- Always Answer
- Ask-If-Unclear (prompt-based)
- Stated Confidence (verbalized 0-100)
- Log-Prob Threshold
- Semantic Entropy

**Statistical Testing:**
- Bootstrap confidence intervals (95%, 10K resamples)
- Paired t-test (p < 0.05 for significance)
- McNemar's test for 3-way classification

---

## Performance

### Expected Results (Planned)

**Note:** Results pending - full experiments not yet run. Expected ranges based on dev set:

| Method | ASK F1 | 3-way Acc | Answer EM | Samples/Query |
|--------|--------|-----------|-----------|---------------|
| Always Answer | 0.00 | ~40% | ~25% | 1 |
| Ask-If-Unclear | ~20% | ~45% | ~20% | 1 |
| Semantic Entropy | ~40% | ~50% | ~24% | 8 |
| **ADG-Core** | ~50-60% | ~55-65% | ~25% | 8 |
| **ADG+IC** | ~60-70% | ~60-70% | ~26% | 8 |

**Hypotheses to test:**
- H1: ADG > baselines on ASK F1 (at equal sample budget)
- H2: ADG reduces unnecessary abstentions vs semantic entropy
- H3: ADG clarifying questions cover gold interpretations
- H4: End-to-end accuracy improves with clarifications
- H5: ADG+IC > ADG-Core on ambiguity detection

**Current Status:** Dev set results support hypotheses, but full evaluation required.

### Known Performance Characteristics

**Latency (per query):**
- GPU (fp16): 2-3s
- GPU (4-bit): 3-5s
- CPU (8-core): 15-30s

**Memory:**
- GPU fp16: 5 GB VRAM
- GPU 4-bit: 2 GB VRAM
- CPU: 16 GB RAM recommended

**Throughput:**
- Single GPU: ~20-30 queries/minute
- With vLLM batching: ~100-200 queries/minute
- CPU: ~2-4 queries/minute

---

## Limitations

### Technical Limitations

1. **Sample Efficiency:** Requires 8x more LLM calls than greedy decoding
2. **Latency:** 2-30s per query (not suitable for real-time)
3. **NLI Dependency:** Quality depends on NLI model's understanding
4. **Threshold Sensitivity:** Needs calibration per dataset/domain
5. **English Only:** Current NLI model is English-only

### Algorithmic Limitations

1. **Implicit Ambiguity:** Only detects ambiguity if LLM samples reveal it
2. **Binary Ambiguity:** Works best with 2-way ambiguities (not 4+)
3. **Cluster Quality:** NLI model may mis-cluster edge cases
4. **Base LLM Quality:** Cannot fix a fundamentally bad base model

### Scope Limitations

1. **Short-form Only:** Designed for factoid questions, not essays
2. **Open-domain:** Tested on general knowledge, not specialized domains
3. **Factual Questions:** Not suitable for subjective/opinion questions
4. **Single-turn:** Limited multi-turn conversation support

**See [limitations.md](limitations.md) for detailed discussion.**

---

## Fairness, Bias, and Ethical Considerations

### Known Biases

1. **Base LLM Bias:**
   - ADG inherits biases from base LLM (e.g., US-centric for "Georgia")
   - May under-detect ambiguity if LLM doesn't explore diverse interpretations
   - Mitigation: Use diverse, well-calibrated base models

2. **Dataset Bias:**
   - Evaluation datasets are Wikipedia/web-based (skewed toward Western knowledge)
   - May not reflect queries from diverse populations
   - Mitigation: Test on domain-specific data before deployment

3. **Language Bias:**
   - English-only system
   - May disadvantage non-native speakers with ambiguous phrasing
   - Mitigation: Multilingual extension needed

### Safety Considerations

1. **Overconfidence:**
   - ADG may still answer confidently when wrong
   - Abstention rate depends on calibrated thresholds
   - **Recommendation:** Human review for high-stakes decisions

2. **Under-Asking:**
   - If base LLM is biased toward one interpretation, ADG may not ask
   - User may not realize alternative interpretations exist
   - **Recommendation:** Monitor ask rate, test with diverse examples

3. **Unhelpful Abstentions:**
   - Abstaining too often frustrates users
   - Balance between safety (abstain) and utility (answer)
   - **Recommendation:** Tune thresholds based on use case

4. **Clarifying Question Quality:**
   - Poor clarifying questions may confuse users
   - May leak information or prime responses
   - **Recommendation:** Human review of generated questions

### Responsible Use

**DO:**
- Test ADG on your specific domain before deployment
- Monitor performance metrics (ask rate, abstain rate, accuracy)
- Provide users with context about system capabilities
- Use in conjunction with human oversight for important decisions
- Document any domain-specific calibration or adaptations

**DON'T:**
- Deploy without testing on representative user queries
- Use for medical, legal, or safety-critical decisions without additional safeguards
- Assume ambiguity detection is perfect (see failure modes)
- Rely solely on ADG for high-stakes decisions
- Use in languages other than English without testing NLI model

---

## Caveats and Recommendations

### For Deployment

1. **Calibrate thresholds** on your domain's data
2. **Monitor metrics** in production (ask/abstain rates, user satisfaction)
3. **A/B test** against baselines to verify improvement
4. **Provide fallback** for abstentions (e.g., "I'm not sure, but here's what I found...")
5. **Collect feedback** to improve over time

### For Research

1. **Report both ADG-Core and ADG+IC** for fair comparison
2. **Use equal sample budgets** when comparing baselines
3. **Statistical significance testing** required for claims
4. **Negative results** are valuable - report honestly
5. **Failure analysis** is encouraged (see failure modes)

### For Users

1. **Be specific** in queries to avoid unnecessary clarifications
2. **Expect some abstentions** - honesty is valuable
3. **Provide feedback** when clarifying questions are confusing
4. **Don't rely solely on AI** for important decisions

---

## Model Card Authors

- ADG Team (University AI Course Project)
- Developed with assistance from Claude Code (Anthropic)

## Model Card Contact

For questions, issues, or feedback:
- **GitHub Issues:** https://github.com/adg-team/ask-dont-guess/issues
- **Documentation:** See [docs/](../docs/) directory

---

## Citation

```bibtex
@software{adg2024,
  title = {Ask, Don't Guess: Training-Free Ambiguity Detection for Small LLMs},
  author = {ADG Team},
  year = {2024},
  url = {https://github.com/adg-team/ask-dont-guess},
  license = {Apache-2.0}
}
```

---

## Changelog

### Version 0.1.0 (November 2024)

**Initial release:**
- ADG-Core and ADG+IC implementations
- 5 baseline methods
- Evaluation framework
- Gradio demo
- Documentation

**Components:**
- Base LLM: Qwen2.5-1.5B-Instruct (default)
- NLI: DeBERTa-large-MNLI (default)
- Datasets: AmbigQA, CLAMBER, NQ-open (loaders implemented)

**Known Issues:**
- Full experiments not yet run (results pending)
- English-only
- No web frontend (Gradio demo only)
- DPO extension not implemented

---

## Glossary

- **ADG:** Ask, Don't Guess
- **NLI:** Natural Language Inference
- **ANSWER:** Decision to answer directly
- **ASK:** Decision to ask clarifying question
- **ABSTAIN:** Decision to decline answering
- **Semantic Entropy:** Uncertainty measure based on meaning clusters
- **DeBERTa:** Decoding-enhanced BERT with Disentangled Attention
- **p1, p2, Δ:** Cluster structure features

---

## See Also

- [algorithm.md](algorithm.md) - How ADG works
- [limitations.md](limitations.md) - Detailed limitations and failure modes
- [reproduction.md](reproduction.md) - How to reproduce results
- [data_card.md](data_card.md) - Dataset documentation
- [SPEC.md](SPEC.md) - Full system specification

---

**Last Updated:** November 24, 2024
