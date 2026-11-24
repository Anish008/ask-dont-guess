# Data Card: ADG Evaluation Datasets

**Last updated:** 2024-11-24
**Status:** Placeholder - awaiting dataset verification

## Datasets

### AmbigQA

**Source:** Hugging Face Hub
**Dataset ID:** `sewon/ambig_qa` (TODO: verify)
**Config:** `light`
**License:** TODO(verify)
**Citation:** Min et al., EMNLP 2020

**Description:** Ambiguous open-domain questions with disambiguated QA pairs.

**Fields (expected, TODO verify):**
- `id`: Example identifier
- `question`: Original ambiguous question
- `annotations`: List of annotations
  - `type`: "singleAnswer" | "multipleQAs"
  - `answer`: List of answer strings
  - `qaPairs`: List of disambiguated question-answer pairs

**Splits:**
- train: TODO(verify count)
- validation: TODO(verify count)

**Statistics (TODO: compute with `adg data stats`):**
- Total examples: TODO
- Ambiguous examples: TODO
- Average interpretations per ambiguous question: TODO
- Question length distribution: TODO

---

### NQ-open

**Source:** Hugging Face Hub
**Dataset ID:** `nq_open` (TODO: verify)
**License:** TODO(verify)
**Citation:** Google Research

**Description:** Open-domain QA from Natural Questions.

**Fields (expected, TODO verify):**
- `question`: Question text
- `answer`: List of answer strings

**Splits:**
- train: TODO(verify count)
- validation: TODO(verify count)
- test: TODO(verify count)

**Statistics (TODO: compute):**
- Total examples: TODO
- After deduplication against AmbigQA: TODO (removed: TODO)
- Question length distribution: TODO

---

### CLAMBER

**Source:** Official GitHub repository
**Repository:** TODO(verify URL)
**License:** TODO(verify)
**Citation:** Zhang et al., ACL 2024

**Description:** Taxonomy and benchmark of ambiguous information needs.

**Fields (expected, TODO verify):**
- `question`: Question text
- `needs_clarification`: Boolean flag
- `ambiguity_category`: Category from taxonomy
- `gold_clarifying_question`: Optional gold clarifying question

**Splits:**
- TODO(verify if splits exist, otherwise create deterministic split)

**Statistics (TODO: compute):**
- Total examples: TODO
- Needs clarification: TODO
- Ambiguity categories distribution: TODO

---

## Data Processing

### Deduplication

Per spec §9.3: AmbigQA questions are drawn from NQ-open.

**Procedure:**
1. Normalize questions: lowercase, strip punctuation, collapse whitespace
2. Build set of AmbigQA questions
3. Filter NQ-open to exclude matches

**Results (TODO: compute):**
- NQ-open original count: TODO
- Removed (appeared in AmbigQA): TODO
- NQ-open final count: TODO

### Splits

**Calibration split:**
- 30% of evaluation pool for threshold calibration
- Stratified by `is_ambiguous`
- Seed: 42
- Stored as ID lists in `data/splits/*.json`

**Dev subsets:**
- Small balanced sets for fast iteration
- Default: 100 examples, stratified
- Created with `--limit N --stratify --seed S`

---

## Usage

```bash
# Verify dataset schemas (must run before loading)
# TODO: Implement verification script

# Build processed datasets
# adg data build --datasets ambigqa nq_open clamber

# Compute statistics
# adg data stats

# Create dev subset
# adg data subset --limit 100 --stratify
```

---

## Limitations and Caveats

1. **Dataset ambiguity labels are noisy:** Some `singleAnswer` examples may be arguably ambiguous, and vice versa. This is documented in the spec §9.4.

2. **Verification required:** All dataset IDs, field names, and licenses must be verified before actual use. This card will be updated with observed statistics.

3. **Leakage prevention:** NQ-open deduplication is critical to prevent evaluation leakage.

---

## Responsible AI Considerations

- **Privacy:** All datasets are publicly available research datasets
- **Bias:** Datasets may reflect biases in their source corpora
- **Intended use:** Evaluation of ambiguity detection systems only
- **Out of scope:** Production deployment without additional validation

---

## References

- **AmbigQA:** Min, S., et al. (2020). Ambiguous Questions Answered. EMNLP 2020.
- **CLAMBER:** Zhang, Y., et al. (2024). CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs. ACL 2024.
- **Natural Questions:** Kwiatkowski, T., et al. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL 2019.
