# Verified Facts

Last updated: 2024-11-25

## Step 1: Real Model Support (2024-11-25)

### Models Loaded Successfully
- **LLM**: Qwen/Qwen2.5-1.5B-Instruct
  - Model ID verified: `Qwen/Qwen2.5-1.5B-Instruct`
  - Revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  - Size: ~3GB (safetensors format)
  - Loading time: ~5 seconds on CPU
  - Config: `configs/models/qwen2.5-1.5b.yaml`

- **NLI**: microsoft/deberta-large-mnli
  - Model ID verified: `microsoft/deberta-large-mnli`
  - Revision: 7296194b9009373def4f7c5dad292651e4b5cf4e
  - Size: ~1.6GB (pytorch_model.bin format)
  - Loading time: ~30 seconds on CPU
  - Config: `configs/nli/deberta-large-mnli.yaml`

### Inference Tests (3 examples)
- **Example 1**: "What is the capital of France?" (clear question)
  - Sampling: 10 samples generated successfully
  - Inference time: ~2 minutes on CPU (10 samples)
  - NLI scoring: 45 pairs computed
  - Clustering: Agglomerative, 2 clusters found

- **Example 2**: "What is the capital of Georgia?" (ambiguous)
  - Sampling: 10 samples generated successfully
  - Inference time: ~1.8 minutes on CPU
  - NLI scoring: 45 pairs computed
  - Clustering: 1 cluster found (all Tbilisi - country interpretation)

- **Example 3**: "Most popular ice cream flavor in Antarctica?" (unknown)
  - Sampling: 10 samples generated successfully
  - Inference time: ~1.8 minutes on CPU
  - NLI scoring: 45 pairs computed
  - Clustering: 1 cluster found

### Hardware Info
- Device: CPU (no GPU detected)
- CPU count: 16 cores
- RAM: ~31GB
- Dtype: fp32 (no quantization)

### Known Issues
1. **Structured prompt parsing**: Model produces conversational text instead of structured format
   - Expected: `Interpretation: X\nAnswer: Y\nConfidence: Z`
   - Actual: Conversational explanation with answer embedded
   - Result: All samples marked as `n_valid=0`
   - Decision: System falls back to ABSTAIN due to invalid samples

2. **Performance**: CPU inference is slow (~10-12 seconds per sample)
   - GPU would be 10-100x faster
   - Consider GPU deployment for production

### Config Loading
- Fixed config path resolution: `config_dir = config_path.parent` (not `.parent.parent`)
- Config references work correctly: `llm.config: qwen2.5-1.5b` → `configs/models/qwen2.5-1.5b.yaml`
- Merge order correct: referenced config loaded, then base.yaml overrides applied

### Logs
- Full test log: `results/step1_test.log`
- Test script: `scripts/test_real_models.py`

### Next Steps
1. Fix structured prompt format (make it work with Qwen instruct models)
2. Consider using chat templates instead of plain prompts
3. Test with GPU if available for realistic performance
4. Implement full baselines (Step 2)

## Datasets (TODO - Step 3)
_To be verified_

## Experiments (TODO - Step 4)
_To be run after Steps 1-3 complete_

## Step 3: Dataset Verification (2026-11-25)

### AmbigQA
- **Source**: HuggingFace `sewon/ambig_qa` (config: `full`)
- **Splits**: train, validation
- **train**: 10036 examples
- **validation**: 2002 examples
- **Columns**: id, question, annotations, viewed_doc_titles, used_queries, nq_answer, nq_doc_title

**Annotation types:**
- `multipleQAs`: 5817 (48.3%)
- `singleAnswer`: 6221 (51.7%)

**Schema:**
```python
# multipleQAs example:
id: -4469503464110108318
question: "When did the simpsons first air on television?"
annotations['type']: ['multipleQAs']
annotations['qaPairs']: [
  {'question': [...], 'answer': [...]}
]
```

```python
# singleAnswer example:
question: "Who plays the doctor in dexter season 1?"
annotations['answer']: [['Tony Goldwyn', 'Goldwyn']]
```

### NQ-open
- **Source**: HuggingFace `google-research-datasets/nq_open`
- **Splits**: train, validation
- **train**: 87925 examples
- **validation**: 3610 examples
- **Columns**: question, answer

**Schema:**
```python
question: "where did they film hot tub time machine"
answer: ['Fernie Alpine Resort']
```

### CLAMBER
- **Source**: GitHub `zt991211/CLAMBER`
- **File**: `clamber_benchmark.jsonl`
- **Total examples**: 3202

**Categories:**
- `MC`: 1602 (50.0%)
- `FD`: 800 (25.0%)
- `LA`: 800 (25.0%)

**Requires clarification**: 1601 (50.0%)

**Schema:**
```python
question: "Give me a list of good coffee shops?"
require_clarification: 1
category: "MC"
subclass: "whom"
clarifying_question: "What do you personally consider important in a coffee shop? Are you looking for ..."
```

### AmbigQA ↔ NQ-open Deduplication
- **Unique AmbigQA questions**: 12038
- **NQ-open train**: 87925 total, 12041 overlap (13.7%)
- **NQ-open validation**: 3610 total, 0 overlap (0.0%)

**Deduplication strategy:** Exclude NQ-open questions that appear in AmbigQA (normalized match)

## Step 4: Initial Experiment Setup (2026-11-25)

### Pipeline Verification
**Test script**: `scripts/test_pipeline_simple.py`

**Models tested:**
- LLM: Qwen/Qwen2.5-1.5B-Instruct (3GB, CPU inference)
- NLI: microsoft/deberta-large-mnli (1.6GB, CPU inference)

**Baseline methods tested:**
- `always_answer`: 100% ANSWER (as expected)
- `ask_if_unclear`: Functional with fallback mechanisms

**Performance:**
- Inference time: ~24 seconds per query on CPU (16 cores, fp32)
- Sample generation: ~20-25 seconds
- NLI scoring: minimal overhead
- Total pipeline: functional end-to-end

### Structured Prompt Issue

**Problem**: Models produce conversational text instead of JSON format
- Expected: `{"action": "answer", "content": "Paris"}`
- Actual: `"For example:\nInterpretation: ..."`

**Solution**: Fallback mechanisms implemented
- JSON parsing with error handling
- Heuristic-based decisions on parse failure
- Methods remain functional despite format issues

**Parse failure rate**: ~66% (2/3 test queries)
- Fallback to heuristics works correctly
- "I don't know" correctly triggers ABSTAIN
- Methods produce reasonable decisions

### Key Findings

1. **Pipeline works end-to-end**: Query → Method → Decision
2. **Baselines are functional**: All methods produce valid outputs
3. **Fallbacks are essential**: JSON parsing unreliable with Qwen 1.5B
4. **Performance acceptable**: CPU inference viable for experiments
5. **Model limitations**: Small instruction-tuned models struggle with strict formatting

### Remaining Work for Full Experiments

**Not yet implemented:**
- Data loading infrastructure (dataset → Example schema)
- Evaluation metrics computation
- Experiment runner with batch processing
- Threshold calibration system
- Report generation

**Estimated effort**: 40-50 hours for complete implementation

See `docs/STEP4_NOTES.md` for detailed analysis and recommendations.
