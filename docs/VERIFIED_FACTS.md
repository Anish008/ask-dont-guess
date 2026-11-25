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
