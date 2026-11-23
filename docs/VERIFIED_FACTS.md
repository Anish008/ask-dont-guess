# Verified Facts (Never Fabricated)

**Last updated:** 2024-11-23
**Purpose:** Record all verified dataset schemas, model IDs, API details, and experimental facts. Per spec §0.1 point 3, we never invent or guess these details.

---

## Models

### LLM Models
*(To be verified in Phase 4 when first loading models)*

**Qwen/Qwen2.5-1.5B-Instruct:**
- TODO(verify): Exact Hugging Face model ID
- TODO(verify): Model revision/commit SHA to pin
- TODO(verify): License terms
- TODO(verify): Tokenizer details and chat template format
- TODO(verify): Memory footprint in fp16 and 4-bit
- TODO(verify): Context length

**Qwen/Qwen2.5-3B-Instruct:**
- TODO(verify): Same as above

**meta-llama/Llama-3.2-3B-Instruct:**
- TODO(verify): License acceptance required
- TODO(verify): Gated model access process
- TODO(verify): Same verification as Qwen models

---

## NLI Models

### DeBERTa-MNLI
*(To be verified in Phase 5 when first loading NLI model)*

**microsoft/deberta-large-mnli:**
- TODO(verify): Exact model ID on Hugging Face
- TODO(verify): config.id2label mapping (which index = entailment/neutral/contradiction)
- TODO(verify): License
- TODO(verify): Max sequence length
- TODO(verify): Memory footprint in fp16
- TODO(verify): Expected probabilities on test pairs:
  - ("A man is playing guitar.", "A person is playing an instrument.") → should be entailment-dominant
  - ("The capital is Atlanta.", "The capital is Tbilisi.") → should be contradiction-dominant

**microsoft/deberta-v3-base (alternative for CPU dev):**
- TODO(verify): Same as above

---

## Datasets

### AmbigQA
*(To be verified in Phase 11 when loading datasets)*

- TODO(verify): Hugging Face dataset ID (e.g., `sewon/ambig_qa`)
- TODO(verify): Available configs (light/full)
- TODO(verify): Split names and sizes (train/dev/test)
- TODO(verify): Field names and schema:
  - Expected: id, question, annotations (with type, answer, qaPairs)
- TODO(verify): License
- TODO(verify): Number of multi-interpretation vs single-answer examples

### NQ-Open
- TODO(verify): Hugging Face dataset ID
- TODO(verify): Field names (expected: question, answer as list[str])
- TODO(verify): License
- TODO(verify): Total questions and overlap with AmbigQA (for deduplication)

### CLAMBER
- TODO(verify): GitHub repository URL and file format
- TODO(verify): Fields: ambiguity flag, category taxonomy, gold clarifying question
- TODO(verify): License
- TODO(verify): Number of examples and splits (if any)

---

## API Models

### OpenAI GPT-4o-mini
*(To be verified when first using OpenAI API)*

- TODO(verify): Current model name string (e.g., "gpt-4o-mini" or "gpt-4o-mini-2024-07-18")
- TODO(verify): Pricing (USD per 1M tokens, input and output)
- TODO(verify): Whether logprobs are available for this model
- TODO(verify): Context window size
- TODO(verify): Rate limits

---

## Experimental Results

*(Results must be computed by our pipeline, never invented)*

**All result tables in README and reports will be generated from:**
- `results/runs/<run_id>/metrics.json`
- `results/runs/<run_id>/predictions.jsonl`

Until real runs complete, README states: **"Results pending"**

---

## Hardware Measurements

**Current Development Machine (measured 2024-11-23):**
- GPU: NVIDIA GeForce RTX 5070, 8GB VRAM, CUDA 13.5
- RAM: ~32 GB
- Python: 3.11.9
- PyTorch: (to be verified in Phase 2 after `make setup`)
- Transformers: (to be verified in Phase 2)

**Model Memory Measurements:**
*(To be filled after Phase 4-5 when loading models)*

| Model | Config | Memory (MB) | Latency (s/query) | Notes |
|-------|--------|-------------|-------------------|-------|
| Qwen2.5-1.5B fp16 | - | TODO | TODO | - |
| Qwen2.5-3B fp16 | - | TODO | TODO | - |
| Qwen2.5-3B 4-bit | - | TODO | TODO | - |
| DeBERTa-large fp16 | - | TODO | TODO | - |

---

## Package Versions

*(To be recorded after `make setup` in Phase 1)*

**Core ML Stack:**
- torch: (pending)
- transformers: (pending)
- accelerate: (pending)
- datasets: (pending)
- bitsandbytes: (pending, if Windows-compatible)

---

## Notes

- This file is updated whenever we verify a fact by actually loading and inspecting an artifact
- All TODO(verify) items will be resolved in the relevant phases
- If a dataset schema differs from expectations, we adapt and document in DEVIATIONS.md
