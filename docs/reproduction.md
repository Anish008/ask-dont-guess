# Reproduction Guide

This document provides step-by-step instructions to reproduce the ADG experiments and results.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Running Experiments](#running-experiments)
5. [Generating Reports](#generating-reports)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements (Mock Mode)
- **OS:** Linux, macOS, or Windows
- **Python:** 3.11 or 3.12
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **GPU:** None required (CPU-only)

### Recommended Requirements (Real Models)
- **OS:** Linux (preferred) or macOS
- **Python:** 3.11 or 3.12
- **RAM:** 16 GB
- **Storage:** 20 GB free space (for models and datasets)
- **GPU:** 8+ GB VRAM (NVIDIA with CUDA 11.8+)

### Full Experiment Requirements
- **GPU:** 16+ GB VRAM (or multiple GPUs)
- **RAM:** 32 GB
- **Storage:** 50 GB free space
- **Time:** ~8-12 hours for full experiment sweep

---

## Environment Setup

### Step 1: Install Prerequisites

#### Python 3.11+
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv

# macOS (Homebrew)
brew install python@3.11

# Windows
# Download from python.org
```

#### uv Package Manager
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Git
```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git

# Windows
# Download from git-scm.com
```

### Step 2: Clone Repository

```bash
git clone https://github.com/adg-team/ask-dont-guess.git
cd ask-dont-guess
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
make setup

# Or manually
py -3.11 -m uv sync --all-extras  # Windows
python3.11 -m uv sync --all-extras  # Linux/macOS
```

This installs:
- Core dependencies (PyTorch, Transformers, etc.)
- Development tools (pytest, ruff, mypy)
- Optional extras (gradio, fastapi, etc.)

### Step 4: Verify Installation

```bash
# Run tests
make test

# Check linting
make lint

# Check type checking
make typecheck
```

All should pass without errors.

---

## Data Preparation

### Quick Test (Mock Data)

No data download needed for mock mode:

```bash
# Run pipeline with mock backends
make demo-mock
```

### Real Datasets

#### Step 1: Download Datasets

```bash
# Build all datasets (AmbigQA, CLAMBER, NQ-open)
make data

# Or use CLI directly
adg data build --datasets ambigqa nq_open clamber
```

**Expected output:**
```
results/data/
├── ambigqa/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── nq_open/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── clamber/
│   ├── dev.json
│   └── test.json
└── manifest.json
```

**Note:** This downloads ~2-5 GB of data. See `docs/data_card.md` for details.

#### Step 2: Create Data Splits

The build process automatically creates:
- **Calibration split:** For threshold tuning
- **Dev split:** For development/debugging (small subset)
- **Test split:** For final evaluation

```bash
# Verify splits
adg data verify --dataset ambigqa
```

---

## Running Experiments

### Quick Development Run

Small-scale experiment for testing:

```bash
# Dev configuration (10 examples per dataset)
adg experiment run --config configs/experiments/dev_small.yaml
```

**What it does:**
- Runs all 7 methods (5 baselines + ADG-Core + ADG+IC)
- On 10 examples from each dataset
- With mock backends (fast, no GPU)
- Outputs: `results/runs/dev_<timestamp>/`

### Single Method Experiment

```bash
# Run only ADG-Core on AmbigQA
adg experiment run \
    --methods adg_core \
    --datasets ambigqa \
    --config configs/base.yaml
```

### Full Experiment (Reproduce Paper Results)

```bash
# Full experiment with real models
# WARNING: Takes 8-12 hours, requires 8+ GB VRAM
adg experiment run --config configs/experiments/full.yaml
```

**Configuration (`configs/experiments/full.yaml`):**
```yaml
name: "full_experiment"
methods:
  - always_answer
  - ask_if_unclear
  - stated_confidence
  - logprob_threshold
  - semantic_entropy
  - adg_core
  - adg_ic

datasets:
  - ambigqa
  - nq_open
  - clamber

llm:
  backend: hf
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  device: auto
  dtype: float16

nli:
  backend: hf
  model_name: "microsoft/deberta-large-mnli"

sampling:
  n_samples: 8
  temperature: 0.7
  top_p: 0.95

seed: 42
```

**Run with specific GPU:**
```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 adg experiment run --config configs/experiments/full.yaml

# Use multiple GPUs (if vLLM installed)
CUDA_VISIBLE_DEVICES=0,1 adg experiment run --config configs/experiments/full.yaml
```

### Experiment Outputs

Each run creates:
```
results/runs/<run_id>/
├── manifest.json           # Run configuration
├── metrics.json            # Computed metrics
├── predictions/            # Per-example predictions
│   ├── ambigqa.jsonl
│   ├── nq_open.jsonl
│   └── clamber.jsonl
├── traces/                 # Decision traces (if enabled)
│   └── ...
└── logs/                   # Execution logs
    └── run.log
```

---

## Generating Reports

### List Available Runs

```bash
# See all experiment runs
adg report list
```

### Generate Report for a Run

```bash
# Generate tables and figures
adg report generate <run_id>

# Specify output directory
adg report generate <run_id> --output results/reports/<run_id>

# Choose table format
adg report generate <run_id> --format latex  # or markdown, csv
```

**Output:**
```
results/reports/<run_id>/
├── main_results.md         # Main results table
├── costs.md                # Cost comparison
├── ablations.md            # Ablation study
└── figures/
    ├── confusion_matrix.pdf
    ├── confusion_matrix.png
    ├── risk_coverage.pdf
    ├── risk_coverage.png
    ├── budget_sweep.pdf
    ├── budget_sweep.png
    ├── feature_scatter.pdf
    ├── feature_scatter.png
    ├── cluster_histogram.pdf
    └── cluster_histogram.png
```

### Generate All Reports

```bash
# Generate reports for all runs
for run_id in $(adg report list --format json | jq -r '.[].id'); do
    adg report generate $run_id
done
```

---

## Reproducing Specific Results

### Main Results Table (Table 1)

```bash
# Run full experiment
adg experiment run --config configs/experiments/full.yaml

# Generate main results table
RUN_ID=$(ls -t results/runs/ | head -1)
adg report generate $RUN_ID --format latex

# Table saved to: results/reports/$RUN_ID/main_results.tex
```

**Expected columns:**
- Method name
- ASK Precision / Recall / F1
- 3-way Accuracy (ANSWER/ASK/ABSTAIN)
- Answer EM / F1
- Average samples per query
- Average tokens per query

### Ablation Study (Table 2)

```bash
# Run ablation experiment
adg experiment run --config configs/experiments/ablation.yaml

# Generate ablation table
RUN_ID=$(ls -t results/runs/ | head -1)
adg report generate $RUN_ID

# Table: results/reports/$RUN_ID/ablations.md
```

**Ablations tested:**
- ADG-Core vs ADG+IC (interpretation channel)
- Different clustering algorithms
- Different sample counts (4, 8, 16)
- Different NLI models

### Risk-Coverage Curves (Figure 1)

```bash
# Run with varying confidence thresholds
adg experiment run --config configs/experiments/risk_coverage.yaml

# Generate figure
RUN_ID=$(ls -t results/runs/ | head -1)
adg report generate $RUN_ID

# Figure: results/reports/$RUN_ID/figures/risk_coverage.pdf
```

### Feature Space Visualization (Figure 2)

```bash
# Any full experiment generates this
adg experiment run --config configs/experiments/full.yaml

RUN_ID=$(ls -t results/runs/ | head -1)
adg report generate $RUN_ID

# Figure: results/reports/$RUN_ID/figures/feature_scatter.pdf
```

---

## Calibration

### Calibrate Decision Thresholds

```bash
# Calibrate on development set
adg calibrate \
    --dataset ambigqa \
    --split calibration \
    --output results/calibration/ambigqa_thresholds.json \
    --metric f1  # or accuracy, precision, recall
```

**Output (`ambigqa_thresholds.json`):**
```json
{
  "ask_threshold": 0.42,
  "answer_threshold": 0.58,
  "refusal_threshold": 0.3,
  "invalid_threshold": 0.3,
  "metric": "f1",
  "score": 0.73,
  "timestamp": "2024-11-24T20:00:00Z"
}
```

### Use Calibrated Thresholds

```bash
# Run experiment with calibrated thresholds
adg experiment run \
    --config configs/experiments/full.yaml \
    --thresholds results/calibration/ambigqa_thresholds.json
```

---

## Validation Checks

### Verify Data Loading

```bash
# Check dataset schemas
python -c "
from adg.data.loaders.ambigqa import AmbigQALoader
loader = AmbigQALoader()
data = loader.load_split('dev')
print(f'Loaded {len(data)} examples')
print(f'Example: {data[0]}')
"
```

### Verify Model Loading

```bash
# Check LLM loading
python -c "
from adg.llm.hf_backend import HFLLMBackend
backend = HFLLMBackend('Qwen/Qwen2.5-1.5B-Instruct')
print(f'Model loaded: {backend.model_name}')
"

# Check NLI loading
python -c "
from adg.nli.hf_nli import HFNLIScorer
scorer = HFNLIScorer('microsoft/deberta-large-mnli')
print(f'NLI model loaded: {scorer.model_name}')
"
```

### Verify Clustering

```bash
# Run clustering test
python -c "
from adg.clustering.algorithms import cluster_samples
from adg.core.types import Sample, SamplingResult
from adg.nli.mock_nli import MockNLIScorer

samples = [
    Sample(interpretation='I1', answer='Paris', sample_id=0),
    Sample(interpretation='I1', answer='Paris is the capital.', sample_id=1),
    Sample(interpretation='I2', answer='London', sample_id=2),
]

result = SamplingResult(query_id='test', samples=samples, cost=None)
scorer = MockNLIScorer()
clusters = cluster_samples(result, scorer, 'agglomerative', 0.25)

print(f'Found {len(clusters.clusters)} clusters')
for c in clusters.clusters:
    print(f'  Cluster {c.id}: {c.representative} ({c.size} samples)')
"
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `CUDA out of memory` or process killed

**Solutions:**
```bash
# 1. Use 4-bit quantization
adg experiment run --config configs/hardware/gpu_8gb_4bit.yaml

# 2. Reduce batch size (vLLM only)
adg experiment run --config configs/base.yaml \
    --llm.batch_size 1

# 3. Use smaller NLI model
adg experiment run --config configs/base.yaml \
    --nli.model_name microsoft/deberta-v3-base

# 4. Run on CPU (slow but works)
adg experiment run --config configs/hardware/cpu.yaml
```

### Slow Inference

**Expected times:**
- GPU (fp16): 2-3s per query
- GPU (4-bit): 3-5s per query
- CPU (8-core): 15-30s per query

**Speed ups:**
```bash
# 1. Use vLLM backend (Linux + CUDA only)
uv sync --extra vllm
adg experiment run --config configs/base.yaml --llm.backend vllm

# 2. Reduce samples
adg experiment run --config configs/base.yaml --sampling.n_samples 4

# 3. Enable caching
adg experiment run --config configs/base.yaml --cache.enabled true
```

### Dataset Download Fails

**Symptoms:** Network errors, timeouts

**Solutions:**
```bash
# 1. Manual download with retry
HF_HUB_DOWNLOAD_TIMEOUT=600 adg data build --datasets ambigqa

# 2. Use Hugging Face cache
export HF_DATASETS_CACHE=/path/to/large/disk
adg data build

# 3. Download datasets separately
python -c "
from datasets import load_dataset
ds = load_dataset('ambigqa', 'light')  # Smaller version
ds.save_to_disk('results/data/ambigqa')
"
```

### Pre-commit Hooks Fail

**Symptoms:** `git commit` fails with ruff/mypy errors

**Solutions:**
```bash
# 1. Auto-fix issues
make format
make lint --fix

# 2. Skip hooks (not recommended)
git commit --no-verify -m "message"

# 3. Update hooks
pre-commit autoupdate
pre-commit run --all-files
```

### Mock vs Real Mode Confusion

**Check which mode you're in:**
```bash
# Mock mode (no model loading)
adg experiment run --config configs/base.yaml --mock

# Real mode
adg experiment run --config configs/base.yaml --no-mock
```

**Mock runs are tagged in manifest:**
```json
{
  "run_id": "...",
  "mock": true,  // <-- Mock mode indicator
  "config": {...}
}
```

**Report generation warns about mock runs:**
```
Warning: This is a mock run.
Per spec: Mock runs should not be reported as real experiments.
Continue anyway? [y/N]
```

---

## Reproducibility Checklist

- [ ] Python 3.11+ installed
- [ ] uv package manager installed
- [ ] Git repository cloned
- [ ] Dependencies installed (`make setup`)
- [ ] Tests pass (`make test`)
- [ ] Data downloaded (`make data`)
- [ ] Thresholds calibrated (`adg calibrate`)
- [ ] Experiment config verified
- [ ] Experiment run completed
- [ ] Reports generated
- [ ] Results match expected structure
- [ ] Seed fixed (default: 42)
- [ ] Hardware configuration documented

---

## Expected Results

**Note:** Actual results depend on:
- Random seed (default: 42)
- Hardware (GPU vs CPU, memory)
- Library versions (PyTorch, Transformers)
- Dataset versions

Small variations (±0.02 F1) are expected. Large deviations suggest a configuration issue.

---

## Getting Help

**Check documentation:**
- [algorithm.md](algorithm.md) - Algorithm details
- [limitations.md](limitations.md) - Known issues
- [SPEC.md](SPEC.md) - Full specification

**Common issues:**
- See [Troubleshooting](#troubleshooting) above
- Check GitHub Issues: https://github.com/adg-team/ask-dont-guess/issues

**Report bugs:**
- Include: OS, Python version, GPU model, error message, full traceback
- Minimal reproduction steps
- Output of `adg --version` and `uv pip list`
