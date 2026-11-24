# Ask, Don't Guess (ADG)

**Training-free ambiguity detection for small instruction-tuned LLMs**

[![CI](https://img.shields.io/badge/CI-pending-yellow)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What is ADG?

When a user asks "What is the capital of Georgia?" — do they mean the US state or the country? Small LLMs (1.5B-3B parameters) usually pick one interpretation silently and answer confidently. Users often don't notice they answered a different question.

**Ask, Don't Guess** wraps a small LLM to decide:
- **ANSWER:** When the question is clear and the model knows the answer
- **ASK:** When the question is ambiguous — with a targeted clarifying question like *"Do you mean Georgia the US state or Georgia the country?"*
- **ABSTAIN:** When the model doesn't have enough information

The system samples ~8 answers, groups them by semantic meaning (using an NLI model), and reads the *structure* of the groups: one coherent group → answer; a few coherent, distinct groups → ask; scattered groups → abstain.

**Key insight:** Semantic entropy compresses meaning clusters into one number. We keep the structure to separate *ambiguity* (different interpretations) from *ignorance* (conflicting answers under the same interpretation).

---

## Features

- **Training-free:** No fine-tuning of the LLM, just inference-time sampling and clustering
- **Transparent:** Every decision shows the samples, clusters, and decision trace
- **Targeted clarification:** Questions built from the actual meaning groups, not generic prompts
- **Multi-turn resolution:** Resolves user replies and answers for the chosen interpretation
- **Research-grade:** 5 baselines, rigorous evaluation, statistical tests, validity checks
- **Production-ready:** FastAPI backend, CLI, Gradio demo, comprehensive tests, CI

---

## Quick Start

### Prerequisites
- Python 3.11 or 3.12
- 8+ GB VRAM (for real models) or CPU (for mock/demo mode)
- uv package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/adg-team/ask-dont-guess.git
cd ask-dont-guess

# Setup environment (installs all dependencies)
make setup

# Run tests
make test
```

### Demo (Mock Mode - No GPU Required)

```bash
# Run with mock backends (instant, deterministic)
make demo-mock
```

### Demo (Real Models)

```bash
# Downloads Qwen2.5-1.5B-Instruct + DeBERTa-MNLI (~4-5 GB)
# Requires GPU or runs slowly on CPU
make demo
```

---

## Results

**Status:** Results pending (experiments in progress)

Final results will be auto-generated from `results/runs/` and include:
- Asking precision/recall/F1 vs 5 baselines
- End-to-end accuracy with simulated user
- Clarifying question quality (LLM judge + human validation)
- Cost comparison (samples, tokens, latency)

---

## Architecture

```
User Query → Sample 8 answers → Group by meaning (NLI) →
  → Cluster structure analysis → Decision (ANSWER/ASK/ABSTAIN) →
    → If ASK: build question from clusters → User reply →
      → Resolve interpretation → Final answer
```

**Core components:**
- **LLM backends:** HF Transformers, vLLM, OpenAI, Mock
- **NLI model:** DeBERTa-MNLI for semantic equivalence
- **Clustering:** Agglomerative clustering on bidirectional entailment
- **Decision engine:** Interpretable rule list with calibrated thresholds
- **Methods:** Always Answer, Ask-If-Unclear prompt, Stated Confidence, Log-Prob Threshold, Semantic Entropy, **ADG-Core**, **ADG+IC** (with interpretation channel)

See `docs/architecture.md` and `docs/SPEC.md` for details.

---

## Repository Structure

```
ask-dont-guess/
├── src/adg/           # Core library
│   ├── llm/           # LLM backends
│   ├── nli/           # NLI scorer
│   ├── clustering/    # Semantic grouping
│   ├── decision/      # Decision engine
│   ├── methods/       # Baselines + our method
│   ├── data/          # Dataset loaders
│   ├── evaluation/    # Metrics and experiments
│   └── ...
├── tests/             # Comprehensive test suite
├── configs/           # Model and experiment configs
├── docs/              # Documentation
│   └── SPEC.md        # Full specification
├── results/           # Experiment outputs (gitignored)
└── frontend/          # Next.js dashboard (optional)
```

---

## Usage

### CLI

```bash
# Analyze a single query
adg analyze "What is the capital of Georgia?"

# Interactive chat
adg chat

# Build datasets
adg data build --datasets ambigqa nq_open clamber

# Run experiments
adg experiment run --config configs/experiments/dev_small.yaml

# Generate report tables and figures
adg report --run <run_id>
```

### Python API

```python
from adg import ADGPipeline

# Load pipeline
pipeline = ADGPipeline.from_config("configs/base.yaml")

# Analyze a query
result = pipeline.analyze("What is the capital of Georgia?")

print(result.action)  # Action.ASK
print(result.message)  # "Do you mean Georgia the US state or..."
print(result.options)  # ["the US state", "the country"]
```

---

## Development

```bash
# Run linter
make lint

# Format code
make format

# Type check
make typecheck

# Run all tests (including slow model tests)
make test-slow

# Clean caches
make clean
```

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Guide for Claude Code (conventions, commands, rules)
- **[docs/SPEC.md](docs/SPEC.md)** - Complete specification (read before coding!)
- **[docs/algorithm.md](docs/algorithm.md)** - Core algorithm details and design
- **[docs/reproduction.md](docs/reproduction.md)** - How to reproduce experiments
- **[docs/limitations.md](docs/limitations.md)** - Known limitations and failure modes
- **[docs/model_card.md](docs/model_card.md)** - Model card with intended use and performance
- **[docs/data_card.md](docs/data_card.md)** - Dataset documentation
- **[docs/IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md)** - Progress tracker

---

## Citation

```bibtex
@software{adg2024,
  title = {Ask, Don't Guess: Training-Free Ambiguity Detection for Small LLMs},
  author = {ADG Team},
  year = {2024},
  url = {https://github.com/adg-team/ask-dont-guess}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## AI Assistance Disclosure

This software was implemented with assistance from Claude Code (Anthropic), under the direction and design decisions of the research team. All research decisions, experimental design, and analysis remain the work of the team.

---

## Team

**ADG Team** - University AI Course Project

For questions or issues, please open an issue on GitHub.

---

**Status:** Phases 1-15 complete ✓ | Phase 16 (API & Frontend) optional | Phase 17 (Documentation) complete ✓
