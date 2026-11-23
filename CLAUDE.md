# Claude Code Guide for Ask, Don't Guess (ADG)

## Project Description

Ask, Don't Guess (ADG) is a training-free, inference-time system that wraps small instruction-tuned LLMs (1.5B-3B parameters) to detect when user queries are ambiguous. Instead of silently guessing one interpretation, it decides whether to ANSWER directly (clear question), ASK a targeted clarifying question (ambiguous), or ABSTAIN honestly (doesn't know). The system uses semantic clustering of sampled answers to distinguish ambiguity from ignorance, then generates clarifying questions from the meaning clusters themselves.

## Specification

**The full specification is in `docs/SPEC.md`; read the relevant section before changing a module.**

This is a production-quality, research-grade implementation following strict engineering standards:
- No fabricated dataset schemas, model IDs, or results
- All claims must be verifiable
- Mock mode for development, real models for experiments
- Comprehensive testing with ≥85% coverage on core algorithm

## Essential Commands

```bash
# Setup (first time)
make setup              # Install all dependencies and setup hooks

# Development workflow
make lint               # Check code style
make format             # Auto-format code
make typecheck          # Type check with mypy (strict on core/)
make test               # Run fast tests (mock-only, no model downloads)
make test-slow          # Run all tests including real model tests

# Demo
make demo-mock          # Run with mock backends (no GPU needed)
make demo               # Run with real models (Qwen2.5-1.5B + DeBERTa)

# Data and experiments
make data               # Build AmbigQA, CLAMBER, NQ-open datasets
make experiment         # Run experiments with baselines
make report             # Generate tables and figures for the report

# Cleanup
make clean              # Remove caches and generated files
```

## Coding Conventions (spec §16)

### Style
- **Line length:** 100 characters
- **Formatter:** ruff (configured in pyproject.toml)
- **Linter:** ruff with E, F, I, B, UP, SIM, RUF, PL rules
- **Type checker:** mypy (strict mode on `src/adg/core/`, `clustering/`, `decision/`, `evaluation/`)

### Code Quality
- **No `print` in library code** - use `structlog` loggers
- **No bare `except`** - catch specific exceptions
- **No mutable default arguments**
- **Use `pathlib` everywhere**, not string paths
- **Explicit `encoding="utf-8"`** for file I/O
- **Pure functions for core algorithm** - side effects at the edges only

### Documentation
- **Public functions:** Google-style docstrings with Args/Returns/Raises
- **Type hints everywhere** in function signatures
- **Constants in config**, not magic numbers in code

### Testing
- **Mock-only by default** - real models marked with `@pytest.mark.slow`
- **Coverage gates:** ≥85% for core algorithm modules, ≥70% overall
- **Unit tests** for pure functions; **integration tests** with mocks; **e2e tests** for full pipeline

## Critical Rules

### Never Fabricate (spec §0.1 point 3)
- **DO NOT** invent dataset URLs, field names, model IDs, or experimental results
- **ALWAYS** verify by loading the actual artifact (model config, dataset schema)
- **RECORD** verified facts in `docs/VERIFIED_FACTS.md` with date and method
- If you cannot verify something, leave a `# TODO(verify): ...` comment

### No Placeholder Implementations (spec §0.6)
- **NO** `pass`, `return None`, or `NotImplementedError` in core functionality
- Core = sampling, NLI scoring, clustering, features, decision, clarification, baselines, metrics
- Mocks (`mock_backend.py`, `mock_nli.py`) are allowed ONLY for tests and UI development
- Mocks must be clearly labeled and never reported as real experiments

### Work in Phases (spec §0.1 point 2)
1. State what you will implement and how completion will be verified
2. Implement the phase
3. Run tests, linters, type checks
4. Update `docs/IMPLEMENTATION_CHECKLIST.md`
5. Commit with Conventional Commit message
6. **Do not start the next phase while critical failures remain**

### Ask Before Expensive Actions (spec §0.1 point 6)
- Downloading more than ~2 GB of data
- Starting a run expected to take >30 minutes
- Spending money on API calls (estimate cost first)
- Deleting data or results

## Repository Structure

```
ask-dont-guess/
├── src/adg/           # Core library (src layout)
│   ├── core/          # Types, errors, logging, config
│   ├── llm/           # LLM backends (HF, vLLM, OpenAI, Mock)
│   ├── nli/           # NLI scorer (DeBERTa, Mock)
│   ├── sampling/      # Answer sampling
│   ├── clustering/    # Semantic grouping (NLI-based)
│   ├── decision/      # ANSWER/ASK/ABSTAIN decision engine
│   ├── clarify/       # Clarifying question generation
│   ├── conversation/  # Multi-turn state management
│   ├── pipeline/      # Orchestration
│   ├── methods/       # 5 baselines + ADG-Core + ADG+IC
│   ├── data/          # Dataset loaders and processing
│   ├── simuser/       # Simulated user for evaluation
│   ├── evaluation/    # Metrics, end-to-end evaluation
│   ├── tracking/      # Run management
│   ├── reporting/     # Tables and plots
│   ├── api/           # FastAPI service
│   ├── cli/           # Typer CLI
│   └── demo/          # Gradio demo
├── tests/             # Mirrors src/ structure
├── configs/           # YAML configs (models, methods, experiments)
├── docs/              # Documentation
│   └── SPEC.md        # Full specification (read before coding!)
├── frontend/          # Next.js dashboard (optional, Phase 16)
└── results/           # Run outputs (gitignored except calibration/)
```

## Phase Progress

Current phase: **Phase 1 - Repository Inspection & Project Setup**

See `docs/IMPLEMENTATION_CHECKLIST.md` for detailed status.

## Git Workflow

- **Branch:** `main` is protected (will be, once pushed)
- **Commits:** Conventional Commits format (`feat(module): description`, `fix:`, `chore:`, `docs:`)
- **PRs:** 1 approving review, CI green (when team is active)
- **Small, focused commits** preferred over large batches

## Support

- **Full spec:** `docs/SPEC.md`
- **Implementation plan:** `docs/IMPLEMENTATION_PLAN.md`
- **Architecture:** `docs/architecture.md` (to be written in Phase 2)
- **Issues:** Report at repository Issues page
