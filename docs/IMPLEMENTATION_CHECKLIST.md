# Implementation Checklist
Last updated: 2024-11-23, Phase 2 COMPLETE ✓

## Completed Features

### Phase 1 - Repository Inspection & Project Setup ✓
- [x] Python 3.11.9 installed and verified — `py -3.11 --version`
- [x] uv 0.13.9 package manager installed — `py -3.11 -m pip show uv`
- [x] Git repository initialized with clean history — `git log`
- [x] Directory structure created (src, tests, configs, docs, .github) — file tree verified
- [x] pyproject.toml with dependencies and tool config — `uv sync` succeeds
- [x] Makefile with all targets — file exists
- [x] .gitignore with comprehensive patterns — file exists
- [x] .editorconfig — file exists
- [x] .pre-commit-config.yaml — file exists, hooks installed
- [x] CI workflow (.github/workflows/ci.yml) — file exists
- [x] README.md (minimal, with "Results pending") — file exists
- [x] LICENSE (Apache-2.0) — file exists
- [x] CHANGELOG.md structure — file exists
- [x] CLAUDE.md with project guide — file exists
- [x] docs/IMPLEMENTATION_PLAN.md — copied from plan mode
- [x] docs/IMPLEMENTATION_CHECKLIST.md (this file) — updated
- [x] docs/VERIFIED_FACTS.md — created with structure
- [x] docs/DEVIATIONS.md — created, Windows compatibility documented
- [x] docs/architecture.md stub — created
- [x] Dependencies installed (106 packages) — `uv sync --extra dev --extra serve --extra openai --extra embed`
- [x] Pre-commit hooks installed — `pre-commit install`
- [x] Lint passes — `ruff check` returns "All checks passed!"
- [x] Type check passes — `mypy src/adg/core/` returns "Success: no issues found"
- [x] Tests pass (2/2) with 50% coverage — `pytest` returns "2 passed"

### Phase 2 - Environment & Configuration ✓
- [x] config/schema.py with pydantic Settings models — AppConfig with 15 nested configs
- [x] config/loader.py with YAML loading and validation — Supports reference resolution, CLI overrides
- [x] core/logging.py with structlog setup — JSON for prod, pretty console for dev
- [x] core/seeding.py with reproducible seed derivation — Hash-based deterministic seeding
- [x] core/hashing.py for content-addressed cache keys — Stable 16-char hex hashes
- [x] core/device.py for hardware detection — CUDA/MPS/CPU with auto dtype selection
- [x] Hardware configs created (cpu.yaml, gpu_8gb.yaml, gpu_8gb_4bit.yaml, gpu_16gb.yaml, mps.yaml)
- [x] Model configs created (qwen2.5-1.5b.yaml, qwen2.5-3b.yaml, mock.yaml)
- [x] NLI configs created (deberta-large-mnli.yaml, deberta-v3-base.yaml, mock.yaml)
- [x] base.yaml updated with full config structure
- [x] Tests written for all Phase 2 modules (26 tests)
- [x] Lint passes — `ruff check` returns "All checks passed!"
- [x] Type check passes — `mypy` returns "Success: no issues found"
- [x] Tests pass (26/26) with 89% coverage — All verification complete

## Remaining Features (by Phase)

### Phase 3 - Data Models & Interfaces
- [ ] core/types.py with all pydantic models (§4.1)
- [ ] core/errors.py with exception hierarchy
- [ ] Protocol definitions (LLMBackend, NLIScorer, Method, SimulatedUser)
- [ ] Registry patterns for backends and methods

### Phase 4 - LLM Inference
- [ ] llm/base.py with LLMBackend protocol
- [ ] llm/hf_backend.py (Transformers)
- [ ] llm/mock_backend.py (deterministic test backend)
- [ ] llm/prompts/ directory with structured/plain templates
- [ ] llm/parsing.py with interpretation/answer extraction
- [ ] sampling/sampler.py with seeded sampling
- [ ] TODO(verify): Qwen2.5-1.5B-Instruct model ID and config

### Phase 5 - NLI Inference
- [ ] nli/base.py with NLIScorer protocol
- [ ] nli/hf_nli.py (DeBERTa scorer)
- [ ] nli/mock_nli.py (rule-based test scorer)
- [ ] TODO(verify): DeBERTa-MNLI model ID and label mapping

### Phase 6 - Semantic Clustering
- [ ] clustering/relations.py (pairwise NLI, dedup, guard)
- [ ] clustering/algorithms.py (3 algorithms)
- [ ] clustering/features.py (ClusterFeatures computation)
- [ ] clustering/interpretation.py (interpretation channel)

### Phase 7 - Decision Engine
- [ ] decision/rules.py (ordered rule list)
- [ ] decision/engine.py (rule executor + learned mode)
- [ ] decision/calibration.py (threshold search on dev split)

### Phase 8 - Clarification
- [ ] clarify/generator.py (template + LLM modes)
- [ ] clarify/interpretation.py (ReplyResolver)

### Phase 9 - Conversation
- [ ] conversation/state.py (SessionState, Turn models)
- [ ] conversation/manager.py (multi-turn orchestration)

### Phase 10 - Pipeline Integration & Testing
- [ ] pipeline/adg_pipeline.py (full orchestration)
- [ ] Full mock pipeline test with curated examples
- [ ] All §4.9 edge-case tests pass

### Phase 11 - Dataset Pipeline
- [ ] data/loaders/ for AmbigQA, CLAMBER, NQ-open
- [ ] data/schema.py (Example, GoldInterpretation models)
- [ ] data/dedup.py (AmbigQA ⊂ NQ-open removal)
- [ ] data/splits.py (deterministic dev/test splits)
- [ ] TODO(verify): Dataset schemas and licenses

### Phase 12 - Baselines
- [ ] methods/always_answer.py
- [ ] methods/ask_if_unclear.py
- [ ] methods/stated_confidence.py
- [ ] methods/logprob_threshold.py
- [ ] methods/semantic_entropy.py
- [ ] methods/adg_method.py (ADG-Core and ADG+IC)

### Phase 13 - Evaluation
- [ ] simuser/llm_user.py + rule_user.py
- [ ] evaluation/metrics.py (ask P/R/F1, 3-way, E2E)
- [ ] evaluation/e2e.py (multi-turn simulation)
- [ ] evaluation/runner.py (experiments grid)
- [ ] evaluation/stats.py (bootstrap CIs, paired tests)

### Phase 14 - Results & Reporting
- [ ] reporting/tables.py (Markdown, LaTeX, CSV)
- [ ] reporting/plots.py (matplotlib figures)
- [ ] adg report command generates all tables and figures

### Phase 15 - Demo
- [ ] demo/gradio_app.py (quick demo)
- [ ] demo/examples.yaml (curated queries)

### Phase 16 - API & Frontend (Optional)
- [ ] api/app.py (FastAPI service)
- [ ] frontend/ (Next.js, only if time permits)

### Phase 17 - Documentation
- [ ] README.md complete with auto-generated results
- [ ] docs/algorithm.md
- [ ] docs/reproduction.md
- [ ] docs/limitations.md
- [ ] docs/model_card.md
- [ ] docs/data_card.md

### Phase 18 - Demo Preparation
- [ ] docs/demo_script.md
- [ ] Demo video or screenshots

## Tests

| Phase | Suite | Passed | Failed | Coverage | Command |
|-------|-------|--------|--------|----------|---------|
| 1 | Imports | 2 | 0 | 50% | `pytest tests/` |
| 2 | Core + Config | 26 | 0 | 89% | `pytest tests/ --cov` |

## Known Bugs
(None yet)

## Current Limitations
- Phases 1-2 complete: environment and configuration ready
- No models loaded yet (will be verified in Phase 4-5)
- No dataset access yet (will be verified in Phase 11)
- Uncalibrated decision thresholds (will be calibrated in Phase 7)

## Deviations from Spec
See `docs/DEVIATIONS.md` (none yet)

## Next Steps
1. Git commit Phase 2 completion
2. Begin Phase 3: Data Models & Interfaces
   - Create core/types.py with pydantic models for all data structures (§4.1)
   - Create core/errors.py with exception hierarchy
   - Define Protocol interfaces (LLMBackend, NLIScorer, Method, SimulatedUser)
   - Implement registry patterns for backends and methods
3. Proceed to Phase 4: LLM Inference (first model download)
