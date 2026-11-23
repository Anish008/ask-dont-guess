# Implementation Checklist
Last updated: 2024-11-23, Phase 1 COMPLETE ✓

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

## Remaining Features (by Phase)

### Phase 2 - Environment & Configuration
- [ ] config/schema.py with pydantic Settings models
- [ ] config/loader.py with YAML loading and validation
- [ ] core/logging.py with structlog setup
- [ ] core/seeding.py with reproducible seed derivation
- [ ] core/hashing.py for content-addressed cache keys
- [ ] core/device.py for hardware detection

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

| Suite | Passed | Failed | Skipped | Command |
|-------|--------|--------|---------|---------|
| (Phase 1 pending) | - | - | - | `make test` |

## Known Bugs
(None yet)

## Current Limitations
- Phase 1 only: basic structure, no functionality yet
- No models loaded yet (will be verified in Phase 4-5)
- No dataset access yet (will be verified in Phase 11)

## Deviations from Spec
See `docs/DEVIATIONS.md` (none yet)

## Next Steps
1. Complete Phase 1: .editorconfig, pre-commit, CI, README, LICENSE, CHANGELOG
2. Run `make setup` to install dependencies
3. Verify all Phase 1 targets pass
4. Git commit Phase 1 completion
5. Begin Phase 2: Environment & Configuration
