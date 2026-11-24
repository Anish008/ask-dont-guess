# Implementation Checklist
Last updated: 2024-11-24, Phase 9 COMPLETE ✓

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

### Phase 3 - Data Models & Interfaces ✓
- [x] core/types.py with 20+ pydantic models (§4.1) — Query, Sample, ADGResult, all data structures
- [x] core/errors.py with exception hierarchy — ADGError base, 15+ typed exceptions with codes/hints
- [x] core/protocols.py with Protocol definitions — LLMBackend, NLIScorer, Method, SimulatedUser
- [x] core/registry.py for backend/method registration — 4 registry classes with decorator support
- [x] core/__init__.py exports — All types, errors, protocols, registries exported
- [x] Tests written for all Phase 3 modules (63 tests) — test_types, test_errors, test_protocols, test_registry
- [x] Lint passes — `ruff check` returns "All checks passed!"
- [x] Type check passes — `mypy` returns "Success" (1 non-blocking psutil stub warning)
- [x] Tests pass (89/89) with 94% coverage — All Phase 1-3 tests passing

### Phase 4 - LLM Inference ✓
- [x] llm/prompts.py with structured/plain templates — format_structured_prompt, format_plain_prompt, format_clarification_prompt
- [x] llm/parsing.py for output extraction — parse_structured_output with refusal/hedge detection
- [x] llm/mock_backend.py (deterministic test backend) — Hash-based deterministic sampling, no model loading
- [x] llm/hf_backend.py (Transformers backend) — HuggingFace models with fp16/4-bit quantization, tenacity retry
- [x] sampling/sampler.py with seeded sampling — Seed derivation from query_id, delegates to LLMBackend
- [x] llm/register.py for backend registration — Registers mock and hf backends
- [x] Tests written for all Phase 4 modules (37 tests) — test_prompts, test_parsing, test_mock_backend, test_sampler
- [x] Lint passes — `ruff check` returns "All checks passed!"
- [x] Type check passes — `mypy` returns "Success: no issues found"
- [x] Tests pass (126/126) with 85% coverage — All Phase 1-4 tests passing

### Phase 5 - NLI Inference ✓
- [x] nli/mock_nli.py (rule-based test scorer) — Jaccard similarity-based mock scorer, no model loading
- [x] nli/hf_nli.py (DeBERTa scorer) — Bidirectional entailment scoring with microsoft/deberta-large-mnli
- [x] nli/register.py for NLI scorer registration — Registers mock and hf NLI scorers
- [x] nli/__init__.py exports — HFNLIScorer, MockNLIScorer
- [x] Tests written for all Phase 5 modules (18 tests) — test_mock_nli, test_hf_nli
- [x] Lint passes — `ruff check` returns "All checks passed!"
- [x] Type check passes — `mypy` returns "Success: no issues found"
- [x] Tests pass (144/144) with 78% coverage — All Phase 1-5 tests passing

### Phase 6 - Semantic Clustering ✓
- [x] clustering/relations.py (pairwise NLI scoring with deduplication) — compute_pairwise_relations with identical answer shortcuts
- [x] clustering/algorithms.py (3 clustering algorithms) — greedy_bidirectional, agglomerative, connected_components
- [x] clustering/features.py (aggregate features computation) — compute_aggregate_features with p1, p2, coverage, separation
- [x] clustering/interpretation.py (interpretation channel) — compute_interpretation_divergence for detecting multiple interpretations
- [x] clustering/__init__.py exports — All 4 main functions exported
- [x] Tests written for all Phase 6 modules (33 tests) — test_relations, test_algorithms, test_features, test_interpretation
- [x] Tests pass (177/177) with 55% coverage — All Phase 1-6 tests passing

### Phase 7 - Decision Engine ✓
- [x] decision/rules.py (ordered rule list with thresholds) — 4 rules: abstain_refusal, abstain_invalid, ask, answer
- [x] decision/engine.py (DecisionEngine with confidence scores) — Applies rules, computes normalized confidence scores
- [x] decision/calibration.py (threshold calibration via grid search) — ThresholdCalibrator with accuracy, precision, F1 metrics
- [x] decision/__init__.py exports — All decision functions and classes exported
- [x] core/types.py Decision model added — action, confidence, confidence_scores, rule_name
- [x] Tests written for all Phase 7 modules (33 tests) — test_rules, test_engine, test_calibration
- [x] Tests pass (210/210) with 36% coverage — All Phase 1-7 tests passing

### Phase 8 - Clarification Generation ✓
- [x] clarify/generator.py (template + LLM modes) — QuestionGenerator with template and LLM-based generation
- [x] clarify/interpretation.py (ReplyResolver) — Resolves user replies via numeric, exact, fuzzy, and fallback methods
- [x] clarify/__init__.py exports — QuestionGenerator, ReplyResolver exported
- [x] Tests written for all Phase 8 modules (41 tests) — test_generator, test_interpretation
- [x] Tests pass (251/251) with 85% coverage — All Phase 1-8 tests passing

### Phase 9 - Conversation Management ✓
- [x] conversation/state.py (SessionState management) — Helper functions for managing session state, turns, and expiration
- [x] conversation/manager.py (ConversationManager) — Multi-turn orchestration with full pipeline integration
- [x] conversation/__init__.py exports — ConversationManager and state management functions exported
- [x] Tests written for all Phase 9 modules (39 tests) — test_state, test_manager
- [x] Tests pass (290/290) with 86% coverage — All Phase 1-9 tests passing

## Remaining Features (by Phase)


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
| 3 | Data Models & Interfaces | 63 | 0 | 94% | `pytest tests/ --cov` |
| 4 | LLM Inference | 37 | 0 | 63% | `pytest tests/unit/llm tests/unit/sampling --cov` |
| 5 | NLI Inference | 18 | 0 | 45% | `pytest tests/unit/nli --cov` |
| 6 | Semantic Clustering | 33 | 0 | 55% | `pytest tests/unit/clustering --cov` |
| 7 | Decision Engine | 33 | 0 | 36% | `pytest tests/unit/decision --cov` |
| 8 | Clarification | 41 | 0 | 35% | `pytest tests/unit/clarify --cov` |
| 9 | Conversation | 39 | 0 | 60% | `pytest tests/unit/conversation --cov` |
| **Total** | **All** | **290** | **0** | **86%** | `pytest tests/ --cov` |

## Known Bugs
(None yet)

## Current Limitations
- Phases 1-5 complete: environment, configuration, data models, LLM inference, and NLI inference ready
- Real model testing pending (HF backends implemented, not yet tested with actual model downloads)
- Clustering algorithms not yet implemented (will be in Phase 6)
- No dataset access yet (will be verified in Phase 11)
- Uncalibrated decision thresholds (will be calibrated in Phase 7)

## Deviations from Spec
See `docs/DEVIATIONS.md` (none yet)

## Next Steps
1. Git commit Phase 5 completion
2. Begin Phase 6: Semantic Clustering
   - Implement clustering/relations.py for pairwise NLI scoring with deduplication
   - Implement clustering/algorithms.py with 3 clustering algorithms (greedy_bidirectional, agglomerative, connected_components)
   - Implement clustering/features.py for ClusterFeatures computation
   - Implement clustering/interpretation.py for interpretation channel (ER)
   - Write comprehensive tests for clustering
3. Proceed to Phase 7: Decision Engine
