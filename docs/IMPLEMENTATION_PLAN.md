# Ask, Don't Guess (ADG) - Phase 1 Implementation Plan

## Project Overview
Building a production-quality, research-grade Python system for a university AI course that wraps small LLMs to decide when to ANSWER, ASK clarifying questions, or ABSTAIN from answering.

## Environment Assessment

### Hardware & Software
- **OS:** Windows x64
- **Python:** 3.14.2 (spec requests 3.11-3.12, but 3.14 should work with pinning)
- **GPU:** NVIDIA RTX 5070 with 8GB VRAM + CUDA 13.5
  - **Perfect fit** for Qwen2.5-1.5B/3B in fp16 or 4-bit (matches spec §15.2 "8 GB GPU" profile)
- **RAM:** 32 GB (excellent)
- **Disk:** 413 GB free (sufficient)
- **Node.js:** v24.12.0, npm 11.7.0 (ready for frontend)
- **API Access:** HF_TOKEN ✓, OPENAI_API_KEY ✓

### Critical Missing Dependencies
- `uv` (required by spec for dependency management)
- PyTorch, transformers, and all Python ML packages
- Git repository initialization

### Hardware Recommendation
Based on 8GB VRAM:
- **Primary model:** Qwen/Qwen2.5-1.5B-Instruct (fp16, ~3GB)
- **NLI model:** microsoft/deberta-large-mnli (fp16, ~1.6GB) or deberta-v3-base for CPU dev
- **Secondary model:** Qwen2.5-3B (4-bit, ~2GB) or fp16 (~6-7GB, tight but feasible)
- **Hardware config:** `configs/hardware/gpu_8gb.yaml` (may need 4-bit for 3B model)

## Phase 1: Repository Inspection & Project Setup

### Objectives (from spec §20, Phase 1)
1. Complete repository and environment inspection (✓ partially done)
2. Create full repository skeleton following spec §5
3. Set up development environment with uv
4. Configure essential tooling (git, pre-commit, CI templates)
5. Create foundational docs (CLAUDE.md, IMPLEMENTATION_CHECKLIST.md, stubs for others)
6. Write comprehensive IMPLEMENTATION_PLAN.md (this becomes docs/IMPLEMENTATION_PLAN.md)
7. Get user confirmation before proceeding

### Key Decisions & Adaptations

#### 1. Python Version Compatibility
**Issue:** Environment has Python 3.14.2, spec requests 3.11-3.12
**Decision:** Use Python 3.14 but pin requires-python = ">=3.11,<3.15" in pyproject.toml. Test that pydantic v2, PyTorch ≥2.3, and transformers ≥4.45 work. If incompatible, will document deviation.

#### 2. Missing uv
**Issue:** uv not installed
**Decision:** Install via `pip install uv` as first step. Document this requirement prominently in README.

#### 3. Git Repository
**Issue:** Not a git repo
**Decision:** Initialize with `git init`, create .gitignore, make initial commit with spec files, then scaffold structure.

#### 4. PONYTAIL MODE Active
**Context:** Startup hook indicates PONYTAIL MODE at "full" level
**Implication:** Follow lazy/efficient principles:
  - Reuse existing work (the two spec files)
  - Minimal initial structure (expand only as needed)
  - No placeholder implementations in core (spec §0.6 hard rule)
  - Start with what's needed for Phase 1 completion criteria

### Phase 1 Verification Criteria (from spec §20)
✓ User confirms this plan
⟐ `make setup` runs (creates env, installs deps)
⟐ `make lint`, `make typecheck`, `make test` pass (trivial tests only at first)
⟐ CI workflow exists and passes on first push
⟐ All docs stubs created with structure
⟐ pyproject.toml validates
⟐ CLAUDE.md, IMPLEMENTATION_CHECKLIST.md, DEVIATIONS.md, VERIFIED_FACTS.md exist

### Detailed Implementation Steps

#### Step 0: Verify/Install Python 3.11 or 3.12
**User Decision:** Install Python 3.11 or 3.12 for this project (not 3.14)

**Action:** Check if Python 3.11 or 3.12 is available:
```bash
# Check available Python versions
python3.11 --version  # Try 3.11
python3.12 --version  # Try 3.12
py -3.11 --version    # Windows py launcher
py -3.12 --version
```

**If neither available:** Guide user to install Python 3.11 or 3.12:
- Windows: Download from python.org or use `winget install Python.Python.3.11`
- Recommend 3.11.x (LTS, widest ML package support)

**Once available:** Use that Python for all subsequent commands
- Example: `python3.11 -m pip install uv` or `py -3.11 -m pip install uv`

#### Step 1: Install uv and Initialize Repository
```bash
# Install uv (using correct Python version)
python3.11 -m pip install uv  # or python3.12

# Initialize git
git init
git config user.name "ADG Team"
git config user.email "team@example.com"

# Create .gitignore (spec §5)
# Move spec files to docs/
mkdir docs
mv Ask_Dont_Guess_ClaudeCode_Prompt.md docs/SPEC.md
mv Ask_Dont_Guess_Full_Deliverable.md docs/SPEC_FULL.md

# Initial commit
git add .gitignore docs/
git commit -m "chore: initialize repository with specification"
```

#### Step 2: Create Minimal Repository Structure
Following spec §5, create these directories and essential files ONLY:
```
ask-dont-guess/
├── README.md                     # Minimal, states "Results pending"
├── CLAUDE.md                     # Per spec §0.4
├── LICENSE                       # Apache-2.0 or MIT (ask user)
├── CHANGELOG.md                  # Empty structure
├── pyproject.toml                # Full structure, correct deps
├── Makefile                      # setup, lint, format, typecheck, test targets
├── .gitignore                    # Per spec requirements
├── .editorconfig
├── .pre-commit-config.yaml       # ruff, mypy (core only), detect-secrets
├── .github/workflows/ci.yml      # Minimal: lint, typecheck, test
├── configs/
│   └── base.yaml                 # Stub with comments
├── src/adg/
│   ├── __init__.py               # __version__ only
│   ├── py.typed
│   └── core/
│       ├── __init__.py
│       └── types.py              # Empty structure, imports only
├── tests/
│   ├── conftest.py               # Minimal fixtures
│   └── test_imports.py           # Trivial: import adg succeeds
└── docs/
    ├── SPEC.md                   # Already exists
    ├── IMPLEMENTATION_PLAN.md    # This file, copied from plan mode
    ├── IMPLEMENTATION_CHECKLIST.md  # Per spec §0.7 structure
    ├── VERIFIED_FACTS.md         # Empty with header
    ├── DEVIATIONS.md             # Empty with header
    └── architecture.md           # Stub
```

**Rationale (PONYTAIL):** Start with the minimum needed for Phase 1 verification. Don't create every single directory and stub file from spec §5 yet - expand in later phases as features are built. This avoids premature structure.

#### Step 3: Configure pyproject.toml
Per spec §2, separate core deps from extras:
- **Core:** python >=3.11,<3.15, torch>=2.3, transformers>=4.45, accelerate, datasets, pydantic>=2.6, pydantic-settings, pyyaml, typer, rich, numpy, scipy, pandas, scikit-learn, structlog, tenacity, diskcache, matplotlib, orjson
- **Extras:** [serve], [openai], [quant], [vllm], [tracking], [embed], [dev], [dpo]
- **Build system:** hatchling (for src layout)
- **Tool config:** ruff (line-length=100, E,F,I,B,UP,SIM,RUF), mypy (strict on core), pytest (markers for slow tests)

**Critical:** Verify exact model IDs in comments as TODO(verify) since we can't load models yet.

#### Step 4: Create Makefile
Targets needed for Phase 1:
```makefile
.PHONY: setup lint format typecheck test clean

setup:
    uv sync --all-extras
    uv run pre-commit install

lint:
    uv run ruff check src/ tests/

format:
    uv run ruff format src/ tests/

typecheck:
    uv run mypy src/adg/core/

test:
    uv run pytest tests/ -v

clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
```

#### Step 5: Create Foundational Documentation

**CLAUDE.md** (spec §0.4):
- One paragraph project description
- Link to docs/SPEC.md
- Commands: make setup, test, lint, typecheck, demo-mock, demo
- Coding conventions from spec §16
- Never fabricate rule

**IMPLEMENTATION_CHECKLIST.md** (spec §0.7):
- Structure per spec
- Phase 1 items as "in progress"
- Tests table (empty initially)

**VERIFIED_FACTS.md:**
```markdown
# Verified Facts (never fabricated)
Last updated: <date>

## Models
(To be verified in Phase 4-5 when loading models)

## Datasets
(To be verified in Phase 11 when loading data)

## NLI Model
(To be verified in Phase 5)

## API Models
(To be verified if/when using OpenAI)
```

**DEVIATIONS.md:**
```markdown
# Deviations from Specification
(None yet - will be populated as needed)
```

#### Step 6: Minimal CI Workflow
`.github/workflows/ci.yml` with:
- Lint (ruff check + format check)
- Typecheck (mypy on core only)
- Test (pytest with mock mode, no model downloads)
- Matrix: Python 3.11 and 3.12 (not 3.14 since spec requires 3.11-3.12)
- Cache uv deps
- Badge in README

#### Step 7: Write docs/IMPLEMENTATION_PLAN.md
Copy this plan file into the repository as the official implementation plan.

### Risks and Open Questions

#### Risks
1. **Python 3.14 compatibility:** Some deps might not support it yet
   - **Mitigation:** Test immediately after setup; fall back to 3.11/3.12 if needed

2. **8GB VRAM tight for 3B models:** May need 4-bit quantization
   - **Mitigation:** Start with 1.5B; add 3B in 4-bit in Phase 4-5; measure memory

3. **Windows-specific issues:** Some tools (vLLM, bitsandbytes) may not work
   - **Mitigation:** Document clearly; bitsandbytes Windows support unclear; vLLM Linux-only

4. **Large model downloads:** Each model is 3-7 GB
   - **Mitigation:** Ask user before downloading (spec §0.1 point 6); use mock mode for dev

#### User Decisions (confirmed)

**License:** Apache-2.0 ✓

**Python Version:** Install Python 3.11 or 3.12 for this project ✓
- Action: Verify Python 3.11 or 3.12 is available, or guide installation
- pyproject.toml: requires-python = ">=3.11,<3.13" (strict)

**Priority Optional Features:**
- [quant] 4-bit quantization: **HIGH PRIORITY** ✓ (needed for 3B models in 8GB VRAM)
- [openai] GPT-4o-mini: **HIGH PRIORITY** ✓ (simulated user and judge)
- [vllm] skip initially (Linux-only, Windows environment)
- [tracking] skip initially (local run directories sufficient)

**Project Scope:** Phases 1-18, skip only Phase 19 (DPO extension) ✓
- **Timeline:** Within one week (aggressive but achievable with focused work)
- **Phase 15 (UI):** Gradio demo first (quick), Next.js only if time permits after experiments
- **Phase 17 (docs):** Keep lean - README, algorithm, reproduction, limitations (no fluff)
- **Priority order if time tight:**
  1. Core algorithm + baselines + evaluation (phases 1-13) ← MUST HAVE
  2. Results tables and reporting (phase 14) ← REQUIRED for report
  3. Demo (phase 15, Gradio) ← Nice to have
  4. Frontend (phase 16) ← Only if time
  5. Docs polish (phase 17) ← Lean version sufficient
  6. Demo prep (phase 18) ← Final touches

**Model Priority:**
- Primary: Qwen/Qwen2.5-1.5B-Instruct (fp16, ~3GB)
- Secondary: Qwen/Qwen2.5-3B-Instruct (4-bit, ~2GB) with bitsandbytes
- Tertiary: Llama-3.2-3B if time allows (gated, requires HF license acceptance)

### Post-Phase-1 Preview

**Phase 2:** Environment & configuration (config loading, structlog, device detection)
**Phase 3:** Data models & interfaces (pydantic types, protocols, registries)
**Phase 4:** LLM inference (HF backend, mock, prompts, parsing) ← First model download
**Phase 5:** NLI inference (DeBERTa scorer, mock) ← Second model download
**Phases 6-9:** Core algorithm (clustering, decision, clarification, conversation)
**Phase 10:** Algorithm testing with mocks
**Phases 11-13:** Datasets, baselines, evaluation
**Phases 14-16:** Results, API, frontend
**Phases 17-19:** Docs, demo, optional DPO

**Critical path:** Phases 1-10 (core algorithm) must be rock-solid before evaluation.

### Success Criteria for Phase 1
- [ ] User approves this plan
- [ ] uv installed and working
- [ ] Git repo initialized with clean history
- [ ] Repository structure created (minimal per PONYTAIL)
- [ ] pyproject.toml validates (`uv sync --dry-run` succeeds)
- [ ] `make setup` completes without errors
- [ ] `make lint` passes (no files to lint yet, but command works)
- [ ] `make typecheck` passes on minimal core/types.py
- [ ] `make test` passes (trivial import test)
- [ ] CI workflow exists (even if not pushed yet)
- [ ] CLAUDE.md, IMPLEMENTATION_CHECKLIST.md, VERIFIED_FACTS.md, DEVIATIONS.md exist
- [ ] This plan copied to docs/IMPLEMENTATION_PLAN.md
- [ ] Git commit: "chore: phase 1 complete - project setup"

### Verification Commands
After Phase 1 completion:
```bash
# Verify setup
make setup
uv sync --all-extras  # Should succeed

# Verify tooling
make lint             # Should pass
make format          # Should pass
make typecheck       # Should pass on core/
make test            # Should pass trivial tests

# Verify git
git log --oneline    # Should show clean commits
git status           # Should be clean

# Verify docs
cat docs/IMPLEMENTATION_CHECKLIST.md  # Phase 1 should be marked complete
cat CLAUDE.md                          # Should have all required sections
```

## Notes
- Following PONYTAIL mode (full): minimal structure, no premature abstractions, expand as needed
- Strictly adhering to spec §0.6: NO placeholder implementations in core functionality
- All model IDs, dataset schemas will be verified when actually loaded (phases 4-5, 11)
- README will show "Results pending" until real experiments run (phase 14+)
- Mock mode will be the default for all development until real models tested

## Immediate Next Steps After Approval

1. **Check Python 3.11/3.12 availability** → install if needed
2. **Install uv** using correct Python version
3. **Initialize git repository** with clean .gitignore
4. **Create minimal directory structure** (PONYTAIL: only what Phase 1 needs)
5. **Write pyproject.toml** with correct deps (core + [openai] + [quant] + [dev])
6. **Run `make setup`** to verify environment
7. **Create foundational docs** (CLAUDE.md, checklists, VERIFIED_FACTS.md, DEVIATIONS.md)
8. **Set up CI workflow** (.github/workflows/ci.yml)
9. **Copy this plan** to docs/IMPLEMENTATION_PLAN.md
10. **Git commit** with "chore: phase 1 complete - project setup"
11. **Run all Phase 1 verification commands** to confirm success
12. **Update IMPLEMENTATION_CHECKLIST.md** marking Phase 1 complete
13. **Begin Phase 2** (Environment & configuration)

## Estimated Time for Phase 1
- With user available for decisions: **1-2 hours**
- Includes Python setup, structure creation, verification

## Ready to Proceed
This plan follows all requirements from spec §0.5 and §20 Phase 1. Upon approval:
- Will create production-quality, research-grade setup
- No shortcuts on core algorithm (per §0.6)
- Will verify everything, fabricate nothing (per §0.1 point 3)
- Will work in phases with clean commits (per §0.1 point 2)
- Targets phases 1-18 completion within one week
