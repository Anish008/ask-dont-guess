# Deviations from Specification

**Last updated:** 2024-11-23
**Purpose:** Document any cases where reality differs from the spec, and how we adapted.

Per spec §0.1 point 9: "When the spec and reality disagree (a dataset field is named differently, a model lacks a feature), adapt the implementation, document the deviation here, and tell the user."

---

## Current Deviations

### [Phase 1] Windows Compatibility - vLLM and Quantization Extras Skipped
- **Spec expected:** All optional extras installable, including `[vllm]` and `[quant]`
- **Reality:**
  - vLLM has Linux-only dependencies (nccl-extensions)
  - bitsandbytes Windows support unclear at time of setup
- **Adaptation:**
  - Installed core + dev + serve + openai + embed extras successfully
  - Skipped vllm (Linux-only, documented in spec §2.3 anyway)
  - Skipped quant temporarily (will retry in Phase 4-5 when actually loading models)
  - Skipped tracking (not priority per user decision)
- **Impact:** Minimal
  - vLLM was always optional and Linux-only
  - 4-bit quantization can be attempted later when needed for 3B models
  - Core functionality and all priority features unaffected
- **Verification:** `uv sync` with selected extras completed successfully
- **Date:** 2024-11-23

---

## Format for Recording Deviations

When adding a deviation, use this format:

### [Phase X] Brief Title
- **Spec expected:** What the specification said
- **Reality:** What we actually found
- **Adaptation:** How we changed the implementation
- **Impact:** What this affects (minimal / moderate / significant)
- **Verification:** How this was confirmed
- **Date:** YYYY-MM-DD

---

## Example (for reference)

### [Phase 11] AmbigQA Field Name Difference
- **Spec expected:** Field named `disambiguated_questions`
- **Reality:** Actual field is named `qaPairs`
- **Adaptation:** Updated loader to use `qaPairs` and map to internal schema
- **Impact:** Minimal - internal schema unaffected
- **Verification:** Loaded dataset and printed `dataset[0].keys()`
- **Date:** 2024-XX-XX

---

## Non-Deviations (Intentional Design Choices)

These are intentional choices, not deviations:

1. **ADG+IC variant with interpretation channel (§0.8)**: This is marked [ER] (Engineering Recommendation) in the spec and is implemented alongside ADG-Core as planned.

2. **Using Python 3.11 instead of 3.14**: User decision to install 3.11 for maximum compatibility, as spec requires 3.11-3.12.

3. **Skipping Phase 19 (DPO)**: User decision to focus on phases 1-18, clearly documented in implementation plan.

4. **Gradio demo priority over Next.js**: User decision to build Gradio first (Phase 15), Next.js only if time permits (Phase 16).
