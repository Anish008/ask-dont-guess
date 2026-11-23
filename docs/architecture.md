# System Architecture

**Status:** To be completed in Phase 2

This document will describe the complete system architecture for Ask, Don't Guess, including:

1. **High-level Architecture**
   - Component diagram
   - Data flow diagram
   - Request lifecycle

2. **Core Components**
   - LLM inference layer (backends, prompts, parsing)
   - NLI scoring (DeBERTa-MNLI)
   - Semantic clustering (relations, algorithms, features)
   - Decision engine (rules, calibration)
   - Clarification generation and resolution
   - Conversation management

3. **Methods Layer**
   - Baselines (Always Answer, Ask-If-Unclear, Confidence, LogProb, Semantic Entropy)
   - ADG-Core (proposal-faithful)
   - ADG+IC (with interpretation channel)

4. **Evaluation Framework**
   - Dataset pipeline
   - Simulated user
   - Metrics computation
   - Experiment runner
   - Statistical analysis

5. **API and UI**
   - FastAPI service with SSE
   - Next.js dashboard (optional)
   - Gradio demo

6. **Design Decisions**
   - Why these technologies
   - Trade-offs made
   - Alternatives considered

---

**Mermaid diagrams and detailed component descriptions will be added during Phase 2 implementation.**

For the complete specification, see `SPEC.md` §3.
