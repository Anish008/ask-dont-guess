# ADG Demo Script

**Duration:** 3-5 minutes
**Audience:** Technical audience, researchers, potential users
**Goal:** Demonstrate ADG's ability to detect ambiguity, ask clarifying questions, and abstain honestly

---

## Setup (Before Demo)

### Prerequisites
1. Run `make demo-mock` to start Gradio interface
2. Navigate to http://localhost:7860
3. Ensure examples are loaded and interface is responsive
4. Have this script visible for reference

### Equipment
- Laptop/screen for presentation
- Browser with Gradio interface open
- Optional: Second screen for showing code/architecture

---

## Demo Flow (3-5 minutes)

### **Part 1: Introduction (30 seconds)**

**Say:**
> "I'm going to show you Ask, Don't Guess - a system that wraps small LLMs to detect when questions are ambiguous. Instead of silently picking one interpretation, it asks clarifying questions. The system makes three decisions: ANSWER directly, ASK for clarification, or ABSTAIN honestly."

**Show:**
- Point to the three action badges: ✅ ANSWER, ❓ ASK, ⚠️ ABSTAIN
- Briefly mention: "This works by sampling multiple answers and clustering them by meaning"

---

### **Part 2: Clear Question → ANSWER (45 seconds)**

**Example:** "What is the capital of France?"

**Say:**
> "Let's start with a clear, unambiguous question."

**Type:** `What is the capital of France?`

**Click:** Ask button

**Wait** (~2 seconds for mock backend)

**Show:**
- Action badge: ✅ ANSWER
- Response: "The capital of France is Paris."
- **Click** "Decision Details" accordion

**Point out:**
- Confidence scores (ANSWER should be highest)
- Decision rule: "answer" rule fired
- Key features:
  - p1 (largest cluster share): ~1.0 (consensus)
  - delta (separation): high (clear winner)
  - Number of clusters: 1

**Say:**
> "Notice all samples agreed on one meaning - Paris. The system confidently answers directly."

---

### **Part 3: Ambiguous Question → ASK (90 seconds)**

**Example:** "What is the capital of Georgia?"

**Say:**
> "Now let's try an ambiguous question. Georgia could be the US state or the country."

**Type:** `What is the capital of Georgia?`

**Click:** Ask button

**Wait** (~2 seconds)

**Show:**
- Action badge: ❓ ASK
- Response: "Do you mean Georgia the US state or Georgia the country?"
- Options: "the US state" / "the country"

**Point out in Decision Details:**
- Ask score is high (multiple interpretations)
- p1 and p2 are both substantial (two clusters)
- delta is small (not a clear winner)
- Clusters section shows TWO clusters:
  - Cluster 0: "Atlanta" (US state interpretation)
  - Cluster 1: "Tbilisi" (country interpretation)

**Say:**
> "The system detected two distinct interpretations and built a targeted clarifying question from the actual meaning clusters."

**Click:** "the country" option

**Wait** (~1 second)

**Show:**
- Final answer: "The capital of Georgia is Tbilisi."

**Say:**
> "After clarification, it answers for the chosen interpretation. This is multi-turn resolution."

---

### **Part 4: Unknown Question → ABSTAIN (45 seconds)**

**Example:** "What temperature was recorded at a private research station at 3:17 AM on March 15, 2019?"

**Say:**
> "Finally, let's ask something the model doesn't know."

**Type:** `What temperature was recorded at a private research station at 3:17 AM on March 15, 2019?`

**Click:** Ask button

**Wait** (~2 seconds)

**Show:**
- Action badge: ⚠️ ABSTAIN
- Response: "I'm not confident enough to answer this question."

**Point out in Decision Details:**
- Low confidence for all actions
- Multiple small clusters or scattered samples
- Coverage is low (no dominant interpretation)

**Say:**
> "When the model doesn't know, it's honest instead of making something up. The samples were scattered - a sign of ignorance, not ambiguity."

---

### **Part 5: How It Works (30 seconds)**

**Say:**
> "The key insight: ADG preserves the structure of meaning clusters, not just entropy. This lets us distinguish:
> - One big cluster → Answer confidently
> - A few distinct clusters → Ambiguity → Ask
> - Many scattered clusters → Ignorance → Abstain"

**Optional:** Show the examples list at bottom of interface to demonstrate the curated test cases

---

## Timing Breakdown

| Section | Time | Cumulative |
|---------|------|------------|
| Introduction | 0:30 | 0:30 |
| Clear Question (ANSWER) | 0:45 | 1:15 |
| Ambiguous Question (ASK) | 1:30 | 2:45 |
| Unknown Question (ABSTAIN) | 0:45 | 3:30 |
| How It Works | 0:30 | 4:00 |
| **Total** | **4:00** | |

*Add 30-60s for Q&A buffer*

---

## Extended Demo (If Time Permits)

### Additional Examples (1-2 minutes each)

**More Ambiguous Questions:**
- "How do I program in Java?" (language vs island)
- "Tell me about Mercury" (planet vs element)

**More Clear Questions:**
- "What is 2 + 2?" (simple arithmetic)
- "What color is the sky on a clear day?" (common knowledge)

**More Unknown Questions:**
- "What will be the winning lottery numbers next week?" (future prediction)
- "What is my grandmother's favorite color?" (personal information)

### API Demo (If Technical Audience)

**Show:**
```bash
# Terminal window
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of Georgia?"}'
```

**Say:**
> "ADG also exposes a REST API for integration into applications."

---

## Troubleshooting

### If Gradio Doesn't Start
**Fallback:** Use screenshots (see `docs/assets/screenshots/`)
1. Show ANSWER screenshot with decision panel
2. Show ASK screenshot with clusters
3. Show ABSTAIN screenshot

### If Examples Don't Work
**Backup queries:**
- Clear: "What is the capital of France?"
- Ambiguous: "What is the capital of Georgia?"
- Unknown: "What was recorded at the station on March 15, 2019?"

### If Demo Runs Long
**Skip:**
- Decision Details deep dive (just mention it exists)
- Extended examples
- API demo

---

## Q&A Preparation

### Expected Questions

**Q: Does this work with any LLM?**
A: Yes, it's training-free and wraps any instruction-tuned LLM. We tested with 1.5B-3B models where uncertainty is higher.

**Q: How much does it cost?**
A: About 8x the cost of single-answer generation (8 samples vs 1). But you get better reliability and user experience.

**Q: What about latency?**
A: 2-3 seconds on GPU, 15-30s on CPU. Can be parallelized with vLLM for production.

**Q: Can it handle multi-way ambiguity?**
A: Yes, though it works best with 2-3 interpretations. With 4+ interpretations, it may abstain.

**Q: How is this different from semantic entropy?**
A: Semantic entropy compresses to one number (entropy). We keep the cluster structure to separate ambiguity (few distinct clusters) from ignorance (many scattered clusters).

**Q: Does it need training data?**
A: Only for calibrating decision thresholds (~500 examples). The LLM itself is never fine-tuned.

**Q: What about other languages?**
A: Currently English-only due to NLI model. Multilingual NLI models exist but untested.

---

## Post-Demo Resources

**Point audience to:**
- GitHub: https://github.com/adg-team/ask-dont-guess
- Documentation: `docs/algorithm.md` for technical details
- Paper/Report: (link when available)
- Try it: `make demo-mock` to run locally

**Key takeaways to emphasize:**
1. ADG detects ambiguity that LLMs miss
2. Targeted clarifying questions from actual meaning clusters
3. Honest abstention when model doesn't know
4. Training-free, works with small LLMs
5. Production-ready with API and comprehensive testing

---

## Notes

- **Pace:** Speak clearly but don't rush. 4 minutes is sufficient.
- **Focus:** Emphasize the problem (silent guessing) and solution (ask, don't guess)
- **Visuals:** The cluster visualization in Decision Details is powerful - use it
- **Relatability:** Everyone has experienced ambiguous questions from voice assistants
- **Technical depth:** Adjust based on audience (less for general, more for ML researchers)

---

## Rehearsal Checklist

- [ ] Demo runs in under 5 minutes
- [ ] All 3 examples work correctly
- [ ] Decision Details panel opens smoothly
- [ ] Transitions between examples are clear
- [ ] Can answer expected questions
- [ ] Have backup plan if tech fails
- [ ] Know how to restart Gradio if needed (`Ctrl+C`, then `make demo-mock`)

---

**Version:** 1.0
**Last Updated:** November 24, 2024
**Hardware Tested:** CPU (mock mode), GPU with real models
