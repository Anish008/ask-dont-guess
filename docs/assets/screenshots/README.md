# Demo Screenshots Guide

This directory should contain screenshots for the ADG demo fallback plan.

## Required Screenshots

### 1. `01-interface-overview.png`
**What to capture:**
- Full Gradio interface at http://localhost:7860
- Shows main components:
  - Title: "Ask, Don't Guess (ADG)"
  - Input textbox with placeholder
  - Ask button
  - Examples section
  - Response area

**How to take:**
1. Run `make demo-mock`
2. Open http://localhost:7860
3. Take full-window screenshot before entering any query

---

### 2. `02-answer-example.png`
**What to capture:**
- Clear question that gets an ANSWER response
- Shows:
  - Query: "What is the capital of France?"
  - Action badge: ✅ ANSWER
  - Response: "The capital of France is Paris."
  - Decision Details panel (collapsed is fine)

**How to take:**
1. Type "What is the capital of France?"
2. Click Ask
3. Wait for response
4. Take screenshot showing the full response

---

### 3. `03-answer-decision-details.png`
**What to capture:**
- Same ANSWER example with Decision Details expanded
- Shows:
  - Confidence scores (ANSWER highest)
  - Decision rule: "answer"
  - Features: p1 ~1.0, delta high, 1 cluster
  - Cluster visualization (single cluster)

**How to take:**
1. Click "Decision Details" accordion
2. Take screenshot showing expanded details panel

---

### 4. `04-ask-example.png`
**What to capture:**
- Ambiguous question that gets an ASK response
- Shows:
  - Query: "What is the capital of Georgia?"
  - Action badge: ❓ ASK
  - Clarifying question: "Do you mean Georgia the US state or Georgia the country?"
  - Options: "the US state" / "the country" as clickable chips

**How to take:**
1. Clear previous query or refresh page
2. Type "What is the capital of Georgia?"
3. Click Ask
4. Wait for response
5. Take screenshot showing the clarifying question with options

---

### 5. `05-ask-decision-details.png`
**What to capture:**
- Same ASK example with Decision Details expanded
- Shows:
  - Ask score is high
  - p1 and p2 both substantial
  - delta is small
  - TWO clusters visible:
    - Cluster 0: representative answer about Atlanta
    - Cluster 1: representative answer about Tbilisi

**How to take:**
1. Click "Decision Details" accordion
2. Take screenshot showing both clusters clearly

---

### 6. `06-ask-resolution.png`
**What to capture:**
- After clicking an option in the ASK flow
- Shows:
  - Final answer: "The capital of Georgia is Tbilisi."
  - Or: "The capital of Georgia is Atlanta." (if "the US state" was clicked)

**How to take:**
1. From the ASK example, click one of the options (e.g., "the country")
2. Wait for final answer
3. Take screenshot showing the resolved answer

---

### 7. `07-abstain-example.png`
**What to capture:**
- Unknown/unanswerable question that gets ABSTAIN
- Shows:
  - Query: "What temperature was recorded at a private research station at 3:17 AM on March 15, 2019?"
  - Action badge: ⚠️ ABSTAIN
  - Response: "I'm not confident enough to answer this question."

**How to take:**
1. Clear previous query or refresh page
2. Type the unknown question
3. Click Ask
4. Wait for response
5. Take screenshot showing the abstain response

---

### 8. `08-abstain-decision-details.png`
**What to capture:**
- Same ABSTAIN example with Decision Details expanded
- Shows:
  - Low confidence scores across all actions
  - Multiple small clusters or scattered samples
  - Low coverage

**How to take:**
1. Click "Decision Details" accordion
2. Take screenshot showing decision details for ABSTAIN

---

### 9. `09-examples-panel.png`
**What to capture:**
- The Examples section at bottom of interface
- Shows curated example questions

**How to take:**
1. Scroll to examples section if needed
2. Take screenshot showing the example questions

---

### 10. `10-api-call.png` (Optional)
**What to capture:**
- Terminal window showing API call and response
- Shows:
  ```bash
  curl -X POST http://localhost:8000/v1/analyze \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the capital of Georgia?"}'
  ```
  - JSON response with decision and options

**How to take:**
1. Run `make api-dev` in one terminal
2. In another terminal, run the curl command
3. Take screenshot of the curl command and JSON response

---

## Screenshot Specifications

**Format:** PNG (lossless, good for text)
**Resolution:** At least 1920x1080 (or native screen resolution)
**Window size:** Maximize browser window for consistency
**Browser:** Use Chrome or Firefox for best Gradio rendering
**Annotations:** None needed (keep screenshots clean)

## File Naming Convention

- Use numbers for ordering: `01-`, `02-`, etc.
- Use descriptive names: `answer-example`, `ask-decision-details`
- Use lowercase with hyphens
- Format: `{number}-{description}.png`

## Usage

These screenshots are used for:
1. **Fallback demo**: If live demo fails, present these screenshots
2. **Documentation**: Embedded in docs and README
3. **Presentations**: For slides and talks
4. **GitHub README**: Visual examples in the project README

## Taking All Screenshots (Script)

```bash
# 1. Start demo
make demo-mock

# 2. Open browser to http://localhost:7860

# 3. Take screenshots in order:
#    - Interface overview (before any input)
#    - Type "What is the capital of France?" → Ask → Screenshot
#    - Expand Decision Details → Screenshot
#    - Clear/refresh
#    - Type "What is the capital of Georgia?" → Ask → Screenshot
#    - Expand Decision Details → Screenshot
#    - Click an option → Screenshot
#    - Clear/refresh
#    - Type unknown question → Ask → Screenshot
#    - Expand Decision Details → Screenshot
#    - Scroll to Examples → Screenshot

# 4. For API screenshot:
#    make api-dev
#    # In another terminal:
#    curl -X POST http://localhost:8000/v1/analyze ...
#    # Screenshot the terminal
```

## Verification Checklist

- [ ] All 9-10 screenshots captured
- [ ] Screenshots are clear and high resolution
- [ ] Text is readable
- [ ] No personal information visible
- [ ] Consistent browser window size
- [ ] Named according to convention
- [ ] Placed in `docs/assets/screenshots/` directory

---

**Note:** Actual screenshots should be taken by the user running `make demo-mock` and following this guide. The screenshots are not included in git due to size, but should be generated for presentations and documentation.
