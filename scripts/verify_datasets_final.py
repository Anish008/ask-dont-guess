"""Final dataset verification script.

Loads all datasets and records findings to VERIFIED_FACTS.md
"""
# ruff: noqa: E402, T201
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

results_file = Path("docs/VERIFIED_FACTS.md")

print("\n" + "=" * 80)
print("STEP 3: Dataset Verification - Final")
print("=" * 80 + "\n")

findings = []
findings.append(f"## Step 3: Dataset Verification ({datetime.now().strftime('%Y-%m-%d')})")
findings.append("")

# ==============================================================================
# 1. AmbigQA
# ==============================================================================

print("-" * 80)
print("1. AmbigQA from HuggingFace")
print("-" * 80)

try:
    from datasets import load_dataset

    ambigqa = load_dataset("sewon/ambig_qa", "full")
    print("[OK] Loaded")

    findings.append("### AmbigQA")
    findings.append("- **Source**: HuggingFace `sewon/ambig_qa` (config: `full`)")
    findings.append(f"- **Splits**: {', '.join(ambigqa.keys())}")

    for split_name, split_data in ambigqa.items():
        print(f"  {split_name}: {len(split_data)} rows")
        findings.append(f"- **{split_name}**: {len(split_data)} examples")

    findings.append(f"- **Columns**: {', '.join(ambigqa['train'].column_names)}")
    findings.append("")

    # Analyze annotation types
    type_counts = {"multipleQAs": 0, "singleAnswer": 0}
    for split in ["train", "validation"]:
        for item in ambigqa[split]:
            ann_type = item["annotations"]["type"][0]
            if ann_type in type_counts:
                type_counts[ann_type] += 1

    findings.append("**Annotation types:**")
    for ann_type, count in type_counts.items():
        pct = count / (len(ambigqa["train"]) + len(ambigqa["validation"])) * 100
        findings.append(f"- `{ann_type}`: {count} ({pct:.1f}%)")
        print(f"  {ann_type}: {count} ({pct:.1f}%)")

    findings.append("")
    findings.append("**Schema:**")
    example = ambigqa["train"][0]
    findings.append("```python")
    findings.append("# multipleQAs example:")
    findings.append(f"id: {example['id']}")
    findings.append(f"question: \"{example['question']}\"")
    findings.append(f"annotations['type']: {example['annotations']['type']}")
    findings.append("annotations['qaPairs']: [")
    if example["annotations"]["qaPairs"]:
        qa = example["annotations"]["qaPairs"][0]
        findings.append("  {'question': [...], 'answer': [...]}")
    findings.append("]")
    findings.append("```")
    findings.append("")

    # Find singleAnswer example
    for item in ambigqa["validation"]:
        if item["annotations"]["type"][0] == "singleAnswer":
            findings.append("```python")
            findings.append("# singleAnswer example:")
            findings.append(f"question: \"{item['question']}\"")
            findings.append(f"annotations['answer']: {item['annotations']['answer']}")
            findings.append("```")
            findings.append("")
            break

except Exception as e:
    print(f"[FAIL] {e}")
    findings.append("### AmbigQA")
    findings.append(f"- **Status**: FAILED - {e}")
    findings.append("")

# ==============================================================================
# 2. NQ-open
# ==============================================================================

print("\n" + "-" * 80)
print("2. NQ-open from HuggingFace")
print("-" * 80)

try:
    nq_open = load_dataset("google-research-datasets/nq_open")
    print("[OK] Loaded")

    findings.append("### NQ-open")
    findings.append("- **Source**: HuggingFace `google-research-datasets/nq_open`")
    findings.append(f"- **Splits**: {', '.join(nq_open.keys())}")

    for split_name, split_data in nq_open.items():
        print(f"  {split_name}: {len(split_data)} rows")
        findings.append(f"- **{split_name}**: {len(split_data)} examples")

    findings.append(f"- **Columns**: {', '.join(nq_open['train'].column_names)}")
    findings.append("")

    findings.append("**Schema:**")
    example = nq_open["train"][0]
    findings.append("```python")
    findings.append(f"question: \"{example['question']}\"")
    findings.append(f"answer: {example['answer']}")  # List of answer strings
    findings.append("```")
    findings.append("")

except Exception as e:
    print(f"[FAIL] {e}")
    findings.append("### NQ-open")
    findings.append(f"- **Status**: FAILED - {e}")
    findings.append("")

# ==============================================================================
# 3. CLAMBER
# ==============================================================================

print("\n" + "-" * 80)
print("3. CLAMBER from GitHub")
print("-" * 80)

clamber_file = Path("data/external/CLAMBER/clamber_benchmark.jsonl")

if clamber_file.exists():
    try:
        samples = []
        with clamber_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Handle double-encoded JSON
                    if isinstance(data, str):
                        data = json.loads(data)
                    samples.append(data)

        print(f"[OK] Loaded {len(samples)} samples")

        findings.append("### CLAMBER")
        findings.append("- **Source**: GitHub `zt991211/CLAMBER`")
        findings.append("- **File**: `clamber_benchmark.jsonl`")
        findings.append(f"- **Total examples**: {len(samples)}")
        findings.append("")

        # Category breakdown
        category_counts = {}
        req_clarification = 0
        for sample in samples:
            if isinstance(sample, dict):
                cat = sample.get("category", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                if sample.get("require_clarification", 0) == 1:
                    req_clarification += 1

        findings.append("**Categories:**")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            pct = count / len(samples) * 100
            findings.append(f"- `{cat}`: {count} ({pct:.1f}%)")
            print(f"  {cat}: {count} ({pct:.1f}%)")

        findings.append("")

        # Clarification requirement
        findings.append(
            f"**Requires clarification**: {req_clarification} ({req_clarification/len(samples)*100:.1f}%)"
        )
        print(
            f"  Requires clarification: {req_clarification} ({req_clarification/len(samples)*100:.1f}%)"
        )
        findings.append("")

        findings.append("**Schema:**")
        example = samples[0]
        findings.append("```python")
        findings.append(f"question: \"{example['question']}\"")
        findings.append(f"require_clarification: {example['require_clarification']}")
        findings.append(f"category: \"{example['category']}\"")
        findings.append(f"subclass: \"{example['subclass']}\"")
        findings.append(f"clarifying_question: \"{example['clarifying_question'][:80]}...\"")
        findings.append("```")
        findings.append("")

    except Exception as e:
        print(f"[FAIL] {e}")
        findings.append("### CLAMBER")
        findings.append(f"- **Status**: FAILED - {e}")
        findings.append("")
else:
    print("[SKIP] File not found")
    findings.append("### CLAMBER")
    findings.append(f"- **Status**: SKIPPED - File not found at {clamber_file}")
    findings.append("")

# ==============================================================================
# 4. Deduplication
# ==============================================================================

print("\n" + "-" * 80)
print("4. Deduplication Analysis")
print("-" * 80)

try:

    def normalize(text):
        return text.lower().strip().replace("?", "").replace(".", "").replace(",", "")

    # AmbigQA questions
    ambigqa_questions = set()
    for split in ["train", "validation"]:
        for item in ambigqa[split]:
            ambigqa_questions.add(normalize(item["question"]))

    findings.append("### AmbigQA ↔ NQ-open Deduplication")
    findings.append(f"- **Unique AmbigQA questions**: {len(ambigqa_questions)}")

    # NQ-open overlaps
    for split_name in ["train", "validation"]:
        nq_questions = [normalize(item["question"]) for item in nq_open[split_name]]
        overlaps = [q for q in nq_questions if q in ambigqa_questions]

        findings.append(
            f"- **NQ-open {split_name}**: {len(nq_questions)} total, {len(overlaps)} overlap ({len(overlaps)/len(nq_questions)*100:.1f}%)"
        )
        print(
            f"  NQ-open {split_name}: {len(overlaps)}/{len(nq_questions)} overlap ({len(overlaps)/len(nq_questions)*100:.1f}%)"
        )

    findings.append("")
    findings.append(
        "**Deduplication strategy:** Exclude NQ-open questions that appear in AmbigQA (normalized match)"
    )
    findings.append("")

except Exception as e:
    print(f"[FAIL] {e}")
    findings.append("### Deduplication")
    findings.append(f"- **Status**: FAILED - {e}")
    findings.append("")

# ==============================================================================
# Write to file
# ==============================================================================

print("\n" + "-" * 80)
print("Writing findings to", results_file)
print("-" * 80)

# Read existing content
if results_file.exists():
    with results_file.open(encoding="utf-8") as f:
        existing = f.read()
else:
    existing = "# Verified Facts\n\nLast updated: {date}\n\n"

# Check if Step 3 already exists
if "## Step 3:" in existing:
    # Replace Step 3 section
    parts = existing.split("## Step 3:")
    before = parts[0]

    # Find next section
    if "## Step 4:" in parts[1] or "## Datasets" in parts[1]:
        # Keep content after Step 3
        after_parts = parts[1].split("\n## ", 1)
        after = "\n## " + after_parts[1] if len(after_parts) > 1 else ""
    else:
        after = ""

    new_content = before + "\n".join(findings) + "\n" + after
else:
    # Append to end
    new_content = existing.rstrip() + "\n\n" + "\n".join(findings) + "\n"

with results_file.open("w", encoding="utf-8") as f:
    f.write(new_content)

print(f"[OK] Written to {results_file}")

print("\n" + "=" * 80)
print("STEP 3 COMPLETE")
print("=" * 80 + "\n")
