"""Verify datasets for ADG experiments.

Step 3: Load AmbigQA, NQ-open, CLAMBER and verify schemas.
"""
# ruff: noqa: E402, T201
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("\n" + "=" * 80)
print("STEP 3: Dataset Verification")
print("=" * 80 + "\n")

# ==============================================================================
# 1. AmbigQA from HuggingFace
# ==============================================================================

print("-" * 80)
print("1. Loading AmbigQA from HuggingFace")
print("-" * 80 + "\n")

try:
    from datasets import load_dataset

    print("Loading sewon/ambig_qa...")
    ambigqa = load_dataset("sewon/ambig_qa", "full")

    print("[OK] AmbigQA loaded")
    print(f"\nSplits: {list(ambigqa.keys())}")

    for split_name, split_data in ambigqa.items():
        print(f"\n{split_name}:")
        print(f"  Rows: {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")

        # Show first example
        if len(split_data) > 0:
            example = split_data[0]
            print(f"\n  First example ({split_name}):")
            print(f"    id: {example.get('id', 'N/A')}")
            print(f"    question: {example.get('question', 'N/A')[:100]}...")

            # Check annotations structure
            if "annotations" in example:
                ann = example["annotations"]
                print(f"    annotations type: {type(ann)}")
                if isinstance(ann, list) and len(ann) > 0:
                    first_ann = ann[0]
                    print(
                        f"    annotations[0] keys: {list(first_ann.keys()) if isinstance(first_ann, dict) else 'N/A'}"
                    )
                    if isinstance(first_ann, dict):
                        print(f"    annotations[0]['type']: {first_ann.get('type', 'N/A')}")
                        if "qaPairs" in first_ann:
                            print(
                                f"    annotations[0]['qaPairs']: {len(first_ann['qaPairs'])} pairs"
                            )
                            if len(first_ann["qaPairs"]) > 0:
                                qa_pair = first_ann["qaPairs"][0]
                                print(
                                    f"      qaPairs[0]['question']: {qa_pair.get('question', 'N/A')[:80]}..."
                                )
                                print(f"      qaPairs[0]['answer']: {qa_pair.get('answer', 'N/A')}")

except Exception as e:
    print(f"[FAIL] AmbigQA loading failed: {e}")
    import traceback

    traceback.print_exc()

# ==============================================================================
# 2. NQ-open from HuggingFace
# ==============================================================================

print("\n" + "-" * 80)
print("2. Loading NQ-open from HuggingFace")
print("-" * 80 + "\n")

try:
    print("Loading nq_open...")
    nq_open = load_dataset("nq_open")

    print("[OK] NQ-open loaded")
    print(f"\nSplits: {list(nq_open.keys())}")

    for split_name, split_data in nq_open.items():
        print(f"\n{split_name}:")
        print(f"  Rows: {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")

        # Show first example
        if len(split_data) > 0:
            example = split_data[0]
            print(f"\n  First example ({split_name}):")
            for key, value in example.items():
                if isinstance(value, str):
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")

except Exception as e:
    print(f"[FAIL] NQ-open loading failed: {e}")
    import traceback

    traceback.print_exc()

# ==============================================================================
# 3. CLAMBER from GitHub
# ==============================================================================

print("\n" + "-" * 80)
print("3. Loading CLAMBER from GitHub")
print("-" * 80 + "\n")

clamber_dir = Path("data/external/CLAMBER")
clamber_repo = "https://github.com/zt991211/CLAMBER.git"

if not clamber_dir.exists():
    print(f"Cloning {clamber_repo}...")
    import subprocess

    clamber_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", clamber_repo, str(clamber_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print(f"[OK] CLAMBER cloned to {clamber_dir}")
    else:
        print(f"[FAIL] Clone failed: {result.stderr}")
else:
    print(f"[OK] CLAMBER already exists at {clamber_dir}")

# Find data files
print("\nSearching for data files in CLAMBER...")
data_files = list(clamber_dir.rglob("*.json")) + list(clamber_dir.rglob("*.jsonl"))
print(f"Found {len(data_files)} JSON/JSONL files:")

for file_path in sorted(data_files)[:10]:  # Show first 10
    rel_path = file_path.relative_to(clamber_dir)
    size_kb = file_path.stat().st_size / 1024
    print(f"  {rel_path} ({size_kb:.1f} KB)")

if len(data_files) > 10:
    print(f"  ... and {len(data_files) - 10} more")

# Try to load a sample file
if data_files:
    sample_file = None
    for f in data_files:
        if "test" in f.name.lower() or "dev" in f.name.lower():
            sample_file = f
            break

    if not sample_file:
        sample_file = data_files[0]

    print(f"\nLoading sample file: {sample_file.name}")
    try:
        import json

        with sample_file.open(encoding="utf-8") as f:
            # Try JSONL first
            first_line = f.readline()
            if first_line.strip():
                try:
                    sample = json.loads(first_line)
                    print("[OK] JSONL format detected")
                    print(f"Sample keys: {list(sample.keys())}")
                    print("\nSample data:")
                    for key, value in list(sample.items())[:5]:
                        if isinstance(value, str):
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                except json.JSONDecodeError:
                    # Try as regular JSON
                    f.seek(0)
                    data = json.load(f)
                    print("[OK] JSON format detected")
                    print(f"Type: {type(data)}")
                    if isinstance(data, list) and len(data) > 0:
                        print(f"Length: {len(data)}")
                        print(f"Sample[0] keys: {list(data[0].keys())}")
                        print("\nSample data:")
                        for key, value in list(data[0].items())[:5]:
                            if isinstance(value, str):
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                    elif isinstance(data, dict):
                        print(f"Top-level keys: {list(data.keys())}")
    except Exception as e:
        print(f"[FAIL] Could not load sample: {e}")

# ==============================================================================
# 4. Deduplication check
# ==============================================================================

print("\n" + "-" * 80)
print("4. AmbigQA <-> NQ-open deduplication")
print("-" * 80 + "\n")

try:
    # Normalize text for comparison
    def normalize(text):
        return text.lower().strip().replace("?", "").replace(".", "").replace(",", "")

    # Get AmbigQA questions
    ambigqa_questions = set()
    if "train" in ambigqa:
        for item in ambigqa["train"]:
            ambigqa_questions.add(normalize(item["question"]))
    if "validation" in ambigqa:
        for item in ambigqa["validation"]:
            ambigqa_questions.add(normalize(item["question"]))

    print(f"Total unique AmbigQA questions: {len(ambigqa_questions)}")

    # Check NQ-open overlap
    if "train" in nq_open:
        nq_train_questions = [normalize(item["question"]) for item in nq_open["train"]]
        overlaps = [q for q in nq_train_questions if q in ambigqa_questions]
        print(f"NQ-open train questions: {len(nq_train_questions)}")
        print(
            f"Overlap with AmbigQA: {len(overlaps)} ({len(overlaps)/len(nq_train_questions)*100:.1f}%)"
        )

        if len(overlaps) > 0:
            print("\nFirst 3 overlapping questions:")
            for i, q in enumerate(overlaps[:3], 1):
                print(f"  {i}. {q[:80]}...")

except Exception as e:
    print(f"[FAIL] Deduplication check failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("STEP 3 COMPLETE: Dataset verification done")
print("=" * 80 + "\n")
