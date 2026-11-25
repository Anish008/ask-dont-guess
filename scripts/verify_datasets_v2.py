"""Verify datasets for ADG experiments - Version 2.

Step 3: Load AmbigQA, NQ-open, CLAMBER and verify schemas.
"""
# ruff: noqa: E402, T201
import json
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

ambigqa = None
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

                # Handle both dict and list
                if isinstance(ann, dict):
                    print(f"    annotations keys: {list(ann.keys())}")
                    print(f"    type: {ann.get('type', 'N/A')}")

                    if "qaPairs" in ann:
                        qa_pairs = ann["qaPairs"]
                        print(f"    qaPairs: {len(qa_pairs)} pairs")
                        if len(qa_pairs) > 0:
                            qa_pair = qa_pairs[0]
                            print(f"      qaPairs[0]: {qa_pair}")

                    if "answer" in ann:
                        answer = ann["answer"]
                        print(f"    answer: {answer}")

                elif isinstance(ann, list) and len(ann) > 0:
                    first_ann = ann[0]
                    print(f"    annotations[0] type: {type(first_ann)}")
                    if isinstance(first_ann, dict):
                        print(f"    annotations[0] keys: {list(first_ann.keys())}")

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

nq_open = None
# Try different possible dataset IDs
nq_dataset_ids = [
    "google-research-datasets/nq_open",
    "nq_open",
    "natural_questions",
]

for dataset_id in nq_dataset_ids:
    try:
        print(f"Trying {dataset_id}...")
        nq_open = load_dataset(dataset_id)
        print(f"[OK] NQ-open loaded from {dataset_id}")
        break
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}...")
        continue

if nq_open:
    print(f"\nSplits: {list(nq_open.keys())}")

    for split_name in list(nq_open.keys())[:2]:  # Show first 2 splits
        split_data = nq_open[split_name]
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
                elif isinstance(value, list):
                    print(f"    {key}: list of {len(value)} items")
                    if len(value) > 0:
                        print(f"      [0]: {value[0]}")
                else:
                    print(f"    {key}: {value}")
else:
    print("[FAIL] Could not load NQ-open from any known dataset ID")
    print("Will search HuggingFace Hub for correct dataset...")
    # Try to find it via API
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        datasets_info = api.list_datasets(search="nq open", limit=10)
        print("\nFound datasets matching 'nq open':")
        for ds in datasets_info:
            print(f"  - {ds.id}")
    except Exception as e:
        print(f"  Search failed: {e}")

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

# Try to load the benchmark file
if data_files:
    benchmark_file = None
    for f in data_files:
        if "benchmark" in f.name.lower():
            benchmark_file = f
            break

    if benchmark_file:
        print(f"\nLoading {benchmark_file.name}...")
        try:
            samples = []
            with benchmark_file.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            print(f"  Line {line_num}: JSON decode error: {e}")
                            if line_num <= 2:
                                print(f"    Content: {line[:100]}...")

            print(f"[OK] Loaded {len(samples)} samples from {benchmark_file.name}")

            if samples:
                print(f"\nSample[0] keys: {list(samples[0].keys())}")
                print("\nSample data:")
                sample = samples[0]
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"  {key}: {value[:100]}...")
                    elif isinstance(value, list):
                        print(f"  {key}: list of {len(value)} items")
                        if len(value) > 0 and isinstance(value[0], str):
                            print(f"    [0]: {value[0][:80]}...")
                    else:
                        print(f"  {key}: {value}")

                # Check categories
                categories = {}
                for sample in samples:
                    cat = sample.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1

                print("\nCategories:")
                for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                    print(f"  {cat}: {count}")

        except Exception as e:
            print(f"[FAIL] Could not load benchmark: {e}")
            import traceback

            traceback.print_exc()

# ==============================================================================
# 4. Deduplication check
# ==============================================================================

print("\n" + "-" * 80)
print("4. AmbigQA <-> NQ-open deduplication")
print("-" * 80 + "\n")

if ambigqa and nq_open:
    try:
        # Normalize text for comparison
        def normalize(text):
            return text.lower().strip().replace("?", "").replace(".", "").replace(",", "")

        # Get AmbigQA questions
        ambigqa_questions = set()
        for split in ["train", "validation"]:
            if split in ambigqa:
                for item in ambigqa[split]:
                    ambigqa_questions.add(normalize(item["question"]))

        print(f"Total unique AmbigQA questions: {len(ambigqa_questions)}")

        # Check NQ-open overlap
        for split_name in ["train", "validation", "test"]:
            if split_name in nq_open:
                nq_questions = [normalize(item["question"]) for item in nq_open[split_name]]
                overlaps = [q for q in nq_questions if q in ambigqa_questions]
                print(f"\nNQ-open {split_name}:")
                print(f"  Total questions: {len(nq_questions)}")
                print(
                    f"  Overlap with AmbigQA: {len(overlaps)} ({len(overlaps)/len(nq_questions)*100:.1f}%)"
                )

                if len(overlaps) > 0:
                    print("  First 3 overlapping:")
                    for i, q in enumerate(overlaps[:3], 1):
                        print(f"    {i}. {q[:70]}...")

    except Exception as e:
        print(f"[FAIL] Deduplication check failed: {e}")
        import traceback

        traceback.print_exc()
else:
    print("[SKIP] Cannot perform deduplication check - missing datasets")
    if not ambigqa:
        print("  - AmbigQA not loaded")
    if not nq_open:
        print("  - NQ-open not loaded")

print("\n" + "=" * 80)
print("STEP 3 COMPLETE: Dataset verification done")
print("=" * 80 + "\n")
