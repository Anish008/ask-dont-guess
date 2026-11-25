"""Simple pipeline test with baseline methods.

Tests the full pipeline with a few examples before running full experiments.
"""
# ruff: noqa: E402, T201
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adg.config.loader import load_config
from adg.core.types import Query
from adg.llm.hf_backend import HFLLMBackend
from adg.methods.always_answer import AlwaysAnswer
from adg.methods.ask_if_unclear import AskIfUnclear
from adg.nli.hf_nli import HFNLIScorer

print("\n" + "=" * 80)
print("Simple Pipeline Test")
print("=" * 80 + "\n")

# Load config
config = load_config(Path("configs/base.yaml"))
print(f"Config loaded: {config.llm.model_id}\n")

# Load backends
print("Loading backends...")
llm = HFLLMBackend(config)
print("[OK] LLM loaded")

nli = HFNLIScorer(config)
print("[OK] NLI loaded\n")

# Create test queries
queries = [
    Query(query_id="clear1", text="What is the capital of France?"),
    Query(query_id="ambiguous1", text="What is the capital of Georgia?"),
    Query(query_id="unknown1", text="What is the most popular ice cream flavor in Antarctica?"),
]

# Test methods
methods = [
    ("always_answer", AlwaysAnswer(llm)),
    ("ask_if_unclear", AskIfUnclear(llm)),
]

results = []

for method_name, method in methods:
    print("-" * 80)
    print(f"Testing: {method_name}")
    print("-" * 80)

    for query in queries:
        print(f"\nQuery: {query.text}")

        try:
            output = method.decide(query)
            print(f"  Action: {output.action}")
            print(f"  Message: {output.message[:80]}...")
            print(f"  Score: {output.score}")

            results.append(
                {
                    "method": method_name,
                    "query_id": query.query_id,
                    "query": query.text,
                    "action": output.action,
                    "message": output.message[:100],
                    "score": output.score,
                }
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

            results.append(
                {
                    "method": method_name,
                    "query_id": query.query_id,
                    "query": query.text,
                    "action": "ERROR",
                    "error": str(e),
                }
            )

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80 + "\n")

print(f"Tested {len(methods)} methods on {len(queries)} queries\n")

for method_name, _ in methods:
    method_results = [r for r in results if r["method"] == method_name]
    actions = {}
    for r in method_results:
        action = r.get("action", "ERROR")
        actions[action] = actions.get(action, 0) + 1

    print(f"{method_name}:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}")

# Cleanup
print("\n" + "-" * 80)
print("Cleanup...")
llm.close()
nli.close()
print("[OK] Done\n")
