"""Test baseline methods implementations.

Quick verification that all baselines work correctly.
"""
# ruff: noqa: E402
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adg.config.loader import load_config
from adg.core.types import Query
from adg.llm.hf_backend import HFLLMBackend
from adg.methods.ask_if_unclear import AskIfUnclear
from adg.methods.stated_confidence import StatedConfidence
from adg.nli.hf_nli import HFNLIScorer

print("\n" + "=" * 80)
print("Testing Baseline Methods")
print("=" * 80 + "\n")

# Load config
config = load_config(Path("configs/base.yaml"))
print(f"Config loaded: {config.llm.model_id}")

# Load backends
print("\nLoading backends...")
llm = HFLLMBackend(config)
print("[OK] LLM loaded")

nli = HFNLIScorer(config)
print("[OK] NLI loaded")

# Test queries
queries = [
    Query(query_id="test1", text="What is the capital of France?"),
    Query(query_id="test2", text="What is the capital of Georgia?"),
]

print("\n" + "-" * 80)
print("Testing AskIfUnclear")
print("-" * 80)
method = AskIfUnclear(llm)
for query in queries:
    print(f"\nQuery: {query.text}")
    result = method.decide(query)
    print(f"Action: {result.action}")
    print(f"Message: {result.message[:100]}")
    print(f"Parse failed: {result.trace.get('parse_failed', False)}")

print("\n" + "-" * 80)
print("Testing StatedConfidence")
print("-" * 80)
method = StatedConfidence(llm)
for query in queries[:1]:  # Just test first one to save time
    print(f"\nQuery: {query.text}")
    result = method.decide(query)
    print(f"Action: {result.action}")
    print(f"Confidence: {result.score}")
    print(f"Parse failed: {result.trace.get('parse_failed', False)}")

# Cleanup
print("\n" + "-" * 80)
print("Cleanup...")
llm.close()
nli.close()
print("[OK] Cleanup complete")

print("\n" + "=" * 80)
print("Baseline tests complete!")
print("=" * 80 + "\n")
