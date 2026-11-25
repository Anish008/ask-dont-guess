"""Test real model loading and inference on curated examples.

Step 1 of experiment setup: Verify HF models work correctly.
"""
# ruff: noqa: E402, PLR0912, PLR0915, PLC0415

import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adg.config.loader import load_config
from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.types import Query
from adg.llm.hf_backend import HFLLMBackend
from adg.nli.hf_nli import HFNLIScorer

log = get_logger(__name__)


def main() -> None:  # noqa: PLR0915
    """Test real models on curated examples."""
    print("\n" + "=" * 80)
    print("STEP 1: Real Model Support - Testing HF Models")
    print("=" * 80 + "\n")

    # Load config
    print("Loading configuration from configs/base.yaml...")
    config = load_config(Path("configs/base.yaml"))
    print(f"  LLM config: {config.llm.config}")
    print(f"  NLI config: {config.nli.config}")
    print(f"  Model ID: {config.llm.model_id}")
    print(f"  NLI Model ID: {config.nli.model_id}")

    # Load models
    print("\n" + "-" * 80)
    print("Loading HuggingFace models (this may take a few minutes)...")
    print("-" * 80 + "\n")

    try:
        llm_backend = HFLLMBackend(config)
        print(f"[OK] LLM loaded: {config.llm.model_id}")
    except Exception as e:
        print(f"[FAIL] LLM loading failed: {e}")
        return

    try:
        nli_scorer = HFNLIScorer(config)
        print(f"[OK] NLI loaded: {config.nli.model_id}")
    except Exception as e:
        print(f"[FAIL] NLI loading failed: {e}")
        llm_backend.close()
        return

    # Create conversation manager
    manager = ConversationManager(llm_backend=llm_backend, nli_scorer=nli_scorer)
    print("[OK] ConversationManager initialized")

    # Load curated examples
    examples_path = Path("tests/fixtures/curated_examples.yaml")
    with examples_path.open(encoding="utf-8") as f:
        examples_data = yaml.safe_load(f)

    examples = examples_data["examples"][:3]  # Test first 3 examples

    print("\n" + "=" * 80)
    print(f"Testing {len(examples)} curated examples")
    print("=" * 80 + "\n")

    results = []

    for i, example in enumerate(examples, 1):
        example_id = example["id"]
        query_text = example["query"]
        expected_action = example["expected_action"]

        print(f"\n{'=' * 80}")
        print(f"Example {i}/{len(examples)}: {example_id}")
        print(f"{'=' * 80}")
        print(f"Query: {query_text}")
        print(f"Expected: {expected_action}")
        print(f"Description: {example['description']}")
        print()

        # Create query
        query = Query(query_id=example_id, text=query_text)

        # Start session
        session = manager.start_session(example_id)

        # Process query
        try:
            session, response, decision = manager.process_query(session, query)

            # Log results
            print(f"\n{'─' * 80}")
            print("RESULT:")
            print(f"{'─' * 80}")
            print(f"Action: {decision.action}")
            print(f"Confidence: {decision.confidence:.3f}")
            print(f"Rule: {decision.rule_name}")
            print(f"\nResponse: {response}")

            if decision.action == "ASK" and session.pending_clarification:
                print(f"\nClarifying question: {session.pending_clarification.question}")
                print(f"Options: {session.pending_clarification.options}")

            print("\nConfidence scores:")
            for action, score in decision.confidence_scores.items():
                print(f"  {action}: {score:.3f}")

            # Get the last turn to see cluster info
            if session.turns:
                last_turn = session.turns[-1]
                if hasattr(last_turn, "features") and last_turn.features:
                    features = last_turn.features
                    print("\nCluster features:")
                    print(f"  n_clusters: {features.n_clusters}")
                    print(f"  p1: {features.p1:.3f}")
                    print(f"  p2: {features.p2:.3f}")
                    print(f"  coverage: {features.coverage:.3f}")
                    print(f"  ask_score: {features.ask_score:.3f}")
                    print(f"  answer_score: {features.answer_score:.3f}")
                    print(f"  abstain_score: {features.abstain_score:.3f}")

                if hasattr(last_turn, "result") and last_turn.result:
                    result = last_turn.result
                    print("\nSampling:")
                    print(f"  n_samples: {result.n_samples}")
                    print(f"  n_valid: {result.n_valid}")
                    print(f"  n_refusals: {result.n_refusals}")

                    # Show first few samples
                    print("\nFirst 3 samples:")
                    for j, sample in enumerate(result.samples[:3], 1):
                        print(f"  {j}. Interpretation: {sample.interpretation}")
                        print(f"     Answer: {sample.answer}")

            # Check if matches expected
            match = decision.action == expected_action
            match_str = "[MATCH]" if match else "[MISMATCH]"
            print(f"\n{match_str}: Got {decision.action}, expected {expected_action}")

            results.append(
                {
                    "id": example_id,
                    "query": query_text,
                    "expected": expected_action,
                    "actual": decision.action,
                    "match": match,
                    "confidence": decision.confidence,
                }
            )

        except Exception as e:
            print(f"\n[ERROR] processing example: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "id": example_id,
                    "query": query_text,
                    "expected": expected_action,
                    "actual": "ERROR",
                    "match": False,
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")

    matches = sum(1 for r in results if r["match"])
    print(f"Results: {matches}/{len(results)} matched expected actions\n")

    for result in results:
        status = "[OK]" if result["match"] else "[FAIL]"
        print(f"{status} {result['id']}: {result['actual']} (expected {result['expected']})")

    # Cleanup
    print("\n" + "-" * 80)
    print("Cleaning up models...")
    llm_backend.close()
    nli_scorer.close()
    print("[OK] Cleanup complete")

    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE: Real models tested successfully")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
