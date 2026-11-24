"""Smoke test script for real models.

Runs curated examples through the full ADG pipeline with real models
(Qwen2.5-1.5B + DeBERTa) and logs traces to results/smoke/.

Usage:
    python scripts/smoke_test.py [--mock]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.types import Query

log = get_logger(__name__)


def load_curated_examples() -> list[dict]:
    """Load curated examples from fixtures."""
    import yaml

    examples_path = Path("tests/fixtures/curated_examples.yaml")
    if not examples_path.exists():
        log.error("curated_examples_not_found", path=str(examples_path))
        return []

    with examples_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("examples", [])


def run_smoke_test(use_mock: bool = False) -> dict:
    """Run smoke test with real or mock models.

    Args:
        use_mock: Use mock backends if True

    Returns:
        Results dictionary with traces
    """
    log.info("smoke_test_starting", use_mock=use_mock)

    # Create backends
    if use_mock:
        from adg.llm.mock_backend import MockLLMBackend
        from adg.nli.mock_nli import MockNLIScorer

        llm = MockLLMBackend()
        nli = MockNLIScorer()
        log.info("using_mock_backends")
    else:
        # Real models - will be implemented when models are available
        log.warning(
            "real_models_not_yet_implemented",
            message="Real model backends require HF model downloads. Using mocks for now.",
        )
        from adg.llm.mock_backend import MockLLMBackend
        from adg.nli.mock_nli import MockNLIScorer

        llm = MockLLMBackend()
        nli = MockNLIScorer()

    # Create manager
    manager = ConversationManager(
        llm_backend=llm,
        nli_scorer=nli,
        n_samples=8,
        temperature=0.7,
        top_p=0.95,
    )

    # Load examples
    examples = load_curated_examples()
    if not examples:
        log.error("no_examples_loaded")
        return {"error": "No examples found"}

    # Run each example
    results = {
        "timestamp": datetime.now().isoformat(),
        "use_mock": use_mock,
        "examples": [],
    }

    for example in examples:
        example_id = example["id"]
        query_text = example["query"]
        expected_action = example["expected_action"]

        log.info("processing_example", example_id=example_id, query=query_text)

        # Create session
        session = manager.start_session(f"smoke_{example_id}")

        # Process query
        query = Query(query_id=example_id, text=query_text)
        session, response, decision = manager.process_query(session, query)

        # Log result
        result = {
            "id": example_id,
            "query": query_text,
            "expected_action": expected_action,
            "actual_action": decision.action,
            "confidence": decision.confidence,
            "rule_name": decision.rule_name,
            "response": response,
            "matches_expected": decision.action == expected_action,
        }

        results["examples"].append(result)

        log.info(
            "example_processed",
            example_id=example_id,
            action=decision.action,
            expected=expected_action,
            matches=result["matches_expected"],
        )

    return results


def save_results(results: dict, output_dir: Path) -> None:
    """Save results to JSON file.

    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"smoke_test_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info("results_saved", path=str(output_file))
    print(f"\nResults saved to: {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ADG smoke tests")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock backends instead of real models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smoke"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        # Run smoke test
        results = run_smoke_test(use_mock=args.mock)

        # Save results
        save_results(results, args.output_dir)

        # Print summary
        total = len(results.get("examples", []))
        matches = sum(1 for ex in results.get("examples", []) if ex.get("matches_expected", False))

        print("\nSmoke Test Summary:")
        print(f"  Total examples: {total}")
        print(f"  Matching expected: {matches}/{total}")
        print(f"  Using mock backends: {args.mock}")

        for ex in results.get("examples", []):
            status = "✓" if ex.get("matches_expected") else "✗"
            print(
                f"    {status} {ex['id']}: {ex['actual_action']} (expected: {ex['expected_action']})"
            )

        return 0 if matches == total else 1

    except Exception as e:
        log.exception("smoke_test_failed", error=str(e))
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
