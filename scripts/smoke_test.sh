#!/usr/bin/env bash
# Smoke test script for ADG pipeline
# Runs curated examples with real or mock models

set -e

# Check if --mock flag is passed
MOCK_FLAG=""
if [[ "$1" == "--mock" ]]; then
    MOCK_FLAG="--mock"
    echo "Running smoke test with MOCK backends..."
else
    echo "Running smoke test with REAL models (will download if needed)..."
    echo "Use --mock to run with mock backends instead."
fi

# Run smoke test
python scripts/smoke_test.py $MOCK_FLAG

echo ""
echo "Smoke test complete!"
