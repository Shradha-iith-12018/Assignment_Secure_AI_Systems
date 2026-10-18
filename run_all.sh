#!/usr/bin/env bash
# Canonical end-to-end experiment pipeline.
set -euo pipefail
cd "$(dirname "$0")"

# Examples:
#   SSE_DEVICE=cpu  ./run_all.sh
#   SSE_DEVICE=cuda ./run_all.sh
#   SSE_DEVICE=auto ./run_all.sh
export SSE_DEVICE="${SSE_DEVICE:-auto}"
echo "SSE_DEVICE=$SSE_DEVICE"

python scripts/record_environment.py
python -m src.train
python -m attacks.evasion
python -m attacks.poisoning
python -m defenses.adversarial_training
python -m defenses.feature_squeezing
python -m defenses.poison_filter
python -m src.summary
