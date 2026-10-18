#!/usr/bin/env bash
# Runs the whole pipeline in order. Each step only needs the outputs of the steps before it.
set -euo pipefail
cd "$(dirname "$0")"

python -m src.train                       # Tasks 1-2: baseline model + clean metrics
python -m attacks.evasion                 # Task 5-6: FGSM/PGD vs baseline (ML01)
python -m attacks.poisoning               # Task 5-6: backdoor poisoning (ML02)
python -m defenses.adversarial_training   # Task 7: adversarial training
python -m defenses.feature_squeezing      # Task 7: extra defense vs evasion
python -m defenses.poison_filter          # Task 7: extra defense vs poisoning
python -m src.summary                     # comparison tables -> results/summary.md
