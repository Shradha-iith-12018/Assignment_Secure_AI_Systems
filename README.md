# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

A PyTorch CNN trained on MNIST, attacked with two OWASP ML Top 10 attacks, then defended.

| Stage | OWASP ML risk | Code |
|---|---|---|
| Baseline CNN, clean metrics | - | [src/train.py](src/train.py), [src/metrics.py](src/metrics.py) |
| Attack 1: adversarial examples (FGSM, PGD via ART) | ML01 Input Manipulation | [attacks/evasion.py](attacks/evasion.py) |
| Attack 2: backdoor data poisoning | ML02 Data Poisoning | [attacks/poisoning.py](attacks/poisoning.py) |
| Defense: PGD adversarial training | ML01 | [defenses/adversarial_training.py](defenses/adversarial_training.py) |
| Defense: feature squeezing (preprocessing + detection) | ML01 | [defenses/feature_squeezing.py](defenses/feature_squeezing.py) |
| Defense: activation clustering (poison removal) | ML02 | [defenses/poison_filter.py](defenses/poison_filter.py) |
| Defense: hash-checked, weights-only model loading | ML10 / tampering | [src/common.py](src/common.py) |
| SAST (Bandit, Semgrep) + dependency audit | ML06 | [sast/run_sast.sh](sast/run_sast.sh) |
| SAST control sample (deliberately insecure, never run) | - | [sast/control/](sast/control/) |

The full write-up (threat model, SAST findings, all results) is in [REPORT.md](REPORT.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Run

```bash
./run_all.sh          # full pipeline; writes results/summary.md
./sast/run_sast.sh    # SAST reports in results/sast/
```

Or run one step at a time (each needs the steps above it):

```bash
python -m src.train
python -m attacks.evasion                  # --eps 0.2 by default
python -m attacks.poisoning                # --rate 0.03 --target 0 by default
python -m defenses.adversarial_training
python -m defenses.feature_squeezing
python -m defenses.poison_filter
python -m src.summary
```

## Outputs

- `results/*.json`: full metrics for every run (accuracy, loss, macro F1, confusion matrix, inference time)
- `results/figures/`: confusion matrices and example adversarial / backdoored images
- `results/summary.md`: comparison tables against the baseline
- `data/generated/`: adversarial test sets, poisoned training set, triggered test set (`.npz`)
- `models/`: trained weights and `manifest.json` with their SHA-256 hashes

All runs use seed 42. The attacks and defenses use an L-inf budget of eps = 0.2 in [0, 1] pixel space.
