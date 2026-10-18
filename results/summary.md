# Results summary

## Baseline (clean test set)

| Accuracy | Loss | Macro F1 | Latency (ms/image) | Batched (ms/image) | Train time (s) |
|---|---|---|---|---|---|
| 98.98% | 0.0352 | 0.9897 | 0.412 | 0.2237 | 202 |

## Evasion (ML01), eps = 0.2

White-box = attack crafted against that model. Transfer = baseline's adversarial set.

| Model | Clean | FGSM (white-box) | PGD (white-box) | FGSM (transfer) | PGD (transfer) |
|---|---|---|---|---|---|
| Baseline | 98.98% | 54.26% | 0.95% | - | - |
| Adversarially trained | 98.81% | 96.01% | 93.31% | 97.30% | 97.26% |

## Feature squeezing on the baseline model

| Test set | Raw acc | 1-bit acc | Median 2x2 acc | Flagged by detector |
|---|---|---|---|---|
| clean | 98.98% | 98.66% | 98.04% | 6.45% |
| fgsm | 54.26% | 93.02% | 66.25% | 74.21% |
| pgd | 0.95% | 92.36% | 14.05% | 97.05% |

## Backdoor poisoning (ML02): 1800 images (3.00%) -> label 0

| Model | Clean acc | Triggered acc | Attack success rate |
|---|---|---|---|
| baseline | 98.98% | 98.86% | 0.11% |
| poisoned | 99.08% | 0.02% | 99.98% |
| Sanitized (activation clustering) | 99.10% | 97.74% | 0.91% |

Activation clustering flagged 1789 samples: precision 1.000, recall 0.994.
