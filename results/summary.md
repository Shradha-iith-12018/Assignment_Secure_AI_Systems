# Results summary

## Baseline (clean test set)

| Accuracy | Loss | Macro F1 | Latency (ms/image) | Batched (ms/image) | Train time (s) |
|---|---|---|---|---|---|
| 99.10% | 0.0304 | 0.9909 | 0.970 | 0.6153 | 348 |

## Evasion (ML01), eps = 0.2

White-box = attack crafted against that model. Transfer = baseline's adversarial set.

| Model | Clean | FGSM (white-box) | PGD (white-box) | FGSM (transfer) | PGD (transfer) |
|---|---|---|---|---|---|
| Baseline | 99.10% | 49.09% | 0.89% | - | - |
| Adversarially trained | 98.78% | 96.05% | 93.29% | 97.14% | 97.09% |

## Feature squeezing on the baseline model

| Test set | Raw acc | 1-bit acc | Median 2x2 acc | Flagged by detector |
|---|---|---|---|---|
| clean | 99.10% | 98.91% | 98.09% | 6.78% |
| fgsm | 49.09% | 93.77% | 61.36% | 76.52% |
| pgd | 0.89% | 93.12% | 15.65% | 97.69% |

## Backdoor poisoning (ML02): 1800 images (3.00%) -> label 0

| Model | Clean acc | Triggered acc | Attack success rate |
|---|---|---|---|
| baseline | 99.10% | 98.97% | 0.11% |
| poisoned | 98.94% | 0.01% | 99.99% |
| Sanitized (activation clustering) | 99.08% | 98.78% | 0.29% |

Activation clustering flagged 1795 samples: precision 1.000, recall 0.997.
