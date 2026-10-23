# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

A secure AI/ML engineering project for evaluating the robustness of a convolutional neural network trained on the MNIST handwritten-digit dataset.

The project covers:
- baseline CNN training and evaluation,
- ML threat modeling,
- adversarial evasion attacks,
- backdoor data poisoning,
- adversarial training,
- feature squeezing,
- activation-clustering-based poison filtering,
- secure model artifact handling,
- static application security testing,
- dependency vulnerability auditing,
- reproducible experiment execution.

The implementation is based on PyTorch and IBM Adversarial Robustness Toolbox (ART).

---

## Project Mapping

| Stage | Security / OWASP ML Risk | Main implementation |
|---|---|---|
| Baseline CNN and clean evaluation | — | `src/train.py`, `src/metrics.py` |
| FGSM adversarial examples | ML01 Input Manipulation | `attacks/evasion.py` |
| PGD adversarial examples | ML01 Input Manipulation | `attacks/evasion.py` |
| Backdoor data poisoning | ML02 Data Poisoning | `attacks/poisoning.py` |
| PGD adversarial training | ML01 Input Manipulation | `defenses/adversarial_training.py` |
| Feature squeezing | ML01 Input Manipulation | `defenses/feature_squeezing.py` |
| Activation clustering | ML02 Data Poisoning | `defenses/poison_filter.py` |
| Model artifact integrity | ML10 / artifact tampering | `src/common.py` |
| Non-pickle model serialization | Secure model loading | `src/common.py` |
| Bandit / Semgrep | Secure software engineering | `sast/run_sast.sh` |
| Dependency auditing | ML06 / supply-chain risk | `pip-audit` via `sast/run_sast.sh` |
| SAST control sample | Scanner validation | `sast/control/` |

---

## Repository Structure

```text
Assignment_Secure_AI_Systems/
├── src/
│   ├── common.py
│   ├── metrics.py
│   ├── summary.py
│   └── train.py
├── attacks/
│   ├── evasion.py
│   └── poisoning.py
├── defenses/
│   ├── adversarial_training.py
│   ├── feature_squeezing.py
│   └── poison_filter.py
├── sast/
│   ├── run_sast.sh
│   └── control/
│       └── insecure_patterns.py
├── scripts/
│   └── record_environment.py
├── data/
│   └── generated/
│       ├── adv_fgsm_test_baseline.npz
│       ├── adv_pgd_test_baseline.npz
│       ├── poisoned_train.npz
│       └── triggered_test.npz
├── models/
│   ├── baseline.safetensors
│   ├── poisoned.safetensors
│   ├── robust.safetensors
│   ├── sanitized.safetensors
│   └── manifest.json
├── results/
│   ├── baseline.json
│   ├── evasion_baseline.json
│   ├── poisoning.json
│   ├── adversarial_training.json
│   ├── feature_squeezing.json
│   ├── poison_filter.json
│   ├── environment.json
│   ├── summary.md
│   ├── figures/
│   └── sast/
├── requirements.txt
├── requirements-dev.txt
├── run_all.sh
├── Report.pdf
└── README.md
```

---

## Model Architecture

The CNN processes MNIST images of shape `1 x 28 x 28`.

```text
Conv2d(1 -> 32, 3x3)
ReLU
Conv2d(32 -> 64, 3x3)
ReLU
MaxPool2d(2x2)
Dropout(0.25)
Flatten
Linear(9216 -> 128)
ReLU
Dropout(0.5)
Linear(128 -> 10)
```

Input images are represented in `[0,1]` pixel space. Normalization is performed inside the model so attacks and defenses operate in ordinary pixel space.

---

## Environment

The canonical CPU experiment used:

```text
Python                  3.10.20
PyTorch                 2.11.0
torchvision             0.26.0
NumPy                   2.2.6
scikit-learn            1.7.2
matplotlib              3.10.8
ART                     1.20.1
safetensors             0.8.0
Bandit                  1.9.4
Semgrep                 1.181.0
pip-audit               2.10.1
```

The complete environment record is stored in `results/environment.json`.

To regenerate it:

```bash
SSE_DEVICE=cpu python scripts/record_environment.py
```

or, on a compatible CUDA system:

```bash
SSE_DEVICE=cuda python scripts/record_environment.py
```

---

## Installation

A Python 3.10+ environment is recommended.

```bash
conda create -n secure-ai python=3.10
conda activate secure-ai

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

python -m pip check
```

If an existing PyTorch environment is already available, install the project dependencies into that environment.

---

## CPU / GPU Selection

Execution is controlled through `SSE_DEVICE`.

Force CPU:

```bash
SSE_DEVICE=cpu python -m src.train
```

Require CUDA:

```bash
SSE_DEVICE=cuda python -m src.train
```

Automatic selection:

```bash
SSE_DEVICE=auto python -m src.train
```

`auto` uses CUDA only when a real CUDA operation succeeds. This is more robust than relying only on `torch.cuda.is_available()`.

---

## Baseline Training

```bash
SSE_DEVICE=cpu python -m src.train
```

Default training configuration:

```text
Optimizer       Adam
Learning rate   0.001
Batch size      128
Epochs          5
Loss            Cross-entropy
Seed            42
```

Output model:

```text
models/baseline.safetensors
```

---

## Red Teaming

### FGSM and PGD

```bash
SSE_DEVICE=cpu python -m attacks.evasion
```

Threat model: white-box, untargeted, `L_inf` budget `epsilon = 0.2`.

Generated datasets:

```text
data/generated/adv_fgsm_test_baseline.npz
data/generated/adv_pgd_test_baseline.npz
```

OWASP ML mapping:

```text
ML01 — Input Manipulation
```

### Backdoor Data Poisoning

```bash
SSE_DEVICE=cpu python -m attacks.poisoning
```

The attack poisons 3% of the training set, stamps a fixed trigger, relabels poisoned samples to class `0`, and retrains the model.

Generated datasets:

```text
data/generated/poisoned_train.npz
data/generated/triggered_test.npz
```

Output model:

```text
models/poisoned.safetensors
```

OWASP ML mapping:

```text
ML02 — Data Poisoning
```

---

## Blue Teaming

### Adversarial Training

```bash
SSE_DEVICE=cpu python -m defenses.adversarial_training
```

The model is retrained using clean samples mixed with dynamically generated PGD examples.

Output model:

```text
models/robust.safetensors
```

Evaluation includes clean data, transfer FGSM/PGD, and fresh white-box FGSM/PGD.

### Feature Squeezing

```bash
SSE_DEVICE=cpu python -m defenses.feature_squeezing
```

Implemented squeezers:
- 1-bit pixel quantization
- `2 x 2` median filtering

### Activation Clustering

```bash
SSE_DEVICE=cpu python -m defenses.poison_filter
```

The defense:
1. extracts penultimate-layer activations,
2. reduces them using PCA,
3. clusters each class with `k=2`,
4. flags suspicious small, well-separated clusters,
5. removes suspected poisoned samples,
6. retrains the model.

Output model:

```text
models/sanitized.safetensors
```

---

## Secure Model Artifact Handling

The project uses `safetensors` instead of pickle-backed PyTorch serialization.

This avoids the arbitrary-code-execution risk associated with unsafe pickle deserialization.

Model files are additionally protected by SHA-256 integrity verification. Trusted hashes are stored in:

```text
models/manifest.json
```

Before loading a model, the implementation:
1. validates the artifact name,
2. verifies its SHA-256 digest,
3. loads the tensor-only `.safetensors` artifact,
4. restores the model state.

---

## Static Application Security Testing

Run:

```bash
./sast/run_sast.sh
```

The pipeline uses:
- Bandit
- Semgrep
- pip-audit

Application directories scanned:

```text
src/
attacks/
defenses/
scripts/
```

Final application scan:

```text
Bandit application findings:     0
Semgrep application findings:    0
pip-audit known vulnerabilities: 0
```

Reports are stored in `results/sast/`.

---

## SAST Control Sample

The repository includes:

```text
sast/control/insecure_patterns.py
```

This file contains deliberately insecure examples and is never used by the application.

Its purpose is to validate that the SAST configuration detects insecure patterns.

Expected behavior:

```text
Application code       -> no findings
Control sample         -> findings detected
```

Scanner exit statuses are recorded in:

```text
results/sast/status.txt
```

---

## SAST Remediation

During the security review, Semgrep flagged `torch.load()` because PyTorch model deserialization can involve pickle.

The earlier implementation used `weights_only=True` and SHA-256 verification, but the project was further hardened by replacing pickle-backed model serialization with `safetensors`.

Bandit also flagged subprocess use in the environment-recording script. That code was rewritten to use Python package metadata directly instead of spawning external commands.

Final result:

```text
Bandit application scan  -> 0 findings
Semgrep application scan -> 0 findings
pip-audit                 -> 0 known vulnerabilities
```

---

## Canonical CPU Results

### Baseline

| Metric | Value |
|---|---:|
| Clean accuracy | 99.10% |
| Test loss | 0.0304 |
| Macro F1 | 0.9909 |
| Single-image inference latency | 0.970 ms |
| Batched inference latency | 0.6153 ms/image |
| Training time | 348 s |

### Adversarial Evasion

| Model | Clean | FGSM | PGD |
|---|---:|---:|---:|
| Baseline | 99.10% | 49.09% | 0.89% |
| Adversarially trained | 98.78% | 96.05% | 93.29% |

Transfer performance of the adversarially trained model:

| Attack | Accuracy |
|---|---:|
| FGSM transfer | 97.14% |
| PGD transfer | 97.09% |

### Feature Squeezing

| Dataset | Raw | 1-bit | Median 2x2 | Detector flagged |
|---|---:|---:|---:|---:|
| Clean | 99.10% | 98.91% | 98.09% | 6.78% |
| FGSM | 49.09% | 93.77% | 61.36% | 76.52% |
| PGD | 0.89% | 93.12% | 15.65% | 97.69% |

### Backdoor Poisoning

| Model | Clean accuracy | Triggered accuracy | Attack success rate |
|---|---:|---:|---:|
| Baseline | 99.10% | 98.97% | 0.11% |
| Poisoned | 98.94% | 0.01% | 99.99% |
| Sanitized | 99.08% | 98.78% | 0.29% |

Activation-clustering poison detection:

```text
Flagged samples: 1795
Precision:       1.000
Recall:          0.997
```

---

## Run the Complete Pipeline

CPU:

```bash
SSE_DEVICE=cpu ./run_all.sh
```

Compatible CUDA GPU:

```bash
SSE_DEVICE=cuda ./run_all.sh
```

The complete pipeline regenerates:
- baseline model,
- clean metrics,
- adversarial examples,
- poisoned dataset,
- poisoned model,
- adversarially trained model,
- feature-squeezing results,
- activation-clustering defense,
- sanitized model,
- figures,
- summary tables,
- environment metadata.

Do not mix timing or performance measurements from different machines into a single reported result set.

---

## Results

Machine-readable results are stored in `results/`.

Key files:

```text
results/baseline.json
results/evasion_baseline.json
results/poisoning.json
results/adversarial_training.json
results/feature_squeezing.json
results/poison_filter.json
results/environment.json
results/summary.md
```

Figures are stored in:

```text
results/figures/
```

Security reports are stored in:

```text
results/sast/
```

---

## Reproducibility

All experiments use:

```text
seed = 42
```

Results can still vary slightly across hardware, PyTorch versions, BLAS implementations, CPU/GPU execution, and CUDA/cuDNN versions.

The repository therefore stores `results/environment.json` alongside the canonical results.

---

## Threat Modeling

The project threat model includes:
- assets,
- trust boundaries,
- attacker capabilities,
- STRIDE analysis,
- OWASP Machine Learning Security Top 10 mapping,
- LINDDUN privacy considerations.

Implemented attacker models:

```text
A1 — inference-time adversarial attacker
A2 — training-data poisoner
```

Additional modeled threats include:
- model artifact tampering,
- model extraction,
- membership inference,
- model inversion,
- supply-chain compromise.

---

## Report

The final project report is available as:

```text
Report.pdf
```

All reported metrics come from the canonical result set `cpu-canonical-v1`. The tag `cpu-shradha-v1` preserves an independent CPU reproduction using the same source code, which the report uses as a reproducibility check.

---

## Contributors

This project was completed by two students as part of the Secure Software Engineering course:

```text
Shradha Sanjaykant Suryawanshi   CS25MTECH12018
Suresh Kumar Baddipudi           CS25M26P140002
```

- Shradha Suryawanshi developed the initial end-to-end implementation (baseline CNN, red-team attacks, blue-team defenses, initial SAST pipeline, experiments) and independently reproduced the complete pipeline on a second CPU system (`cpu-shradha-v1`).
- Suresh Kumar Baddipudi performed security hardening and refactoring, improved reproducibility and device portability, strengthened SAST evidence, migrated model serialization to `safetensors`, and generated the canonical result set (`cpu-canonical-v1`).

Both contributors participated in validation, interpretation of results, and preparation of the final report. The full contribution statement is in Section 11 of the report.

---

## Security Note

The deliberately insecure code under:

```text
sast/control/
```

exists only to validate the SAST tools and is not imported or executed by the application.

---

## License / Academic Use

This repository was developed for academic coursework in Secure Software Engineering and Secure AI Systems.
