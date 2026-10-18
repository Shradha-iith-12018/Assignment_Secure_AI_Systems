# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

A PyTorch CNN trained on MNIST, attacked using OWASP ML risks, then defended and re-evaluated.

| Stage | OWASP ML risk | Code |
|---|---|---|
| Baseline CNN, clean metrics | - | `src/train.py`, `src/metrics.py` |
| FGSM and PGD adversarial examples | ML01 Input Manipulation | `attacks/evasion.py` |
| Backdoor data poisoning | ML02 Data Poisoning | `attacks/poisoning.py` |
| PGD adversarial training | ML01 | `defenses/adversarial_training.py` |
| Feature squeezing | ML01 | `defenses/feature_squeezing.py` |
| Activation clustering | ML02 | `defenses/poison_filter.py` |
| Hash-checked, weights-only model loading | ML10 / artifact tampering | `src/common.py` |
| Bandit + Semgrep + pip-audit | Secure software / ML06 supply chain | `sast/run_sast.sh` |
| Deliberately insecure scanner control sample | - | `sast/control/` |

The current PDF write-up is `Report.pdf`. A final LaTeX report will be generated after the canonical final experiment run.

## Environment

Python 3.10+ is recommended. Install runtime and security-analysis dependencies into an isolated environment:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check
```

Record the machine/software environment used for a final experiment:

```bash
python scripts/record_environment.py
```

The generated `results/environment.json` should come from the machine that produced the final reported results.

## CPU/GPU selection

Device selection is controlled by the `SSE_DEVICE` environment variable:

```bash
SSE_DEVICE=cpu  python -m src.train   # force CPU
SSE_DEVICE=cuda python -m src.train   # require a usable CUDA GPU
SSE_DEVICE=auto python -m src.train   # use CUDA only if a real CUDA operation succeeds
```

`auto` is the default. The CUDA probe avoids relying only on `torch.cuda.is_available()`, which can be true even when the installed PyTorch build does not support the detected GPU's compute capability.

## Run one stage at a time

```bash
SSE_DEVICE=cpu python -m src.train
SSE_DEVICE=cpu python -m attacks.evasion
SSE_DEVICE=cpu python -m attacks.poisoning
SSE_DEVICE=cpu python -m defenses.adversarial_training
SSE_DEVICE=cpu python -m defenses.feature_squeezing
SSE_DEVICE=cpu python -m defenses.poison_filter
python -m src.summary
./sast/run_sast.sh
```

For a compatible GPU machine, replace `cpu` with `cuda`.

## Run the complete pipeline

```bash
SSE_DEVICE=cpu ./run_all.sh
# or, on a compatible GPU:
SSE_DEVICE=cuda ./run_all.sh
```

Do not mix timings/metrics from different machines in the final report. Pick one canonical final run and regenerate its JSON files, figures, summary, and environment record together.

## Outputs

- `results/*.json`: experiment metrics
- `results/environment.json`: final execution environment
- `results/figures/`: confusion matrices and attack examples
- `results/summary.md`: comparison tables
- `results/sast/`: Bandit, Semgrep, and pip-audit evidence
- `data/generated/`: adversarial, poisoned, and triggered datasets
- `models/manifest.json`: SHA-256 model-artifact manifest

All experiments use seed 42. The default adversarial L-infinity budget is epsilon = 0.2 in [0, 1] pixel space.
