"""Evaluation metrics and figure helpers for the MNIST experiments."""
from __future__ import annotations

import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score  # noqa: E402

from src.common import RESULTS_DIR, validate_dataset  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, x, y, batch_size=1000, latency_samples=200):
    """Return ``(metrics, predictions)`` for a model and labelled image tensor."""
    validate_dataset(x, y, name="evaluation data")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if latency_samples < 0:
        raise ValueError("latency_samples must be >= 0")

    device = next(model.parameters()).device
    model.eval()
    logits, latencies = [], []

    with torch.no_grad():
        _sync(device)
        start = time.perf_counter()
        for i in range(0, len(x), batch_size):
            logits.append(model(x[i : i + batch_size].to(device)).cpu())
        _sync(device)
        batch_time = time.perf_counter() - start

        for i in range(min(latency_samples, len(x))):
            xi = x[i : i + 1].to(device)
            _sync(device)
            t0 = time.perf_counter()
            model(xi)
            _sync(device)
            latencies.append(time.perf_counter() - t0)

    logits = torch.cat(logits)
    preds = logits.argmax(1)
    metrics = {
        "n": len(y),
        "accuracy": (preds == y).float().mean().item(),
        "loss": F.cross_entropy(logits, y).item(),
        "macro_f1": float(f1_score(y.numpy(), preds.numpy(), average="macro")),
        "confusion_matrix": confusion_matrix(
            y.numpy(), preds.numpy(), labels=list(range(10))
        ).tolist(),
        "batch_ms_per_image": batch_time / len(x) * 1000,
        "single_image_latency_ms_median": (
            statistics.median(latencies) * 1000 if latencies else None
        ),
        "device": str(device),
    }
    return metrics, preds


def plot_confusion(cm, title, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=torch.tensor(cm).numpy()).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def plot_examples(rows, filename, n=8):
    """Draw rows of images and predictions; each row is ``(title, images, preds)``."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if not rows:
        raise ValueError("rows must not be empty")

    n = min(n, *(len(images) for _, images, _ in rows))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(rows), n, figsize=(n * 1.2, len(rows) * 1.5), squeeze=False
    )
    for r, (title, images, preds) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(images[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"pred {int(preds[c])}", fontsize=8)
        axes[r][0].set_ylabel(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)
