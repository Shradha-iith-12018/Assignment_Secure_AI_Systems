"""Blue-team defense 3: activation clustering for poisoned-training-data detection."""
import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from attacks.poisoning import attack_success_rate
from src.common import (
    get_device,
    load_json,
    load_mnist,
    load_model,
    load_npz,
    save_json,
    save_model,
    set_seed,
    train_model,
)
from src.metrics import evaluate

MAX_CLUSTER_FRACTION = 0.35
MIN_SILHOUETTE = 0.35


def activations(model, x, batch_size=1000):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model.features(x[i : i + batch_size].to(device)).cpu()
                for i in range(0, len(x), batch_size)
            ]
        ).numpy()


def find_suspicious(acts, labels):
    """Return flagged global indices and per-class clustering diagnostics."""
    acts = np.asarray(acts)
    labels = np.asarray(labels)
    if acts.ndim != 2 or len(acts) != len(labels):
        raise ValueError("activations must be 2-D and aligned with labels")

    flagged, report = [], {}
    for c in range(10):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            raise ValueError(f"class {c} has too few samples for clustering")

        n_components = min(10, acts.shape[1], len(idx) - 1)
        reduced = PCA(n_components=n_components, random_state=0).fit_transform(acts[idx])
        clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        sizes = np.bincount(clusters, minlength=2)
        small = int(sizes.argmin())
        fraction = float(sizes[small] / len(idx))
        sample_size = min(3000, len(idx))
        silhouette = float(
            silhouette_score(reduced, clusters, sample_size=sample_size, random_state=0)
        )
        is_poisoned = bool(
            fraction < MAX_CLUSTER_FRACTION and silhouette > MIN_SILHOUETTE
        )
        report[c] = {
            "small_cluster_fraction": fraction,
            "silhouette": silhouette,
            "flagged": is_poisoned,
        }
        print(
            f"  class {c}: small cluster {fraction:.3f}  "
            f"silhouette {silhouette:.3f}  flagged {is_poisoned}"
        )
        if is_poisoned:
            flagged.extend(idx[clusters == small].tolist())

    return np.array(sorted(set(flagged)), dtype=np.int64), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")

    set_seed()
    print(f"Execution device: {get_device()}")
    _, _, x_test, y_test = load_mnist()
    data = load_npz("poisoned_train")
    required = {"x", "y", "poison_idx"}
    missing = required.difference(data)
    if missing:
        raise RuntimeError(f"poisoned_train.npz is missing arrays: {sorted(missing)}")

    xp, yp = data["x"], data["y"]
    true_idx = data["poison_idx"].numpy()
    poisoning_result = load_json("poisoning")
    if poisoning_result is None:
        raise FileNotFoundError("results/poisoning.json is required")
    target = int(poisoning_result["target"])

    poisoned_model = load_model("poisoned")
    flagged, report = find_suspicious(activations(poisoned_model, xp), yp.numpy())
    hits = len(np.intersect1d(flagged, true_idx))
    detection = {
        "flagged": len(flagged),
        "true_poison": len(true_idx),
        "precision": hits / max(len(flagged), 1),
        "recall": hits / max(len(true_idx), 1),
        "per_class": report,
    }
    print(
        f"Flagged {len(flagged)} samples: precision {detection['precision']:.3f}  "
        f"recall {detection['recall']:.3f}"
    )

    keep = np.setdiff1d(np.arange(len(xp)), flagged)
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    model, history = train_model(
        xp[keep_t], yp[keep_t], epochs=args.epochs, log_prefix="[sanitized] "
    )
    save_model(model, "sanitized")

    trig = load_npz("triggered_test")
    clean, _ = evaluate(model, x_test, y_test)
    triggered, preds = evaluate(model, trig["x"], trig["y"])
    triggered["attack_success_rate"] = attack_success_rate(preds, target)
    print(
        f"  sanitized clean acc {clean['accuracy']:.4f}  "
        f"attack success {triggered['attack_success_rate']:.4f}"
    )
    save_json(
        "poison_filter",
        {
            "detection": detection,
            "train_loss_history": history,
            "clean": clean,
            "triggered": triggered,
            "device": str(next(model.parameters()).device),
        },
    )


if __name__ == "__main__":
    main()
