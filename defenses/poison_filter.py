"""Blue team defense 3: activation clustering (Chen et al., 2018) to find and remove poisoned
training data (counters ML02).

For each training label, take the poisoned model's penultimate-layer activations, reduce them to
10 dimensions with PCA and split them into 2 clusters with k-means. Clean classes split into two
similar, overlapping halves. A class containing backdoored images splits into a small, well-separated
cluster (the poison) and the rest. A cluster is flagged when it is small AND well separated.
The flagged samples are removed and the model is retrained on what is left.
"""
import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from attacks.poisoning import attack_success_rate
from src.common import (get_device, load_mnist, load_model, load_npz, load_json, save_json, save_model,
                        set_seed, train_model)
from src.metrics import evaluate

MAX_CLUSTER_FRACTION = 0.35
MIN_SILHOUETTE = 0.35  # clean classes measured 0.17-0.26, the backdoored class 0.54


def activations(model, x, batch_size=1000):
    device = get_device()
    with torch.no_grad():
        return torch.cat([model.features(x[i:i + batch_size].to(device)).cpu()
                          for i in range(0, len(x), batch_size)]).numpy()


def find_suspicious(acts, labels):
    """Return (indices flagged as poisoned, per-class diagnostics)."""
    flagged, report = [], {}
    for c in range(10):
        idx = np.where(labels == c)[0]
        reduced = PCA(n_components=10, random_state=0).fit_transform(acts[idx])
        clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        sizes = np.bincount(clusters, minlength=2)
        small = int(sizes.argmin())
        fraction = sizes[small] / len(idx)
        silhouette = float(silhouette_score(reduced, clusters, sample_size=3000, random_state=0))
        is_poisoned = bool(fraction < MAX_CLUSTER_FRACTION and silhouette > MIN_SILHOUETTE)
        report[c] = {"small_cluster_fraction": float(fraction), "silhouette": silhouette, "flagged": is_poisoned}
        print(f"  class {c}: small cluster {fraction:.3f}  silhouette {silhouette:.3f}  flagged {is_poisoned}")
        if is_poisoned:
            flagged.extend(idx[clusters == small].tolist())
    return np.array(flagged, dtype=np.int64), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    set_seed()
    _, _, x_test, y_test = load_mnist()
    data = load_npz("poisoned_train")
    xp, yp, true_idx = data["x"], data["y"], data["poison_idx"].numpy()
    target = load_json("poisoning")["target"]

    flagged, report = find_suspicious(activations(load_model("poisoned"), xp), yp.numpy())
    hits = len(np.intersect1d(flagged, true_idx))
    detection = {
        "flagged": len(flagged),
        "true_poison": len(true_idx),
        "precision": hits / max(len(flagged), 1),
        "recall": hits / len(true_idx),
        "per_class": report,
    }
    print(f"Flagged {len(flagged)} samples: precision {detection['precision']:.3f}  recall {detection['recall']:.3f}")

    keep = np.setdiff1d(np.arange(len(xp)), flagged)
    model, history = train_model(xp[keep], yp[keep], epochs=args.epochs, log_prefix="[sanitized] ")
    save_model(model, "sanitized")

    trig = load_npz("triggered_test")
    clean, _ = evaluate(model, x_test, y_test)
    triggered, preds = evaluate(model, trig["x"], trig["y"])
    triggered["attack_success_rate"] = attack_success_rate(preds, target)
    print(f"  sanitized clean acc {clean['accuracy']:.4f}  attack success {triggered['attack_success_rate']:.4f}")
    save_json("poison_filter", {"detection": detection, "train_loss_history": history,
                                "clean": clean, "triggered": triggered})


if __name__ == "__main__":
    main()
