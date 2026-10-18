"""Blue team defense 2: feature squeezing (Xu et al., NDSS 2018). Input preprocessing plus detection.

Two "squeezers" remove detail an attacker relies on:
  - bit-depth reduction to 1 bit (MNIST digits are almost black-and-white anyway),
  - a 2x2 median filter.
Used two ways:
  1. Preprocessing: classify the squeezed image instead of the raw one.
  2. Detection: flag an input when the model's softmax output moves a lot (L1 distance) after
     squeezing. The threshold is set so that 5% of clean held-out *training* images are flagged,
     so the test set is not used for tuning.

Limitation for the report: this is tested against a non-adaptive attacker (the adversarial sets were
made without knowing about the squeezer). An attacker who knows about it can adapt.
"""
import torch
import torch.nn.functional as F

from src.common import load_mnist, load_model, load_npz, save_json, set_seed
from src.metrics import evaluate

FALSE_POSITIVE_RATE = 0.05


def bit_depth(x, bits=1):
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


def median_filter(x, k=2):
    padded = F.pad(x, (0, k - 1, 0, k - 1), mode="reflect")
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    return patches.reshape(*patches.shape[:4], -1).median(-1).values


SQUEEZERS = {"bit_depth_1": bit_depth, "median_2x2": median_filter}


def squeeze_score(model, x, batch_size=1000):
    """Max L1 distance between softmax(raw) and softmax(squeezed) over all squeezers."""
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            p = F.softmax(model(xb), 1)
            dists = [(p - F.softmax(model(s(xb)), 1)).abs().sum(1) for s in SQUEEZERS.values()]
            scores.append(torch.stack(dists).amax(0).cpu())
    return torch.cat(scores)


def main():
    set_seed()
    x_train, _, x_test, y_test = load_mnist()
    model = load_model("baseline")

    threshold = torch.quantile(squeeze_score(model, x_train[-5000:]), 1 - FALSE_POSITIVE_RATE).item()
    test_sets = {
        "clean": x_test,
        "fgsm": load_npz("adv_fgsm_test_baseline")["x"],
        "pgd": load_npz("adv_pgd_test_baseline")["x"],
    }

    results = {"threshold": threshold, "target_false_positive_rate": FALSE_POSITIVE_RATE}
    for name, x in test_sets.items():
        row = {"raw_accuracy": evaluate(model, x, y_test)[0]["accuracy"]}
        for sq_name, squeeze in SQUEEZERS.items():
            row[f"{sq_name}_accuracy"] = evaluate(model, squeeze(x), y_test)[0]["accuracy"]
        row["flagged_rate"] = (squeeze_score(model, x) > threshold).float().mean().item()
        results[name] = row
        print(f"  {name:5s} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    save_json("feature_squeezing", results)


if __name__ == "__main__":
    main()
