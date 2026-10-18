"""Blue-team defense 2: feature squeezing preprocessing and detection."""
import torch
import torch.nn.functional as F

from src.common import get_device, load_mnist, load_model, load_npz, save_json, set_seed
from src.metrics import evaluate

FALSE_POSITIVE_RATE = 0.05
CALIBRATION_SIZE = 5000


def bit_depth(x, bits=1):
    if bits <= 0:
        raise ValueError("bits must be > 0")
    levels = 2**bits - 1
    return torch.round(x * levels) / levels


def median_filter(x, k=2):
    if k <= 0:
        raise ValueError("filter size must be > 0")
    padded = F.pad(x, (0, k - 1, 0, k - 1), mode="reflect")
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    return patches.reshape(*patches.shape[:4], -1).median(-1).values


SQUEEZERS = {"bit_depth_1": bit_depth, "median_2x2": median_filter}


def squeeze_score(model, x, batch_size=1000):
    """Maximum L1 softmax distance between raw and squeezed inputs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size].to(device)
            p = F.softmax(model(xb), 1)
            dists = [
                (p - F.softmax(model(s(xb)), 1)).abs().sum(1)
                for s in SQUEEZERS.values()
            ]
            scores.append(torch.stack(dists).amax(0).cpu())
    return torch.cat(scores)


def main():
    set_seed()
    print(f"Execution device: {get_device()}")
    x_train, _, x_test, y_test = load_mnist()
    model = load_model("baseline")

    calibration = x_train[-CALIBRATION_SIZE:]
    threshold = torch.quantile(
        squeeze_score(model, calibration), 1 - FALSE_POSITIVE_RATE
    ).item()
    test_sets = {
        "clean": x_test,
        "fgsm": load_npz("adv_fgsm_test_baseline")["x"],
        "pgd": load_npz("adv_pgd_test_baseline")["x"],
    }

    results = {
        "threshold": threshold,
        "target_false_positive_rate": FALSE_POSITIVE_RATE,
        "calibration_size": len(calibration),
        "device": str(next(model.parameters()).device),
    }
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
