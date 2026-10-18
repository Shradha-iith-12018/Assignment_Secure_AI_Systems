"""Red-team attack 2: backdoor data poisoning (OWASP ML02 Data Poisoning)."""
import argparse

import torch

from src.common import (
    get_device,
    load_mnist,
    load_model,
    save_json,
    save_model,
    save_npz,
    set_seed,
    train_model,
    validate_dataset,
)
from src.metrics import evaluate, plot_confusion, plot_examples


def add_trigger(x):
    """Stamp a 4x4 white patch one pixel in from the bottom-right corner."""
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
        raise ValueError("trigger input must have shape (N, 1, 28, 28)")
    x = x.clone()
    x[:, :, -5:-1, -5:-1] = 1.0
    return x


def poison(x, y, rate, target, seed=0):
    """Return poisoned images, labels, and the poisoned sample indices."""
    validate_dataset(x, y, name="poisoning input")
    if not 0.0 < rate < 1.0:
        raise ValueError("poisoning rate must be in (0, 1)")
    if target not in range(10):
        raise ValueError("target must be an integer in [0, 9]")

    g = torch.Generator().manual_seed(seed)
    candidates = (y != target).nonzero(as_tuple=False).squeeze(1)
    n_poison = int(round(rate * len(x)))
    if n_poison <= 0:
        raise ValueError("poisoning rate selects zero samples")
    if n_poison > len(candidates):
        raise ValueError("poisoning rate selects more samples than are eligible")

    idx = candidates[torch.randperm(len(candidates), generator=g)[:n_poison]]
    xp, yp = x.clone(), y.clone()
    xp[idx] = add_trigger(x[idx])
    yp[idx] = target
    return xp, yp, idx


def attack_success_rate(preds, target):
    if target not in range(10):
        raise ValueError("target must be an integer in [0, 9]")
    return (preds == target).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=0.03, help="fraction of training set to poison")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 < args.rate < 1.0:
        parser.error("--rate must be in (0, 1)")
    if args.target not in range(10):
        parser.error("--target must be in [0, 9]")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")

    set_seed()
    print(f"Execution device: {get_device()}")
    x_train, y_train, x_test, y_test = load_mnist()
    xp, yp, idx = poison(x_train, y_train, args.rate, args.target)
    save_npz("poisoned_train", x=xp, y=yp, poison_idx=idx)
    print(f"Poisoned {len(idx)} of {len(x_train)} training images -> label {args.target}")

    keep = y_test != args.target
    x_trig, y_trig = add_trigger(x_test[keep]), y_test[keep]
    save_npz("triggered_test", x=x_trig, y=y_trig)

    model, history = train_model(xp, yp, epochs=args.epochs, log_prefix="[poisoned] ")
    save_model(model, "poisoned")

    results = {
        "rate": args.rate,
        "target": args.target,
        "n_poisoned": len(idx),
        "train_loss_history": history,
        "device": str(next(model.parameters()).device),
    }
    for model_name, m in (("baseline", load_model("baseline")), ("poisoned", model)):
        clean, _ = evaluate(m, x_test, y_test)
        trig, preds = evaluate(m, x_trig, y_trig)
        trig["attack_success_rate"] = attack_success_rate(preds, args.target)
        results[model_name] = {"clean": clean, "triggered": trig}
        plot_confusion(
            trig["confusion_matrix"],
            f"{model_name} - triggered test",
            f"cm_{model_name}_triggered.png",
        )
        print(
            f"  {model_name:8s} clean acc {clean['accuracy']:.4f}  "
            f"triggered acc {trig['accuracy']:.4f}  attack success {trig['attack_success_rate']:.4f}"
        )
        if model_name == "poisoned":
            plot_examples([("triggered", x_trig, preds)], "examples_backdoor.png")

    save_json("poisoning", results)


if __name__ == "__main__":
    main()
