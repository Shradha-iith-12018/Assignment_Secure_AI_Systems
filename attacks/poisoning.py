"""Red team attack 2: backdoor data poisoning (OWASP ML02, Data Poisoning Attack).

A small white square (the trigger) is stamped on a fraction of the training images, and those images
are relabelled to a target class. The model trained on this data still looks normal on clean test
data, but predicts the target class whenever the trigger is present.

Outputs: data/generated/poisoned_train.npz, data/generated/triggered_test.npz, models/poisoned.pt.
"""
import argparse

import torch

from src.common import load_mnist, load_model, save_json, save_model, save_npz, set_seed, train_model
from src.metrics import evaluate, plot_confusion, plot_examples


def add_trigger(x):
    """Stamp a 4x4 white patch one pixel in from the bottom-right corner."""
    x = x.clone()
    x[:, :, -5:-1, -5:-1] = 1.0
    return x


def poison(x, y, rate, target, seed=0):
    """Return (poisoned x, poisoned y, indices of poisoned samples)."""
    g = torch.Generator().manual_seed(seed)
    candidates = (y != target).nonzero().squeeze(1)
    idx = candidates[torch.randperm(len(candidates), generator=g)[:int(rate * len(x))]]
    xp, yp = x.clone(), y.clone()
    xp[idx] = add_trigger(x[idx])
    yp[idx] = target
    return xp, yp, idx


def attack_success_rate(preds, target):
    return (preds == target).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=0.03, help="fraction of training set to poison")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    set_seed()
    x_train, y_train, x_test, y_test = load_mnist()
    xp, yp, idx = poison(x_train, y_train, args.rate, args.target)
    save_npz("poisoned_train", x=xp, y=yp, poison_idx=idx)
    print(f"Poisoned {len(idx)} of {len(x_train)} training images -> label {args.target}")

    # Triggered test set: every test image that is not already the target class, with the trigger.
    keep = y_test != args.target
    x_trig, y_trig = add_trigger(x_test[keep]), y_test[keep]
    save_npz("triggered_test", x=x_trig, y=y_trig)

    model, history = train_model(xp, yp, epochs=args.epochs, log_prefix="[poisoned] ")
    save_model(model, "poisoned")

    results = {"rate": args.rate, "target": args.target, "n_poisoned": len(idx), "train_loss_history": history}
    for model_name, m in (("baseline", load_model("baseline")), ("poisoned", model)):
        clean, _ = evaluate(m, x_test, y_test)
        trig, preds = evaluate(m, x_trig, y_trig)
        trig["attack_success_rate"] = attack_success_rate(preds, args.target)
        results[model_name] = {"clean": clean, "triggered": trig}
        plot_confusion(trig["confusion_matrix"], f"{model_name} - triggered test", f"cm_{model_name}_triggered.png")
        print(f"  {model_name:8s} clean acc {clean['accuracy']:.4f}  triggered acc {trig['accuracy']:.4f}  "
              f"attack success {trig['attack_success_rate']:.4f}")
        if model_name == "poisoned":
            plot_examples([("triggered", x_trig, preds)], "examples_backdoor.png")

    save_json("poisoning", results)


if __name__ == "__main__":
    main()
