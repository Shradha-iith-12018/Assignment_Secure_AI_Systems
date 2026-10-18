"""Blue team defense 1: PGD adversarial training (counters ML01).

Every training batch is half clean images and half PGD adversarial images (Madry et al., 2018).
The adversarial half is crafted against the model *as it trains*, not taken from the baseline's
saved adversarial set, so the model cannot just memorise one fixed set of perturbations.

The defended model ("robust") is then tested on:
  - the clean test set,
  - the adversarial sets made against the baseline (transfer attack),
  - fresh white-box FGSM/PGD made against the robust model itself (the honest test).
"""
import argparse
import time

import torch
import torch.nn.functional as F

from attacks.evasion import DEFAULT_EPS, generate
from src.common import load_mnist, load_npz, save_json, save_model, set_seed, train_model
from src.metrics import evaluate, plot_confusion


def pgd(model, x, y, eps, step, iters):
    """L-inf PGD with a random start, used inside the training loop."""
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--iters", type=int, default=10, help="PGD steps during training")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    set_seed()
    x_train, y_train, x_test, y_test = load_mnist()

    step = 2.5 * args.eps / args.iters
    start = time.perf_counter()
    model, history = train_model(
        x_train, y_train, epochs=args.epochs, log_prefix="[robust] ",
        adversary=lambda m, xb, yb: pgd(m, xb, yb, args.eps, step, args.iters),
    )
    train_time = time.perf_counter() - start
    save_model(model, "robust")

    fresh_fgsm, fresh_pgd = generate(model, x_test, args.eps)
    test_sets = {
        "clean": x_test,
        "transfer_fgsm": load_npz("adv_fgsm_test_baseline")["x"],
        "transfer_pgd": load_npz("adv_pgd_test_baseline")["x"],
        "whitebox_fgsm": fresh_fgsm,
        "whitebox_pgd": fresh_pgd,
    }
    results = {"eps": args.eps, "train_iters": args.iters, "epochs": args.epochs,
               "train_time_s": train_time, "train_loss_history": history}
    for name, x in test_sets.items():
        metrics, _ = evaluate(model, x, y_test)
        results[name] = metrics
        plot_confusion(metrics["confusion_matrix"], f"robust - {name}", f"cm_robust_{name}.png")
        print(f"  {name:14s} accuracy {metrics['accuracy']:.4f}  loss {metrics['loss']:.4f}")
    save_json("adversarial_training", results)


if __name__ == "__main__":
    main()
