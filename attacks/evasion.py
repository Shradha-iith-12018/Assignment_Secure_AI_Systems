"""Red-team attack 1: FGSM and PGD evasion (OWASP ML01 Input Manipulation)."""
import argparse

import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from src.common import get_device, load_mnist, load_model, save_json, save_npz, set_seed
from src.metrics import evaluate, plot_confusion, plot_examples

DEFAULT_EPS = 0.2


def _validate_eps(eps: float) -> None:
    if not 0.0 < eps <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")


def art_classifier(model):
    device = next(model.parameters()).device
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def generate(model, x, eps=DEFAULT_EPS):
    """Return FGSM and PGD adversarial versions of ``x`` as CPU tensors."""
    _validate_eps(eps)
    if x.device.type != "cpu":
        x = x.cpu()
    clf = art_classifier(model)
    x_np = x.detach().numpy().astype("float32", copy=False)
    fgsm = FastGradientMethod(clf, eps=eps, batch_size=256).generate(x_np)
    pgd = ProjectedGradientDescent(
        clf,
        eps=eps,
        eps_step=eps / 10,
        max_iter=40,
        batch_size=256,
        verbose=False,
    ).generate(x_np)
    return torch.from_numpy(fgsm).float(), torch.from_numpy(pgd).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    args = parser.parse_args()
    try:
        _validate_eps(args.eps)
    except ValueError as exc:
        parser.error(str(exc))

    set_seed()
    print(f"Execution device: {get_device()}")
    _, _, x_test, y_test = load_mnist()
    model = load_model(args.model)

    print(f"Generating FGSM and PGD (eps={args.eps}) against '{args.model}'...")
    x_fgsm, x_pgd = generate(model, x_test, args.eps)
    save_npz(f"adv_fgsm_test_{args.model}", x=x_fgsm, y=y_test)
    save_npz(f"adv_pgd_test_{args.model}", x=x_pgd, y=y_test)

    results = {"model": args.model, "eps": args.eps, "device": str(next(model.parameters()).device)}
    preds = {}
    for name, x in (("clean", x_test), ("fgsm", x_fgsm), ("pgd", x_pgd)):
        metrics, preds[name] = evaluate(model, x, y_test)
        metrics["linf_perturbation_max"] = (x - x_test).abs().amax().item()
        metrics["l2_perturbation_mean"] = (
            (x - x_test).flatten(1).norm(dim=1).mean().item()
        )
        results[name] = metrics
        plot_confusion(
            metrics["confusion_matrix"],
            f"{args.model} - {name}",
            f"cm_{args.model}_{name}.png",
        )
        print(f"  {name:5s} accuracy {metrics['accuracy']:.4f}  loss {metrics['loss']:.4f}")

    save_json(f"evasion_{args.model}", results)
    plot_examples(
        [
            ("clean", x_test, preds["clean"]),
            (f"FGSM eps={args.eps}", x_fgsm, preds["fgsm"]),
            (f"PGD eps={args.eps}", x_pgd, preds["pgd"]),
        ],
        f"examples_evasion_{args.model}.png",
    )


if __name__ == "__main__":
    main()
