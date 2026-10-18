"""Tasks 1 and 2: train the baseline CNN and report clean-test performance."""
import argparse
import time

from src.common import get_device, load_mnist, save_json, save_model, set_seed, train_model
from src.metrics import evaluate, plot_confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")

    set_seed()
    device = get_device()
    print(f"Execution device: {device}")

    x_train, y_train, x_test, y_test = load_mnist()
    start = time.perf_counter()
    model, history = train_model(
        x_train, y_train, epochs=args.epochs, log_prefix="[baseline] "
    )
    train_time = time.perf_counter() - start
    save_model(model, "baseline")

    metrics, _ = evaluate(model, x_test, y_test)
    metrics.update(
        epochs=args.epochs,
        train_loss_history=history,
        train_time_s=train_time,
        execution_device=str(device),
    )
    save_json("baseline", metrics)
    plot_confusion(
        metrics["confusion_matrix"], "Baseline - clean test set", "cm_baseline_clean.png"
    )

    print(
        f"Clean test accuracy {metrics['accuracy']:.4f}  loss {metrics['loss']:.4f}  "
        f"latency {metrics['single_image_latency_ms_median']:.2f} ms/image"
    )


if __name__ == "__main__":
    main()
