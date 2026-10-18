"""Collect every results/*.json into results/summary.md: the comparison tables for the report."""
from src.common import RESULTS_DIR, load_json


def pct(v):
    return "-" if v is None else f"{100 * v:.2f}%"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    out = ["# Results summary", ""]

    base = load_json("baseline")
    if base:
        out += ["## Baseline (clean test set)", "", table(
            ["Accuracy", "Loss", "Macro F1", "Latency (ms/image)", "Batched (ms/image)", "Train time (s)"],
            [[pct(base["accuracy"]), f"{base['loss']:.4f}", f"{base['macro_f1']:.4f}",
              f"{base['single_image_latency_ms_median']:.3f}", f"{base['batch_ms_per_image']:.4f}",
              f"{base['train_time_s']:.0f}"]]), ""]

    ev, adv = load_json("evasion_baseline"), load_json("adversarial_training")
    if ev:
        rows = [["Baseline", pct(ev["clean"]["accuracy"]), pct(ev["fgsm"]["accuracy"]), pct(ev["pgd"]["accuracy"]), "-", "-"]]
        if adv:
            rows.append(["Adversarially trained", pct(adv["clean"]["accuracy"]),
                         pct(adv["whitebox_fgsm"]["accuracy"]), pct(adv["whitebox_pgd"]["accuracy"]),
                         pct(adv["transfer_fgsm"]["accuracy"]), pct(adv["transfer_pgd"]["accuracy"])])
        out += [f"## Evasion (ML01), eps = {ev['eps']}", "",
                "White-box = attack crafted against that model. Transfer = baseline's adversarial set.", "",
                table(["Model", "Clean", "FGSM (white-box)", "PGD (white-box)", "FGSM (transfer)", "PGD (transfer)"], rows), ""]

    fs = load_json("feature_squeezing")
    if fs:
        rows = [[name, pct(fs[name]["raw_accuracy"]), pct(fs[name]["bit_depth_1_accuracy"]),
                 pct(fs[name]["median_2x2_accuracy"]), pct(fs[name]["flagged_rate"])] for name in ("clean", "fgsm", "pgd")]
        out += ["## Feature squeezing on the baseline model", "",
                table(["Test set", "Raw acc", "1-bit acc", "Median 2x2 acc", "Flagged by detector"], rows), ""]

    po, pf = load_json("poisoning"), load_json("poison_filter")
    if po:
        rows = [[name, pct(po[name]["clean"]["accuracy"]), pct(po[name]["triggered"]["accuracy"]),
                 pct(po[name]["triggered"]["attack_success_rate"])] for name in ("baseline", "poisoned")]
        if pf:
            rows.append(["Sanitized (activation clustering)", pct(pf["clean"]["accuracy"]),
                         pct(pf["triggered"]["accuracy"]), pct(pf["triggered"]["attack_success_rate"])])
        out += [f"## Backdoor poisoning (ML02): {po['n_poisoned']} images ({pct(po['rate'])}) -> label {po['target']}", "",
                table(["Model", "Clean acc", "Triggered acc", "Attack success rate"], rows), ""]
        if pf:
            d = pf["detection"]
            out += [f"Activation clustering flagged {d['flagged']} samples: precision {d['precision']:.3f}, "
                    f"recall {d['recall']:.3f}.", ""]

    (RESULTS_DIR / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
