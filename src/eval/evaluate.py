"""
Evaluation and visualization for all models.
Generates ROC curves, MCC comparison bars, confusion matrices, and summary tables.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

PRIMARY_GREEN = "#2e7d32"
SECONDARY_GREEN = "#66bb6a"
LIGHT_GREEN = "#a5d6a7"
PALE_GREEN = "#e8f5e9"
MUTED_GRAY = "#9e9e9e"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_all_results() -> list[dict]:
    """Load results from all model experiments."""
    all_results = []
    for fname in ["baseline_results.json", "esm_mlp_results.json", "esm_650m_mlp_results.json", "esm_finetune_results.json"]:
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                all_results.extend(json.load(f))
    return all_results


def _get_model_style(model_name: str) -> dict:
    """Return color and linewidth based on model name."""
    if "ESM" in model_name.upper() or "esm" in model_name.lower():
        return {"color": PRIMARY_GREEN, "linewidth": 3.0, "alpha": 1.0}
    return {"color": MUTED_GRAY, "linewidth": 1.8, "alpha": 0.7}


def plot_roc_curves(results: list[dict]):
    """Plot ROC curves for all models, split by evaluation type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, split_type in zip(axes, ["split_random", "split_cluster"]):
        split_results = [r for r in results if r.get("split_type") == split_type]

        non_esm = [r for r in split_results if "esm" not in r.get("model", "").lower()]
        esm = [r for r in split_results if "esm" in r.get("model", "").lower()]

        for r in non_esm:
            if "roc_fpr" in r and "roc_tpr" in r:
                style = _get_model_style(r["model"])
                label = f"{r['model']} (AUROC={r['auroc']:.3f})"
                ax.plot(r["roc_fpr"], r["roc_tpr"], label=label, **style)

        for r in esm:
            if "roc_fpr" in r and "roc_tpr" in r:
                style = _get_model_style(r["model"])
                label = f"{r['model']} (AUROC={r['auroc']:.3f})"
                ax.plot(r["roc_fpr"], r["roc_tpr"], label=label, **style)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        title = "Random Split" if "random" in split_type else "Cluster Split (<=40% identity)"
        ax.set_title(f"ROC Curves — {title}")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.grid(True, alpha=0.15, linestyle="-", color=MUTED_GRAY)

    plt.tight_layout()
    outpath = RESULTS_DIR / "roc_curves.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {outpath}")


def plot_mcc_comparison(results: list[dict]):
    """Bar chart comparing MCC across models and split types."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Split": "Random" if "random" in r.get("split_type", "") else "Cluster",
            "MCC": r["mcc"],
            "MCC_lo": r["mcc_ci"][0],
            "MCC_hi": r["mcc_ci"][1],
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    models = df["Model"].unique()
    x = np.arange(len(models))
    width = 0.35

    random_data = df[df["Split"] == "Random"].set_index("Model")
    cluster_data = df[df["Split"] == "Cluster"].set_index("Model")

    for i, m in enumerate(models):
        is_esm = "esm" in m.lower()
        edgecolor_r = PRIMARY_GREEN if is_esm else "none"
        edgecolor_c = PRIMARY_GREEN if is_esm else "none"
        lw = 2.0 if is_esm else 0.0

        if m in random_data.index:
            r = random_data.loc[m]
            err = [[r["MCC"] - r["MCC_lo"]], [r["MCC_hi"] - r["MCC"]]]
            ax.bar(i - width/2, r["MCC"], width, yerr=err,
                   label="Random Split" if i == 0 else "",
                   color=LIGHT_GREEN, capsize=4, alpha=0.9,
                   edgecolor=edgecolor_r, linewidth=lw)
        if m in cluster_data.index:
            c = cluster_data.loc[m]
            err = [[c["MCC"] - c["MCC_lo"]], [c["MCC_hi"] - c["MCC"]]]
            ax.bar(i + width/2, c["MCC"], width, yerr=err,
                   label="Cluster Split" if i == 0 else "",
                   color=PRIMARY_GREEN, capsize=4, alpha=0.9,
                   edgecolor=edgecolor_c, linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Matthews Correlation Coefficient")
    ax.set_title("MCC: Random Split vs. Cluster Split (Generalization Test)")
    ax.legend(framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.9, color=MUTED_GRAY, linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", alpha=0.15, linestyle="-", color=MUTED_GRAY)

    plt.tight_layout()
    outpath = RESULTS_DIR / "mcc_comparison.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved MCC comparison to {outpath}")


def plot_confusion_matrices(results: list[dict]):
    """Plot confusion matrices for the best model on each split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_type in zip(axes, ["split_random", "split_cluster"]):
        split_results = [r for r in results if r.get("split_type") == split_type]
        if not split_results:
            continue
        best = max(split_results, key=lambda r: r.get("auroc", 0))
        cm = np.array(best["confusion_matrix"])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Non-Toxin", "Toxin"],
                    yticklabels=["Non-Toxin", "Toxin"],
                    annot_kws={"size": 14, "weight": "bold"},
                    linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        title = "Random Split" if "random" in split_type else "Cluster Split"
        ax.set_title(f"Best Model: {best['model']} — {title}")

    plt.tight_layout()
    outpath = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to {outpath}")


def generate_summary_table(results: list[dict]):
    """Generate a summary table of all metrics."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Split": "Random" if "random" in r.get("split_type", "") else "Cluster",
            "AUROC": f"{r['auroc']:.4f} [{r['auroc_ci'][0]:.4f}-{r['auroc_ci'][1]:.4f}]",
            "AUPRC": f"{r['auprc']:.4f} [{r['auprc_ci'][0]:.4f}-{r['auprc_ci'][1]:.4f}]",
            "MCC": f"{r['mcc']:.4f} [{r['mcc_ci'][0]:.4f}-{r['mcc_ci'][1]:.4f}]",
            "TPR@1%FPR": f"{r['tpr_at_1pct_fpr']:.4f}",
            "Accuracy": f"{r['accuracy']:.4f}",
        })

    df = pd.DataFrame(rows)
    outpath = RESULTS_DIR / "summary_table.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved summary table to {outpath}")
    print("\n" + df.to_string(index=False))
    return df


def main():
    results = load_all_results()
    if not results:
        print("No results found. Run model training first.")
        return

    print(f"Loaded {len(results)} result sets")
    plot_roc_curves(results)
    plot_mcc_comparison(results)
    plot_confusion_matrices(results)
    summary = generate_summary_table(results)
    return summary


if __name__ == "__main__":
    main()
