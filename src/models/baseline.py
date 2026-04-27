"""
Baseline classifiers using physicochemical features (SafeBench-Seq style).
No GPU required.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

CHARGE = {
    'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5,
}

MW = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.0, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
}


def compute_features(sequence: str) -> np.ndarray:
    """Compute physicochemical + composition features for a protein sequence."""
    n = len(sequence)
    if n == 0:
        return np.zeros(26)

    aa_counts = {aa: 0 for aa in AA_LIST}
    for c in sequence:
        if c in aa_counts:
            aa_counts[c] += 1
    composition = np.array([aa_counts[aa] / n for aa in AA_LIST])

    hydro = np.mean([HYDROPHOBICITY.get(c, 0) for c in sequence])
    charge = sum(CHARGE.get(c, 0) for c in sequence)
    net_charge_per_res = charge / n
    mw = sum(MW.get(c, 110) for c in sequence)
    log_length = np.log(n)
    aromatic_frac = sum(1 for c in sequence if c in "FWY") / n
    tiny_frac = sum(1 for c in sequence if c in "AGS") / n

    return np.concatenate([composition, [hydro, net_charge_per_res, log_length, mw / 1000, aromatic_frac, tiny_frac]])


def bootstrap_metric(y_true, y_score, metric_fn, n_boot=200, seed=42):
    """Compute metric with 95% bootstrap CI."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue
    if not scores:
        return 0, 0, 0
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    """TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return tpr[max(0, idx)]


def evaluate_model(y_true, y_probs, y_pred, model_name: str) -> dict:
    """Compute all evaluation metrics."""
    auroc_mean, auroc_lo, auroc_hi = bootstrap_metric(y_true, y_probs, roc_auc_score)
    auprc_mean, auprc_lo, auprc_hi = bootstrap_metric(y_true, y_probs, average_precision_score)
    mcc_mean, mcc_lo, mcc_hi = bootstrap_metric(y_true, y_pred, matthews_corrcoef)

    tpr_1 = tpr_at_fpr(y_true, y_probs, 0.01)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)

    return {
        "model": model_name,
        "auroc": auroc_mean,
        "auroc_ci": [auroc_lo, auroc_hi],
        "auprc": auprc_mean,
        "auprc_ci": [auprc_lo, auprc_hi],
        "mcc": mcc_mean,
        "mcc_ci": [mcc_lo, mcc_hi],
        "tpr_at_1pct_fpr": float(tpr_1),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr_arr.tolist(),
        "roc_tpr": tpr_arr.tolist(),
        "y_true": y_true.tolist(),
        "y_probs": y_probs.tolist(),
    }


def run_baselines():
    dataset_path = DATA_DIR / "dataset.csv"
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} sequences")

    print("Computing physicochemical features...")
    features = np.array([compute_features(seq) for seq in df["sequence"]])
    print(f"Feature matrix shape: {features.shape}")

    all_results = []

    for split_type in ["split_random", "split_cluster"]:
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_type}")
        print(f"{'='*60}")

        train_mask = df[split_type] == "train"
        test_mask = df[split_type] == "test"
        X_train, y_train = features[train_mask], df.loc[train_mask, "label"].values
        X_test, y_test = features[test_mask], df.loc[test_mask, "label"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "LogisticRegression": CalibratedClassifierCV(
                LogisticRegression(max_iter=1000, C=1.0), cv=5, method="isotonic"
            ),
            "RandomForest": CalibratedClassifierCV(
                RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
                cv=5, method="isotonic"
            ),
            "LinearSVM": CalibratedClassifierCV(
                LinearSVC(max_iter=5000, C=1.0), cv=5, method="sigmoid"
            ),
        }

        for name, model in models.items():
            print(f"\n  Training {name}...")
            model.fit(X_train_s, y_train)
            y_probs = model.predict_proba(X_test_s)[:, 1]
            y_pred = (y_probs > 0.5).astype(int)

            result = evaluate_model(y_test, y_probs, y_pred, f"baseline_{name}")
            result["split_type"] = split_type
            all_results.append(result)

            print(f"    AUROC: {result['auroc']:.4f} [{result['auroc_ci'][0]:.4f}, {result['auroc_ci'][1]:.4f}]")
            print(f"    MCC:   {result['mcc']:.4f} [{result['mcc_ci'][0]:.4f}, {result['mcc_ci'][1]:.4f}]")
            print(f"    TPR@1%FPR: {result['tpr_at_1pct_fpr']:.4f}")
            print(f"    Accuracy: {result['accuracy']:.4f}")

    outpath = RESULTS_DIR / "baseline_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved baseline results to {outpath}")
    return all_results


if __name__ == "__main__":
    run_baselines()
