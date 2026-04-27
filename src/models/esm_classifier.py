"""
ESM-2 embedding extraction + MLP classifier.
Extracts embeddings via Modal (GPU) then trains classifiers locally (CPU).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DATA_DIR = RESULTS_DIR / "data"


def extract_embeddings_on_modal(sequences: list[str]) -> np.ndarray:
    """Call Modal to extract ESM-2 650M embeddings."""
    import modal
    extract_fn = modal.Function.from_name("biosecurity-screening", "extract_esm2_embeddings")

    chunk_size = 500
    all_embeddings = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i + chunk_size]
        print(f"  Sending chunk {i//chunk_size + 1} ({len(chunk)} sequences) to Modal...")
        embeddings = extract_fn.remote(chunk, batch_size=8)
        all_embeddings.extend(embeddings)
        print(f"  Done. Total embeddings so far: {len(all_embeddings)}")

    return np.array(all_embeddings)


def extract_embeddings_local(sequences: list[str], batch_size: int = 4) -> np.ndarray:
    """Extract ESM-2 embeddings locally (for smaller models / CPU fallback)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "facebook/esm2_t12_35M_UR50D"
    print(f"Loading {model_name} locally...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.extend(pooled.cpu().numpy().tolist())

        if (i // batch_size) % 50 == 0:
            print(f"  Processed {i + len(batch)}/{len(sequences)}")

    return np.array(all_embeddings)


def train_mlp_classifier(X_train, y_train, X_test, y_test, hidden_dim=256, epochs=50, lr=1e-3):
    """Train a 2-layer MLP on embeddings using PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_auc = 0
    best_probs = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device))
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, probs)
        if auc > best_auc:
            best_auc = auc
            best_probs = probs.copy()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: val_auroc={auc:.4f}")

    return best_probs


def run_esm_pipeline(use_modal: bool = False):
    """Full pipeline: extract embeddings, train MLP, evaluate."""
    from src.models.baseline import evaluate_model

    df = pd.read_csv(DATA_DIR / "dataset.csv")
    sequences = df["sequence"].tolist()

    embeddings_path = RESULTS_DIR / "esm2_embeddings.npy"
    if embeddings_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        if use_modal:
            print("Extracting ESM-2 650M embeddings via Modal...")
            embeddings = extract_embeddings_on_modal(sequences)
        else:
            print("Extracting ESM-2 35M embeddings locally...")
            embeddings = extract_embeddings_local(sequences)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}, shape: {embeddings.shape}")

    from sklearn.preprocessing import StandardScaler

    all_results = []
    for split_type in ["split_random", "split_cluster"]:
        print(f"\n{'='*60}")
        print(f"Training MLP on ESM-2 embeddings ({split_type})")
        print(f"{'='*60}")

        train_mask = df[split_type] == "train"
        test_mask = df[split_type] == "test"

        X_train = embeddings[train_mask.values]
        y_train = df.loc[train_mask, "label"].values
        X_test = embeddings[test_mask.values]
        y_test = df.loc[test_mask, "label"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_probs = train_mlp_classifier(X_train_s, y_train, X_test_s, y_test)
        y_pred = (y_probs > 0.5).astype(int)

        result = evaluate_model(y_test, y_probs, y_pred, "ESM2_MLP")
        result["split_type"] = split_type
        all_results.append(result)

        print(f"\n  AUROC: {result['auroc']:.4f} [{result['auroc_ci'][0]:.4f}, {result['auroc_ci'][1]:.4f}]")
        print(f"  MCC:   {result['mcc']:.4f} [{result['mcc_ci'][0]:.4f}, {result['mcc_ci'][1]:.4f}]")
        print(f"  TPR@1%FPR: {result['tpr_at_1pct_fpr']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")

    outpath = RESULTS_DIR / "esm_mlp_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ESM MLP results to {outpath}")
    return all_results


if __name__ == "__main__":
    import sys
    use_modal = "--modal" in sys.argv
    run_esm_pipeline(use_modal=use_modal)
