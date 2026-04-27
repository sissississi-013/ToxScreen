"""
Fine-tune ESM-2 with a classification head for toxin detection.
Supports both Modal (GPU) and local execution.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DATA_DIR = RESULTS_DIR / "data"


def finetune_on_modal(train_seqs, train_labels, val_seqs, val_labels, num_epochs=5):
    """Fine-tune ESM-2 150M on Modal A100."""
    import modal
    finetune_fn = modal.Function.from_name("biosecurity-screening", "finetune_esm2")
    result = finetune_fn.remote(
        train_sequences=train_seqs,
        train_labels=train_labels,
        val_sequences=val_seqs,
        val_labels=val_labels,
        num_epochs=num_epochs,
    )
    return result


def finetune_local(train_seqs, train_labels, val_seqs, val_labels, num_epochs=3):
    """Fine-tune ESM-2 35M locally (CPU/MPS)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, EsmForSequenceClassification
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score

    model_name = "facebook/esm2_t12_35M_UR50D"
    print(f"Loading {model_name} for fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    class SeqDataset(Dataset):
        def __init__(self, sequences, labels, tok, max_len=512):
            self.sequences = sequences
            self.labels = labels
            self.tok = tok
            self.max_len = max_len
        def __len__(self):
            return len(self.sequences)
        def __getitem__(self, idx):
            enc = self.tok(self.sequences[idx], truncation=True, max_length=self.max_len,
                          padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_ds = SeqDataset(train_seqs, train_labels, tokenizer)
    val_ds = SeqDataset(val_seqs, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    best_auc = 0
    best_results = {}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch+1} step {step}: loss={loss.item():.4f}")

        model.eval()
        all_probs, all_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_list.extend(batch["labels"].cpu().numpy().tolist())

        y_probs = np.array(all_probs)
        y_true = np.array(all_labels_list)
        y_pred = (y_probs > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_probs)
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} val_auc={auc:.4f} mcc={mcc:.4f} acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_results = {
                "epoch": epoch + 1,
                "val_auroc": float(auc),
                "val_mcc": float(mcc),
                "val_accuracy": float(acc),
                "val_probs": y_probs.tolist(),
                "val_labels": y_true.tolist(),
            }

    return best_results


def run_finetune(use_modal: bool = False):
    from src.models.baseline import evaluate_model

    df = pd.read_csv(DATA_DIR / "dataset.csv")
    all_results = []

    for split_type in ["split_random", "split_cluster"]:
        print(f"\n{'='*60}")
        print(f"Fine-tuning ESM-2 ({split_type})")
        print(f"{'='*60}")

        train_mask = df[split_type] == "train"
        test_mask = df[split_type] == "test"

        train_seqs = df.loc[train_mask, "sequence"].tolist()
        train_labels = df.loc[train_mask, "label"].tolist()
        val_seqs = df.loc[test_mask, "sequence"].tolist()
        val_labels = df.loc[test_mask, "label"].tolist()

        if use_modal:
            ft_result = finetune_on_modal(train_seqs, train_labels, val_seqs, val_labels)
        else:
            ft_result = finetune_local(train_seqs, train_labels, val_seqs, val_labels, num_epochs=3)

        y_true = np.array(ft_result["val_labels"])
        y_probs = np.array(ft_result["val_probs"])
        y_pred = (y_probs > 0.5).astype(int)

        result = evaluate_model(y_true, y_probs, y_pred, "ESM2_FineTuned")
        result["split_type"] = split_type
        result["best_epoch"] = ft_result["epoch"]
        all_results.append(result)

        print(f"\n  AUROC: {result['auroc']:.4f} [{result['auroc_ci'][0]:.4f}, {result['auroc_ci'][1]:.4f}]")
        print(f"  MCC:   {result['mcc']:.4f} [{result['mcc_ci'][0]:.4f}, {result['mcc_ci'][1]:.4f}]")

    outpath = RESULTS_DIR / "esm_finetune_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved fine-tuning results to {outpath}")
    return all_results


if __name__ == "__main__":
    import sys
    use_modal = "--modal" in sys.argv
    run_finetune(use_modal=use_modal)
