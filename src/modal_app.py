"""
Modal app for GPU-accelerated ESM-2 embedding extraction and fine-tuning.
"""

import modal

app = modal.App("biosecurity-screening")

esm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "accelerate",
        "scikit-learn",
        "numpy",
        "pandas",
        "peft",
    )
)


@app.function(
    image=esm_image,
    gpu="A100",
    timeout=3600,
)
def extract_esm2_embeddings(sequences: list[str], batch_size: int = 8) -> list[list[float]]:
    """Extract mean-pooled embeddings from ESM-2 650M for a batch of sequences."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.extend(pooled.cpu().numpy().tolist())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(sequences)} sequences")

    return all_embeddings


@app.function(
    image=esm_image,
    gpu="A100",
    timeout=7200,
)
def finetune_esm2(
    train_sequences: list[str],
    train_labels: list[int],
    val_sequences: list[str],
    val_labels: list[int],
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
) -> dict:
    """Fine-tune ESM-2 150M with a classification head for toxin detection."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, EsmForSequenceClassification
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
    import numpy as np

    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to("cuda")

    class SeqDataset(Dataset):
        def __init__(self, sequences, labels, tokenizer, max_len=1024):
            self.sequences = sequences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.sequences[idx],
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_ds = SeqDataset(train_sequences, train_labels, tokenizer)
    val_ds = SeqDataset(val_sequences, val_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_auc = 0.0
    best_results = {}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)

        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, preds)
        acc = accuracy_score(all_labels, preds)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} val_auc={auc:.4f} val_mcc={mcc:.4f} val_acc={acc:.4f}")

        if auc > best_val_auc:
            best_val_auc = auc
            best_results = {
                "epoch": epoch + 1,
                "val_auroc": float(auc),
                "val_mcc": float(mcc),
                "val_accuracy": float(acc),
                "val_probs": all_probs.tolist(),
                "val_labels": all_labels.tolist(),
            }

    return best_results
