"""
Extract ESM-2 650M embeddings using Modal A100.
Run with: modal run run_gpu_embeddings.py
"""

import modal
from pathlib import Path

app = modal.App("esm2-embeddings")

esm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.40", "accelerate", "numpy")
)


@app.function(image=esm_image, gpu="A100", timeout=3600)
def extract_batch(sequences: list[str], batch_size: int = 16) -> list[list[float]]:
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.extend(pooled.cpu().numpy().tolist())
        if (i // batch_size) % 20 == 0:
            print(f"  Processed {i + len(batch)}/{len(sequences)}")

    return all_embeddings


@app.local_entrypoint()
def main():
    import numpy as np
    import pandas as pd

    data_dir = Path("results/data")
    df = pd.read_csv(data_dir / "dataset.csv")
    sequences = df["sequence"].tolist()
    print(f"Loaded {len(sequences)} sequences, sending to Modal A100...")

    chunk_size = 2000
    all_embeddings = []
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i + chunk_size]
        print(f"Chunk {i // chunk_size + 1}: {len(chunk)} sequences")
        result = extract_batch.remote(chunk, batch_size=16)
        all_embeddings.extend(result)
        print(f"  Got {len(result)} embeddings, total: {len(all_embeddings)}")

    embeddings = np.array(all_embeddings)
    out_path = Path("results/esm2_650M_embeddings.npy")
    np.save(out_path, embeddings)
    print(f"\nSaved ESM-2 650M embeddings: {embeddings.shape} to {out_path}")
