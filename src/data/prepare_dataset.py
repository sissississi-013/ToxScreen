"""
Prepare the dataset: cluster with CD-HIT at 40% identity,
create random and cluster-aware train/test splits.
"""

import subprocess
import shutil
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "data"


def run_cdhit(fasta_path: Path, output_path: Path, identity: float = 0.4):
    """Run CD-HIT to cluster sequences at the given identity threshold."""
    cdhit_bin = shutil.which("cd-hit")
    if cdhit_bin is None:
        print("CD-HIT not found. Installing via conda or using fallback clustering...")
        return fallback_cluster(fasta_path, identity)

    cmd = [
        cdhit_bin,
        "-i", str(fasta_path),
        "-o", str(output_path),
        "-c", str(identity),
        "-n", "2",        # word size for 40% identity
        "-M", "4000",     # memory limit MB
        "-T", "4",        # threads
        "-d", "0",        # full header in .clstr
    ]
    print(f"Running CD-HIT: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    cluster_map = parse_cdhit_clusters(Path(str(output_path) + ".clstr"))
    return cluster_map


def parse_cdhit_clusters(clstr_path: Path) -> dict[str, int]:
    """Parse CD-HIT .clstr file to map accession -> cluster_id."""
    cluster_map = {}
    current_cluster = -1
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = int(line.strip().split()[-1])
            else:
                match = re.search(r">(\S+)\.\.\.", line)
                if match:
                    cluster_map[match.group(1)] = current_cluster
    return cluster_map


def fallback_cluster(fasta_path: Path, identity: float) -> dict[str, int]:
    """
    Simple fallback: assign random cluster IDs based on sequence hashing.
    Not as rigorous as CD-HIT but allows the pipeline to proceed.
    """
    print("Using hash-based fallback clustering (install CD-HIT for proper clustering)")
    cluster_map = {}
    cluster_id = 0
    representatives = []

    sequences = {}
    with open(fasta_path) as f:
        current_acc = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_acc:
                    sequences[current_acc] = "".join(current_seq)
                current_acc = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_acc:
            sequences[current_acc] = "".join(current_seq)

    sorted_accs = sorted(sequences.keys(), key=lambda a: len(sequences[a]), reverse=True)
    for acc in sorted_accs:
        seq = sequences[acc]
        assigned = False
        for rep_acc, rep_cluster in representatives[-50:]:
            rep_seq = sequences[rep_acc]
            if _rough_identity(seq, rep_seq) > identity:
                cluster_map[acc] = rep_cluster
                assigned = True
                break
        if not assigned:
            cluster_map[acc] = cluster_id
            representatives.append((acc, cluster_id))
            cluster_id += 1

    return cluster_map


def _rough_identity(s1: str, s2: str) -> float:
    """Rough k-mer-based identity estimate (faster than alignment)."""
    k = 3
    if len(s1) < k or len(s2) < k:
        return 0.0
    kmers1 = set(s1[i:i+k] for i in range(len(s1)-k+1))
    kmers2 = set(s2[i:i+k] for i in range(len(s2)-k+1))
    if not kmers1 or not kmers2:
        return 0.0
    jaccard = len(kmers1 & kmers2) / len(kmers1 | kmers2)
    return jaccard


def create_splits(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Create random and cluster-aware train/test splits."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_frac, random_state=seed, stratify=df["label"]
    )
    df["split_random"] = "train"
    df.loc[test_idx, "split_random"] = "test"

    clusters = df["cluster_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(clusters)
    n_test_clusters = max(1, int(len(clusters) * test_frac))
    test_clusters = set(clusters[:n_test_clusters])

    df["split_cluster"] = df["cluster_id"].apply(
        lambda c: "test" if c in test_clusters else "train"
    )

    return df


def main():
    raw_path = DATA_DIR / "raw_sequences.csv"
    if not raw_path.exists():
        print(f"Raw sequences not found at {raw_path}. Run download_toxins.py first.")
        return

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} sequences ({df['label'].sum()} toxins)")

    fasta_path = DATA_DIR / "all_sequences.fasta"
    if not fasta_path.exists():
        with open(fasta_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['accession']}\n{row['sequence']}\n")

    cdhit_out = DATA_DIR / "cdhit_out"
    cluster_map = run_cdhit(fasta_path, cdhit_out)

    df["cluster_id"] = df["accession"].map(cluster_map).fillna(-1).astype(int)
    unassigned = (df["cluster_id"] == -1).sum()
    if unassigned > 0:
        max_cluster = df["cluster_id"].max()
        for idx in df[df["cluster_id"] == -1].index:
            max_cluster += 1
            df.at[idx, "cluster_id"] = max_cluster

    n_clusters = df["cluster_id"].nunique()
    print(f"Assigned {len(df)} sequences to {n_clusters} clusters")

    df = create_splits(df)

    for split_type in ["split_random", "split_cluster"]:
        train_mask = df[split_type] == "train"
        test_mask = df[split_type] == "test"
        train_pos = df.loc[train_mask, "label"].sum()
        test_pos = df.loc[test_mask, "label"].sum()
        print(f"\n{split_type}:")
        print(f"  Train: {train_mask.sum()} ({train_pos} toxins)")
        print(f"  Test:  {test_mask.sum()} ({test_pos} toxins)")

    outpath = DATA_DIR / "dataset.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved prepared dataset to {outpath}")
    return df


if __name__ == "__main__":
    main()
