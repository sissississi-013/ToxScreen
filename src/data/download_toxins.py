"""
Download toxin and non-toxin protein sequences from UniProt REST API
and SafeProtein-Bench for biosecurity screening classification.
"""

import json
import re
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_API = "https://rest.uniprot.org"
CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 30, 1024


def fetch_uniprot_sequences(query: str, max_results: int = 5000) -> list[dict]:
    """Stream sequences from UniProt REST API with pagination."""
    url = f"{UNIPROT_API}/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,sequence,organism_name,lineage,length",
        "size": 500,
    }
    results = []
    page = 0
    while len(results) < max_results:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()

        lines = resp.text.strip().split("\n")
        if page == 0:
            header = lines[0].split("\t")
            data_lines = lines[1:]
        else:
            data_lines = lines[1:] if lines[0].startswith("Entry") else lines

        for line in data_lines:
            parts = line.split("\t")
            if len(parts) >= 5:
                results.append({
                    "accession": parts[0],
                    "sequence": parts[1],
                    "organism": parts[2],
                    "lineage": parts[3],
                    "length": int(parts[4]) if parts[4].isdigit() else len(parts[1]),
                })

        link_header = resp.headers.get("Link", "")
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        if not next_match:
            break
        url = next_match.group(1)
        params = {}
        page += 1
        time.sleep(0.5)

    return results[:max_results]


def filter_sequences(records: list[dict]) -> list[dict]:
    """Filter to canonical AAs, length range, no fragments."""
    filtered = []
    for r in records:
        seq = r["sequence"]
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        if not all(c in CANONICAL_AAS for c in seq):
            continue
        filtered.append(r)
    return filtered


def download_safeprotein_bench() -> list[dict]:
    """Download SafeProtein-Bench hazard dataset from GitHub."""
    url = "https://raw.githubusercontent.com/jigang-fan/SafeProtein/main/SafeProtein_Bench.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    bench = json.loads(resp.text)

    results = []
    for accession, entry in bench.items():
        seq = entry.get("Sequence", entry.get("sequence", ""))
        if not seq or not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        if not all(c in CANONICAL_AAS for c in seq):
            continue
        results.append({
            "accession": accession,
            "sequence": seq,
            "organism": "unknown",
            "lineage": "SafeProtein-Bench",
            "length": len(seq),
            "source": "safeprotein_bench",
        })
    return results


def download_toxins() -> pd.DataFrame:
    """Download reviewed toxin sequences from UniProt KW-0800."""
    print("Downloading toxin sequences from UniProt (KW-0800, reviewed)...")
    query = "(keyword:KW-0800) AND (reviewed:true) AND (length:[30 TO 1024])"
    records = fetch_uniprot_sequences(query, max_results=5000)
    records = filter_sequences(records)
    for r in records:
        r["label"] = 1
        r["source"] = "uniprot_toxin"
    print(f"  Got {len(records)} toxin sequences from UniProt")

    print("Downloading SafeProtein-Bench hazard sequences...")
    sp_records = download_safeprotein_bench()
    existing_seqs = {r["sequence"] for r in records}
    added = 0
    for r in sp_records:
        if r["sequence"] not in existing_seqs:
            r["label"] = 1
            records.append(r)
            existing_seqs.add(r["sequence"])
            added += 1
    print(f"  Added {added} unique sequences from SafeProtein-Bench")
    print(f"  Total toxins: {len(records)}")

    return pd.DataFrame(records)


def download_non_toxins(n_target: int, length_dist: pd.Series) -> pd.DataFrame:
    """Download non-toxin sequences, roughly length-matched."""
    print(f"Downloading {n_target} non-toxin sequences from UniProt...")
    median_len = int(length_dist.median())
    low = max(MIN_LEN, median_len - 200)
    high = min(MAX_LEN, median_len + 200)

    query = (
        f"(NOT keyword:KW-0800) AND (reviewed:true) "
        f"AND (length:[{low} TO {high}]) "
        f"AND (NOT taxonomy_id:10239)"  # exclude viruses
    )
    records = fetch_uniprot_sequences(query, max_results=n_target + 1000)
    records = filter_sequences(records)

    existing_seqs = set()
    unique = []
    for r in records:
        if r["sequence"] not in existing_seqs:
            r["label"] = 0
            r["source"] = "uniprot_nontoxin"
            unique.append(r)
            existing_seqs.add(r["sequence"])

    if len(unique) > n_target:
        unique = unique[:n_target]

    print(f"  Got {len(unique)} non-toxin sequences")
    return pd.DataFrame(unique)


def main():
    toxins_df = download_toxins()
    nontoxins_df = download_non_toxins(
        n_target=len(toxins_df),
        length_dist=toxins_df["length"],
    )

    df = pd.concat([toxins_df, nontoxins_df], ignore_index=True)
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    outpath = DATA_DIR / "raw_sequences.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(df)} sequences ({df['label'].sum()} toxins, {(df['label']==0).sum()} non-toxins) to {outpath}")

    fasta_path = DATA_DIR / "all_sequences.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['accession']}\n{row['sequence']}\n")
    print(f"Saved FASTA to {fasta_path}")

    return df


if __name__ == "__main__":
    main()
