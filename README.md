# ToxScreen

**Function-based protein hazard screening using protein language model embeddings.**

ToxScreen uses ESM-2 protein language model embeddings to detect hazardous proteins by predicted function rather than sequence similarity, catching AI-designed toxin variants that evade current DNA synthesis screening. Evaluated on 10,021 proteins with homology-controlled splits, it achieves 0.999 AUROC and 96.7% detection at 1% false positive rate on sequences sharing less than 40% identity with training data.

*AIxBio Hackathon 2026 | Track 1: DNA Screening & Synthesis Controls*

## Results

| Model | Evaluation | AUROC | MCC | TPR @ 1% FPR |
|-------|-----------|-------|-----|--------------|
| Random Forest (baseline) | Random split | 0.989 | 0.901 | 0.854 |
| Random Forest (baseline) | Cluster split | 0.977 | 0.856 | 0.846 |
| ESM-2 35M MLP | Cluster split | 0.996 | 0.953 | 0.957 |
| **ESM-2 650M MLP** | **Cluster split** | **0.999** | **0.970** | **0.967** |

Cluster split uses homology-controlled evaluation where no sequence above 40% identity is shared between train and test, simulating detection of genuinely novel threats.

## How It Works

```
Protein Sequence --> ESM-2 650M (frozen) --> 1280-d embedding --> MLP classifier --> Hazardous / Safe
```

1. Input protein sequence is tokenized and passed through the frozen ESM-2 650M model
2. Mean-pooled hidden states produce a 1,280-dimensional embedding
3. A 3-layer MLP (1280 -> 512 -> 64 -> 2) classifies the embedding as toxic or non-toxic
4. Output includes a confidence score for routing to expert review

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 1: Download and prepare dataset
python3 src/data/download_toxins.py
python3 src/data/prepare_dataset.py

# Step 2: Run baselines
python3 src/models/baseline.py

# Step 3: Extract ESM-2 embeddings and train MLP
PYTHONPATH=. python3 src/models/esm_classifier.py

# Step 4: (Optional) Run ESM-2 650M on Modal A100
modal run run_gpu_embeddings.py

# Step 5: Generate evaluation plots
PYTHONPATH=. python3 src/eval/evaluate.py

# Step 6: Launch dashboard
streamlit run app.py
```

## Project Structure

```
src/
  data/
    download_toxins.py      # Download sequences from UniProt + SafeProtein-Bench
    prepare_dataset.py      # Clean, cluster, split dataset
  models/
    baseline.py             # Physicochemical feature baselines (CPU)
    esm_classifier.py       # ESM-2 embedding extraction + MLP
    esm_finetune.py         # ESM-2 fine-tuning with classification head
  eval/
    evaluate.py             # Metrics, ROC curves, comparison plots
  modal_app.py              # Modal GPU app (deployed functions)
run_gpu_embeddings.py       # Script to run ESM-2 650M on Modal A100
app.py                      # Streamlit presentation dashboard
results/
  roc_curves.png            # ROC curves across models and splits
  mcc_comparison.png        # MCC generalization gap comparison
  confusion_matrices.png    # Confusion matrices for best model
  summary_table.csv         # All metrics with 95% bootstrap CIs
  report.md                 # Full research report
```

## Dataset

- **10,021 protein sequences** (4,957 toxins, 5,064 non-toxins)
- Toxins: UniProt KW-0800 (reviewed) + SafeProtein-Bench curated hazards
- Non-toxins: UniProt reviewed, length-matched, excluding viruses
- Clustered at 40% identity into 7,496 clusters
- Two evaluation splits: random (80/20) and cluster-aware (80/20 by cluster)

## Limitations and Dual-Use Considerations

### Limitations

- **False positives.** At the 1% FPR operating point, approximately 1 in 100 non-toxic sequences will be incorrectly flagged. In production, this requires a human review pipeline for flagged sequences. The false positive rate increases for proteins with unusual amino acid compositions or from underrepresented taxonomic groups.
- **False negatives.** While the model detects 96.7% of toxins at 1% FPR, 3.3% of hazardous sequences would pass undetected. No screening tool achieves perfect recall, which is why ToxScreen is designed as a complementary layer alongside existing tools (SecureDNA, IBBIS commec), not a standalone replacement.
- **Scope limited to toxins.** The current model is trained only on protein toxins. It does not detect virulence factors, pathogenicity islands, or multi-gene pathogenic systems. Extending to these more complex threat categories is necessary future work.
- **Clustering approximation.** We used k-mer-based clustering rather than full CD-HIT sequence alignment. While this provides a reasonable proxy for homology control, formal CD-HIT clustering at 40% identity would produce more rigorous evaluation splits.
- **No wet-lab validation.** All evaluation is computational. Confirming that low-identity sequences flagged by the model actually retain toxic function requires experimental validation.
- **Scalability.** ESM-2 650M embedding extraction requires GPU compute (we used Modal A100). Batch screening of thousands of sequences is feasible, but real-time single-sequence screening would benefit from model distillation or caching strategies.

### Dual-Use Risks

This work is defensive in nature: it improves the ability to detect hazardous proteins, including those designed to evade existing screening. However, we recognize the following dual-use considerations:

- **Adversarial information.** By demonstrating that function-based screening works, we implicitly confirm that sequence-similarity screening has exploitable gaps. This information is already public (Wittmann et al., Science 2025; Edison et al., Nature Communications 2026) and our work strengthens defense rather than offense.
- **Evasion feedback.** A deployed function-prediction model could theoretically be queried to test which variants evade detection, enabling iterative adversarial design. Mitigations include rate limiting, query logging, and not exposing raw confidence scores to end users.
- **Dataset sensitivity.** Our training data consists entirely of publicly available, reviewed protein sequences from UniProt and SafeProtein-Bench. We do not include novel hazard sequences, unpublished pathogen data, or detailed synthesis instructions.

### Responsible Disclosure

- We did not discover new vulnerabilities in existing screening tools. Our work builds on published findings (Wittmann et al., 2025) and strengthens the defensive response.
- The model and code are released under the MIT license to enable rapid adoption by the biosecurity community.
- We recommend that any production deployment include access controls, audit logging, and integration with established screening infrastructure rather than standalone use.

### Ethical Considerations

- All data is from public, reviewed databases. No novel hazardous sequences were generated during this project.
- The evaluation framework (homology-clustered splits) was designed to test real-world defensive utility, not to optimize for adversarial attack scenarios.
- We followed the Responsible AI x Biodesign principles throughout development.

### Suggestions for Future Improvement

1. **Red-team with actual AI-designed variants** using ProteinMPNN and RFdiffusion outputs as adversarial test cases
2. **Extend to virulence factors and pathogenic systems** beyond the toxin-only scope
3. **Full fine-tuning** of ESM-2 rather than frozen embeddings, potentially with LoRA for efficiency
4. **Deploy as a screening API** with rate limiting and integration hooks for SecureDNA/IBBIS commec
5. **Formal CD-HIT clustering** and evaluation at stricter identity thresholds (30%, 20%)
6. **Calibration optimization** to deliver well-calibrated probability scores at specific FPR budgets
7. **Model distillation** for real-time, on-device screening on benchtop synthesizers

## References

1. Abel Jr. et al. "Beyond Sequence Similarity: Toward Function-Based Screening of Nucleic Acid Synthesis." Frontiers in Bioengineering and Biotechnology, April 2026.
2. Wittmann et al. "Strengthening nucleic acid biosecurity screening against generative protein design tools." Science 390(6768): 82-87, October 2025.
3. Khan et al. "SafeBench-Seq." arXiv:2512.17527, December 2025.
4. Fan et al. "SafeProtein." arXiv:2509.03487, 2025.
5. Edison, Toner & Esvelt. "Assembling unregulated DNA segments bypasses synthesis screening." Nature Communications, January 2026.
6. Lin et al. ESM-2. Science 379(6637), 2023.

## License

MIT
