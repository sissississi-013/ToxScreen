# ToxScreen: Function-Based Protein Hazard Screening Using Protein Language Model Embeddings

**Sissi Wang**
*Independent*

With Apart Research

---

## Abstract

DNA synthesis screening currently identifies sequences of concern through sequence similarity to known pathogens and toxins. AI protein design tools can generate functional variants with low sequence identity that evade this paradigm entirely. We present ToxScreen, a function-prediction screening prototype that classifies protein sequences as hazardous using learned representations from the ESM-2 protein language model, evaluated under homology-clustered splits that simulate detection of genuinely novel threats. On a balanced dataset of 10,021 protein sequences (4,957 toxins sourced from UniProt KW-0800 and SafeProtein-Bench, 5,064 length-matched non-toxins), our ESM-2 650M embedding classifier achieves AUROC 0.999 and TPR 96.7% at 1% FPR under cluster-split evaluation, where no test sequence shares above 40% identity with any training sequence. The best physicochemical baseline (Random Forest on amino acid composition) reaches only 0.977 AUROC and 84.6% TPR at the same operating point. The generalization gap between random and cluster splits is 0.0005 for ESM-2 versus 0.012 for baselines, confirming that protein language model embeddings encode functional signals that persist well below the threshold where BLAST-based screening becomes unreliable. This work implements the function-based screening vision articulated in "Beyond Sequence Similarity" (Abel Jr. et al., Frontiers in Bioengineering and Biotechnology, April 2026) for the most tractable target class: protein toxins.

---

## 1. Introduction

DNA synthesis screening is the primary chokepoint for preventing misuse of synthetic biology. Providers screen orders against databases of known hazardous sequences using BLAST-based homology search (IBBIS Common Mechanism) or exact-match with predicted functional variants (SecureDNA). These approaches work well for natural sequences but face a fundamental challenge: AI protein design tools can now generate proteins that fold into the same 3D structures and perform the same biological functions as dangerous proteins while having entirely different amino acid sequences.

The Microsoft Paraphrase Project (Wittmann et al., Science 2025) demonstrated this concretely: using open-source tools (ProteinMPNN, EvoDiff), they generated 76,089 variants of 72 proteins of concern, and traditional screening tools missed many of these redesigned sequences. As AI protein design improves, sequence-similarity screening will become increasingly unreliable. The Frontiers perspective "Beyond Sequence Similarity" (Abel Jr. et al., April 2026), authored by a consortium including SecureDNA, IBBIS, Microsoft, NIST, and Fourth Eon Bio, calls for function-based screening starting with toxins as the most tractable target.

This project implements that vision. Our main contributions are:

- **A function-prediction screening prototype (ToxScreen)** that uses ESM-2 650M protein language model embeddings to classify toxins by predicted function, achieving 0.999 AUROC on a dataset of 10,021 sequences.
- **Homology-controlled evaluation** using cluster splits at 40% identity, demonstrating that the approach generalizes to sequences with no close homolog in training data (generalization gap of only 0.0005 AUROC).
- **Quantitative comparison** showing ESM-2 embeddings reduce missed toxins by 78% versus the best physicochemical baseline at the biosecurity-relevant 1% FPR operating point.

## 2. Related Work

**Sequence-similarity screening.** SecureDNA uses exact-match search with predicted functional variants and cryptographic privacy (DOPRF protocol), screening down to 30bp with near-zero false positives [1]. IBBIS Common Mechanism uses HMM-based biorisk scanning plus BLAST taxonomy search, with best performance above 50bp [2]. Both rely on sequence similarity to known hazards.

**The AI evasion problem.** Wittmann et al. (2025) showed AI-designed protein variants evade sequence-based screening [3]. Edison, Toner & Esvelt (2026) demonstrated that unregulated DNA fragments from 38 providers could be assembled into 1918 influenza for approximately $3,000 [4]. SafeProtein found up to 70% jailbreak success rates on ESM3 [5].

**Function-prediction approaches.** Abel Jr. et al. (2026) proposed a roadmap for function-based screening, starting with toxins [6]. SafeBench-Seq provided a CPU-only baseline using physicochemical features with homology-clustered evaluation [7]. BioLMTox-2 fine-tuned ESM-2 650M for toxin classification with 0.964 accuracy, but evaluated on random splits only, leaving generalization to novel threats unquantified [8].

**How ToxScreen differs.** Unlike BioLMTox-2 (random-split evaluation only), we evaluate under homology-clustered splits that simulate genuinely novel threats. Unlike SafeBench-Seq (physicochemical features only), we use protein language model embeddings. ToxScreen bridges these approaches: ESM-2 representations evaluated under biosecurity-relevant conditions.

## 3. Methods

### 3.1 Dataset Construction

**Positive class (toxins).** We downloaded 4,944 reviewed toxin sequences from UniProt using keyword KW-0800, excluding viruses and archaea, filtered to 30-1,024 amino acids with canonical residues only. We added 120 unique sequences from SafeProtein-Bench [5], a curated set of 429 experimentally resolved hazardous proteins. Total: 4,957 toxins.

**Negative class (non-toxins).** We downloaded 5,064 reviewed non-toxic protein sequences from UniProt, length-matched to the toxin distribution, excluding viral proteins to prevent taxonomic shortcuts.

**Total dataset:** 10,021 sequences (4,957 toxins, 5,064 non-toxins).

### 3.2 Homology-Clustered Evaluation

Following the methodology of SafeBench-Seq [7], we clustered all sequences at 40% identity using k-mer Jaccard distance as a computationally efficient proxy for CD-HIT, producing 7,496 clusters. We then created two evaluation splits to quantify whether models learn generalizable functional signals or merely memorize sequence patterns:

- **Random split (80/20):** Standard evaluation baseline. Homologous sequences may appear in both train and test, allowing the model to exploit sequence similarity.
- **Cluster split (80/20 by cluster):** Entire clusters are assigned to either train or test. No test sequence shares above 40% identity with any training sequence, simulating detection of a genuinely novel AI-designed toxin variant with no close homolog in the screening database.

The difference in performance between these two splits is our primary diagnostic for generalization. A model that memorizes sequence patterns will show a large drop from random to cluster evaluation; a model that captures functional signals will maintain performance.

### 3.3 Models

We compared three model tiers of increasing representational capacity:

**Tier 1: Physicochemical baselines.** A 26-dimensional feature vector per sequence comprising amino acid composition (20 features), mean Kyte-Doolittle hydrophobicity, net charge per residue at pH 7, log sequence length, estimated molecular weight, aromatic residue fraction (F, W, Y), and tiny residue fraction (A, G, S). We trained Logistic Regression (C=1.0), Random Forest (200 estimators, max depth 15), and calibrated Linear SVM (C=1.0) using CalibratedClassifierCV with 5-fold cross-validation and isotonic/sigmoid calibration.

**Tier 2: ESM-2 35M Embeddings + MLP.** We extracted 480-dimensional mean-pooled embeddings from the frozen ESM-2 35M model (facebook/esm2_t12_35M_UR50D) for all 10,021 sequences. These embeddings served as input to a 3-layer MLP (480 -> 256 -> 64 -> 2) with ReLU activations, dropout (0.3 and 0.2), trained for 50 epochs with the Adam optimizer (lr=1e-3, weight decay 1e-4) and cross-entropy loss.

**Tier 3: ESM-2 650M Embeddings + MLP.** We extracted 1,280-dimensional mean-pooled embeddings from the frozen ESM-2 650M model (facebook/esm2_t33_650M_UR50D) on a Modal A100 GPU for all 10,021 sequences. The classifier architecture mirrored Tier 2 but with a wider first layer (1280 -> 512 -> 64 -> 2), trained for 80 epochs. This model tests whether richer protein representations improve generalization to novel threats.

### 3.4 Metrics

All metrics computed with 200-iteration bootstrap 95% confidence intervals: AUROC, AUPRC, Matthews Correlation Coefficient (MCC), TPR at 1% FPR, and accuracy.

## 4. Results

### Table 1: Full evaluation results across models and split types.

| Model | Split | AUROC [95% CI] | MCC [95% CI] | TPR@1%FPR |
|-------|-------|----------------|--------------|-----------|
| Logistic Regression | Random | 0.948 [0.939-0.957] | 0.773 [0.750-0.798] | 0.330 |
| Random Forest | Random | 0.989 [0.985-0.992] | 0.901 [0.884-0.916] | 0.854 |
| Linear SVM | Random | 0.947 [0.938-0.956] | 0.776 [0.750-0.802] | 0.323 |
| ESM-2 35M MLP | Random | 0.997 [0.996-0.999] | 0.956 [0.943-0.968] | 0.968 |
| **ESM-2 650M MLP** | **Random** | **0.999 [0.999-1.000]** | **0.964 [0.952-0.975]** | **0.978** |
| Logistic Regression | Cluster | 0.947 [0.939-0.956] | 0.779 [0.753-0.806] | 0.471 |
| Random Forest | Cluster | 0.977 [0.972-0.983] | 0.856 [0.834-0.880] | 0.846 |
| Linear SVM | Cluster | 0.947 [0.939-0.956] | 0.767 [0.742-0.795] | 0.488 |
| ESM-2 35M MLP | Cluster | 0.996 [0.993-0.997] | 0.953 [0.940-0.965] | 0.957 |
| **ESM-2 650M MLP** | **Cluster** | **0.999 [0.998-0.999]** | **0.970 [0.959-0.980]** | **0.967** |

### Key Findings

**ESM-2 650M embeddings dramatically outperform physicochemical baselines.** On the cluster split, ESM-2 650M achieves AUROC 0.999 vs. 0.977 for Random Forest. At the 1% FPR operating point, ESM-2 detects 96.7% of toxins compared to 84.6% for Random Forest, a 78% reduction in missed hazards.

**ESM-2 embeddings generalize to novel sequences.** The generalization gap (random minus cluster AUROC) is only 0.0005 for ESM-2 650M versus 0.012 for Random Forest, demonstrating that protein language model embeddings capture functional signals that persist below 40% sequence identity.

**Scaling improves results.** Moving from ESM-2 35M (480-dim) to 650M (1280-dim) improved cluster-split MCC from 0.953 to 0.970 and TPR@1%FPR from 95.7% to 96.7%, with the generalization gap shrinking from 0.001 to 0.0005.

## 5. Discussion and Limitations

### Implications for Biosecurity Infrastructure

These results carry direct implications for the DNA synthesis screening ecosystem. At the 1% FPR operating point, ToxScreen generates one false alarm per 100 non-toxic sequences while catching 96.7% of toxins. This operating characteristic is viable for deployment as a secondary screening layer: sequences that pass BLAST-based screening (SecureDNA [1], IBBIS commec [2]) but trigger the function-prediction model would be routed to expert review, adding defense in depth against AI-designed evasion.

The approach also addresses a concrete legislative need. Section 4(b)(3) of the Biosecurity Modernization and Innovation Act (S.3741, 2026) directs NIST to "research and prototype sequence-to-function models to supplement" homology-based screening [10]. ToxScreen demonstrates this is achievable with current models and infrastructure.

### Limitations

- Our clustering uses k-mer-based approximation rather than full CD-HIT alignment; formal CD-HIT clustering would strengthen the homology control.
- The dataset focuses on protein toxins only; extending to virulence factors and pathogenic systems is needed.
- Evaluation is purely computational; wet-lab validation of predicted functional equivalence is needed before deployment.
- The 3.3% miss rate at 1% FPR means some hazardous sequences would pass undetected; no single screening tool should be relied on alone.

### Future Work

1. Full fine-tuning of ESM-2 (rather than frozen embeddings) with LoRA adapters
2. Extend to virulence factors and multi-gene pathogenic systems
3. Red-team evaluation with AI-designed variants from ProteinMPNN and RFdiffusion
4. Integration with SecureDNA or commec as a deployed screening plugin
5. Model distillation for real-time on-device screening on benchtop synthesizers

## 6. Conclusion

ToxScreen demonstrates that protein language model embeddings enable function-based hazard screening that generalizes to sequences below 40% identity to any training example. This directly addresses the critical gap identified by the field's leading researchers: the inability of sequence-similarity tools to detect AI-designed functional variants. Built during a 3-day hackathon and scaled to ESM-2 650M on Modal A100 GPUs, ToxScreen achieves 0.999 AUROC under stringent homology-controlled evaluation with a generalization gap of only 0.0005. Deploying function-prediction screening as a complement to existing infrastructure is both technically feasible and urgently needed.

## Code and Data

- **Code repository:** [https://github.com/sissississi-013/ToxScreen](https://github.com/sissississi-013/ToxScreen)
- **Data/Datasets:** Constructed from public sources (UniProt KW-0800, SafeProtein-Bench). Download scripts included in repo.
- **Dashboard:** Interactive Streamlit dashboard included in repo (`app.py`)

## Author Contributions

S.W. conceived the project, designed the evaluation methodology, implemented all code, ran experiments, and wrote the report.

## References

[1] SecureDNA. "Exact-match search with functional variant prediction enables automated DNA screening." bioRxiv, 2024.

[2] IBBIS. Common Mechanism for DNA Synthesis Screening. https://ibbis.bio/our-work/common-mechanism/

[3] Wittmann et al. "Strengthening nucleic acid biosecurity screening against generative protein design tools." Science 390(6768): 82-87, 2025.

[4] Edison, Toner & Esvelt. "Assembling unregulated DNA segments bypasses synthesis screening." Nature Communications, 2026.

[5] Fan et al. "SafeProtein: Red-Teaming Framework and Benchmark for Protein Foundation Models." arXiv:2509.03487, 2025.

[6] Abel Jr. et al. "Beyond Sequence Similarity: Toward Function-Based Screening of Nucleic Acid Synthesis." Frontiers in Bioengineering and Biotechnology, 2026.

[7] Khan et al. "SafeBench-Seq: A Homology-Clustered, CPU-Only Baseline for Protein Hazard Screening." arXiv:2512.17527, 2025.

[8] BioLMTox-2. BioLM, 2024. https://biolm.ai/models/biolmtox2/

[9] Lin et al. "Language models of protein sequences at the scale of evolution enable accurate structure prediction." Science 379(6637), 2023.

[10] Biosecurity Modernization and Innovation Act of 2026 (S.3741).

[11] OSTP Framework for Nucleic Acid Synthesis Screening, April 2024.

[12] Kim. "AI Can Already Evade DNA Synthesis Screening. Congress's New Bill Doesn't Address That." The Counterfactual, March 2026.

## Appendix: Limitations and Dual-Use Considerations

### Limitations

- **False positives.** At 1% FPR, approximately 1 in 100 non-toxic sequences is incorrectly flagged. Production deployment requires a human review pipeline. The false positive rate may increase for proteins with unusual compositions or from underrepresented taxonomic groups.
- **False negatives.** 3.3% of hazardous sequences pass undetected at the 1% FPR operating point. ToxScreen is designed as a complementary layer alongside SecureDNA and IBBIS commec, not a standalone replacement.
- **Scope.** The model detects protein toxins only. It does not cover virulence factors, pathogenicity islands, or multi-gene systems.
- **Clustering approximation.** We used k-mer-based clustering rather than full CD-HIT alignment. Formal CD-HIT at 40% identity would produce more rigorous splits.
- **Scalability.** ESM-2 650M embedding extraction requires GPU compute. Real-time screening would benefit from model distillation.

### Dual-Use Risks

This work is defensive: it improves detection of hazardous proteins. However:

- **Adversarial information.** Demonstrating that function-based screening works implicitly confirms sequence-similarity screening has gaps. This is already public knowledge (Wittmann et al., 2025; Edison et al., 2026). Our work strengthens defense.
- **Evasion feedback.** A deployed model could theoretically be queried to test which variants evade detection. Mitigations include rate limiting, query logging, and withholding raw confidence scores from end users.
- **Dataset sensitivity.** All training data is from publicly available, reviewed databases. No novel hazard sequences were generated.

### Responsible Disclosure

We did not discover new vulnerabilities in existing screening tools. Our work builds on published findings and strengthens the defensive response. We recommend any production deployment include access controls, audit logging, and integration with established screening infrastructure.

### Ethical Considerations

All data is from public, reviewed databases. No novel hazardous sequences were generated. The evaluation framework tests defensive utility, not adversarial attack scenarios. We followed Responsible AI x Biodesign principles throughout.

### Suggestions for Future Improvements

1. Red-team with AI-designed variants from ProteinMPNN and RFdiffusion
2. Extend to virulence factors and multi-gene pathogenic systems
3. Full fine-tuning with LoRA adapters for stronger performance
4. Deploy as a screening API with SecureDNA/commec integration hooks
5. Formal CD-HIT clustering at stricter identity thresholds (30%, 20%)
6. Model distillation for on-device benchtop synthesizer screening

## LLM Usage Statement

We used Claude to assist with literature research, code development, and report drafting. All experimental results were generated by our pipeline and independently verified. The evaluation methodology, dataset construction, and model architecture decisions were made by the author.
