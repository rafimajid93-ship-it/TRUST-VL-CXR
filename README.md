# TRUST-VL-CXR

**TRUST-VL-CXR** is a research pipeline for studying **reliability in chest X-ray (CXR) and radiology-report alignment**.

The project trains a multimodal image–text matching model and then freezes it before training a **separate, decoupled reliability monitor**. The monitor learns to detect when a CXR and report are inconsistent and is evaluated against the base model's confidence, evidential uncertainty, and image–text cosine similarity.

> **Research use only.** This project is not a medical device and is not intended for clinical diagnosis or patient-care decisions.

---

## Overview

The main research question is:

> Can an independent reliability monitor detect image–report mismatches better than relying only on a multimodal model's own confidence or uncertainty?

The notebook implements a patient-disjoint and reproducible evaluation pipeline using chest X-rays and report text from a Kaggle mirror of MIMIC-CXR-derived data.

### Core workflow

```text
Chest X-ray ──> Frozen CXR Encoder ──> Image Features ─┐
                                                      ├─> Base Matching Model
Report ───────> Frozen BiomedBERT ───> Text Features ─┘
                                                            │
                                                            │ train first
                                                            ▼
                                                   Freeze Base Model
                                                            │
Image Features + Text Features ─────────────────────────────┤
                                                            ▼
                                              Reliability Transformer
                                                            │
                                      ┌─────────────────────┴─────────────────────┐
                                      ▼                                           ▼
                              Conflict Detection                         Failure-Type Prediction
```

---

## Key Features

- **Frozen medical encoders**
  - TorchXRayVision DenseNet-121 with CheXpert weights for CXR features.
  - Microsoft BiomedBERT for report features.
- **Patient-disjoint splitting** to reduce patient-level information leakage.
- **Image ↔ report retrieval** evaluation in both directions.
- **Evidential pair classifier** that estimates match probability and uncertainty.
- **Decoupled reliability monitor** trained only after the base model is frozen.
- **Cross-modal Transformer monitor** using image, text, difference, product, cosine, and uncertainty-style tokens.
- **Multi-task reliability learning**
  - aligned vs. mismatched pair detection;
  - failure/mismatch-type prediction.
- **Hard-negative testing**, including difficult cross-patient report mismatches.
- **Dataset-aware mismatch construction**, including metadata/view-aware and semantic hard negatives where available.
- **Language perturbation analysis** using report augmentations only for evaluation.
- **Visual corruption / distribution-shift analysis**.
- **PA/AP subgroup analysis**.
- **Calibration analysis**, including Platt calibration, Brier score, and ECE.
- **Patient-cluster bootstrap confidence intervals**.
- **Multi-seed experiments** and resumable experiment artifacts.
- **Reproducibility exports** containing protocol, environment, provenance, metrics, plots, and derived predictions.

---

## Dataset

The notebook expects the Kaggle dataset:

```text
simhadrisadaram/mimic-cxr-dataset
```

Expected resources include:

```text
official_data_iccv_final/
mimic_cxr_aug_train.csv
mimic_cxr_aug_validate.csv
```

The pipeline conservatively reconstructs study–report pairs and uses frontal **PA/AP** images.

The provided validation CSV is treated as a **locked holdout set**. An internal validation split is created from the training pool using patient-level grouping.

### Important Data Note

The notebook does **not** redistribute raw chest X-rays or raw clinical reports. Review the source dataset's current terms and license before publishing or redistributing any data-derived material.

---

## Model Architecture

### 1. Frozen Feature Encoders

**Image encoder**

```text
TorchXRayVision DenseNet-121
Weights: densenet121-res224-chex
```

**Text encoder**

```text
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

The BiomedBERT repository revision is pinned in the experiment configuration for reproducibility.

### 2. Base Multimodal Model

The base model contains:

```text
Image Projection
       │
       ├─────────────┐
       │             │
Text Projection      │
       │             │
       └──> Gated Fusion
                 │
                 ▼
          Evidential Pair Head
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Match Probability     Uncertainty
```

Training combines multimodal alignment with patient-aware false-negative control.

### 3. Reliability Monitor

The reliability monitor is **independent from the trained base pair head**.

Its Transformer receives representations derived from:

```text
Image
Text
Image - Text
Image × Text
Cosine similarity
Uncertainty-style token
```

It predicts:

1. **Conflict probability** — whether the image/report pair is mismatched.
2. **Failure category** — the type of mismatch/failure.

The base model is frozen before monitor training, and parameter hashing is used to verify that it remains unchanged.

---

## Evaluation

### Retrieval

- Image → Text Recall@1 / Recall@5 / Recall@10
- Text → Image Recall@1 / Recall@5 / Recall@10
- Mean Reciprocal Rank (MRR)
- Patient-masked retrieval

### Pair Matching and Reliability

Metrics include:

- AUROC
- AUPRC
- Balanced accuracy
- Brier score
- Expected Calibration Error (ECE)

The reliability monitor is compared with:

- base match/conflict score;
- evidential uncertainty;
- projected image–text cosine similarity.

### Stress Tests

The locked holdout evaluation includes:

- hard report mismatches;
- random report mismatches;
- language perturbations;
- visual corruption / shift analysis;
- PA vs. AP subgroup analysis.

### Statistical Uncertainty

Patient-clustered bootstrap resampling is used to generate **95% confidence intervals** for selected metrics.

---

## Experiment Profiles

| Profile | Seeds | Base Epochs | Monitor Epochs | Bootstrap Reps | Purpose |
|---|---:|---:|---:|---:|---|
| `smoke` | 1 | 2 | 2 | 100 | Fast pipeline/debug check |
| `research` | 1 | 10 | 10 | 500 | Development experiments |
| `paper` | 3 | 16 | 16 | 2000 | Full evaluation |

The current notebook configuration uses:

```python
PROFILE = "paper"
```

Change this to `smoke` for a quick first run.

---

## Running the Project

### Recommended Environment

This notebook is designed for **Kaggle with a GPU enabled**.

A compatible run requires:

- Python
- CUDA-enabled PyTorch
- PyTorch >= 2.6 for the pinned BiomedBERT checkpoint under the current Transformers security requirements
- torchvision
- torchxrayvision
- transformers
- huggingface_hub
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- Pillow
- tqdm

The notebook performs runtime checks and records the environment used for each experiment.

### Steps

1. Create a new Kaggle notebook.
2. Enable a **GPU accelerator**.
3. Add the dataset:

   ```text
   simhadrisadaram/mimic-cxr-dataset
   ```

4. Enable Internet for the first run if model weights are not already cached.
5. Upload/open this project's notebook.
6. Choose the experiment profile:

   ```python
   PROFILE = "smoke"
   ```

7. Run all cells from top to bottom.
8. For the full experiment, change to:

   ```python
   PROFILE = "paper"
   ```

---

## Generated Outputs

Experiment artifacts are written under:

```text
/kaggle/working/TRUST_VL_CXR/<profile>/
```

Typical outputs include:

```text
environment.json
protocol_config.json
dataset_provenance.json
encoder_provenance.json
data_audit.json
feature_audit.json
seed_level_results.csv
aggregate_results.csv
bootstrap_all_seeds.csv
manifest_index_train.csv
manifest_index_internal_val.csv
manifest_index_holdout.csv
pip_freeze.txt
figures/
seed_<seed>/
export/
```

The notebook also builds a compact reproducibility archive:

```text
TRUST_VL_CXR_<profile>_RESULTS.zip
```

Raw reports, raw CXRs, and cached embeddings are intentionally excluded from the compact export.

---

## Reproducibility

Several safeguards are included:

- fixed random seeds;
- deterministic patient-level splitting;
- locked holdout evaluation;
- pinned text-model revision;
- encoder/environment provenance;
- dataset file hashes;
- feature-cache fingerprints;
- deterministic evaluation subsets;
- model parameter SHA-256 hashing;
- patient-cluster bootstrap intervals;
- multi-seed aggregation;
- resumable seed-level experiments.

A seed is considered complete only after its required predictions, metrics, subgroup analyses, bootstrap intervals, and shift-analysis artifacts have been written.

---

## Scope and Limitations

This project studies **CXR–report compatibility and reliability**, not lesion segmentation.

In particular:

- the Kaggle mirror used here does not provide lesion masks;
- no segmentation claims are made;
- no disease labels are generated by report keyword matching;
- report augmentations are used for evaluation rather than training supervision;
- mismatch experiments are controlled stress tests and do not represent real clinical prevalence;
- strong performance on synthetic mismatch detection should not be interpreted as demonstrated clinical safety.

---

## Repository Structure

```text
TRUST-VL-CXR/
├── README.md
└── trust-vl-cxr.ipynb
```

You can rename the notebook to `trust-vl-cxr.ipynb` before uploading it to GitHub for a cleaner repository.

---

## Research Direction

TRUST-VL-CXR explores the broader idea that a deployed multimodal system should have a mechanism that can independently ask:

> **"Should this prediction be trusted under the current image–text relationship?"**

The experiment separates task learning from reliability monitoring so that reliability is not measured only through the task model's own confidence.

---

## Citation

If this repository is used in academic work, please cite the corresponding paper/project once a formal citation is available.

```bibtex
@misc{trustvlcxr,
  title  = {TRUST-VL-CXR: Decoupled Reliability Monitoring for Chest X-Ray and Report Alignment},
  author = {Rafi Majid},
  year   = {2026},
  note   = {Research code repository}
}
```

Replace the placeholder author information with the final project citation before publication.

---
