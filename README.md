# Multi-View Breast Cancer Detection on VinDr-Mammo

**From 0.756 to 0.846 AUC** — a six-stage pipeline combining single-view CNN baselines, multi-view fusion (BreastPairNet), CrossView Transformer, bounding-box annotation ablation, MC Dropout uncertainty quantification, and hybrid triage simulation.

> Haoming Luo, Matthew Hamilton  
> Department of Computer Science, Memorial University, St. John's, NL, Canada  
> Preprint: [`paper/research_paper.pdf`](paper/research_paper.pdf) · Full paper: [`paper/research_paper.md`](paper/research_paper.md)

---

## Results Summary

| Stage | Method | AUC | vs. single-view |
|-------|--------|-----|-----------------|
| Prior work | MamT4 [Ianculescu et al., ICASSP 2024] | 0.840 | — |
| Stage 1 | DenseNet-121 (single-view) | 0.756 | baseline |
| Stage 2 | EfficientNet-B3 +reg (BreastPairNet) | 0.826 | +9.1% |
| Stage 3 | ConvNeXt-Base (CrossView Transformer) | 0.832 | +10.0% |
| **Stage 5** | **16-model ensemble + TTA×5** | **0.846** | **+11.8% ★** |
| External | RSNA 2022 (zero-shot transfer) | 0.601 | domain gap |

Key finding: bounding-box annotations consistently **hurt** AUC by 0.12–0.20 due to shortcut learning (train/test distribution shift).

---

## Repository Structure

```
breast-cancer-detection/
├── config.py                        # Paths and settings (edit before running)
├── requirements.txt
├── tools/                           # Core model and data utilities
│   ├── preprocessing.py             # DICOM → 8-bit PNG (CLAHE, orientation)
│   ├── cnn_tools.py                 # Single-view CNN training
│   ├── multiview_tools.py           # BreastPairNet (Stage 2)
│   ├── crossview_tools.py           # CrossView Transformer (Stage 3)
│   ├── full_dataset_tools.py        # Full-dataset training helpers
│   ├── dicom_tools.py               # DICOM loading utilities
│   ├── dicom_cnn_tools.py           # DICOM-aware CNN wrappers
│   └── ml_tools.py                  # Metrics, bootstrap CI, triage simulation
├── scripts/
│   ├── setup/
│   │   ├── setup_vindr.py           # Verify VinDr-Mammo dataset layout
│   │   ├── download_rsna.py         # Download RSNA 2022 dataset via Kaggle API
│   │   └── check_setup.py           # Check GPU, dependencies, paths
│   ├── preprocessing/
│   │   ├── preprocess_vindr.py      # Convert VinDr DICOMs to PNGs
│   │   └── preprocess_rsna.py       # Convert RSNA DICOMs to PNGs
│   ├── training/
│   │   ├── train_single_cnn.py      # Stage 1: single-view baselines
│   │   ├── train_multiview.py       # Stage 2: BreastPairNet
│   │   ├── train_crossview.py       # Stage 3: CrossView Transformer
│   │   ├── train_asymmetry.py       # Bilateral asymmetry model
│   │   ├── train_yolo.py            # YOLOv8 lesion detector
│   │   └── annotate_pngs.py         # Draw bounding boxes for ablation
│   └── evaluation/
│       ├── run_ensemble.py          # Stage 5: ensemble + TTA
│       ├── evaluate_uncertainty.py  # Stage 4: MC Dropout
│       ├── run_external_validation.py  # Stage 6: RSNA zero-shot eval
│       ├── compute_all_bootstrap_ci.py  # 95% BCa bootstrap CIs (all models)
│       ├── compute_bootstrap_ci.py  # 95% BCa bootstrap CIs (single model)
│       └── generate_gradcam.py      # Grad-CAM visualisations
├── results/                         # ★ All saved result JSONs (see below)
│   ├── result_source_map.json       # ★ Maps every paper claim → source file
│   ├── bootstrap_ci_all.json        # Per-model AUC + S@90 point estimates + CIs
│   ├── per_model/
│   │   ├── stage1/                  # Single-view CNN result JSONs
│   │   ├── stage2/                  # BreastPairNet result JSONs
│   │   ├── stage3/                  # CrossView Transformer result JSONs
│   │   ├── stage4/                  # MC Dropout / uncertainty result JSON
│   │   └── stage5/                  # Ensemble result JSONs
│   └── external_validation/         # RSNA 2022 zero-shot result JSON
└── paper/
    ├── research_paper.md            # Full paper (Markdown, corrected metrics)
    ├── research_paper.tex
    └── research_paper.pdf
```

---

## AI Agent Verification Guide

This section is specifically for AI agents (or human reviewers) who want to verify that the numbers in the paper match the saved results without re-running any training.

### Quick start

The single most important file is **`results/result_source_map.json`**. It maps every paper result to:
- the exact model checkpoint that produced it
- the training log
- the saved result JSON

Read this file first:

```python
import json
with open("results/result_source_map.json") as f:
    source_map = json.load(f)
print(source_map["metric_correction_note"]["summary"])
```

### Verifying a specific paper claim

**Example: "ConvNeXt-Base CrossView Transformer achieves AUC=0.8317, S@90%=0.5758"**

```python
import json

# Load authoritative bootstrap results
with open("results/bootstrap_ci_all.json") as f:
    ci_data = json.load(f)

# Find the ConvNeXt-Base entry
row = next(r for r in ci_data["rows"] if "convnext_base" in r["checkpoint"])
print(f"AUC  = {row['point']['auc']:.4f}  CI [{row['ci_95']['auc'][0]}, {row['ci_95']['auc'][1]}]")
print(f"S@90 = {row['point']['s90']:.4f}  CI [{row['ci_95']['s90'][0]}, {row['ci_95']['s90'][1]}]")
print(f"F1   = {row['point']['f1']:.4f}")
print(f"Sens = {row['point']['sens']:.4f}")
# Expected: AUC=0.8317, S@90=0.5758, F1=0.2587, Sens=0.7475
```

**Example: "EfficientNet-B3 +reg (best Stage 2) achieves AUC=0.8256, S@90%=0.6263"**

```python
row = next(r for r in ci_data["rows"] if "efficientnet_b3_attention_20260511" in r["checkpoint"])
# Expected: AUC=0.8256, S@90=0.6263 [CI: 0.5229–0.7143]
```

### Verifying ALL Table 2 and Table 3 numbers at once

```python
import json

with open("results/bootstrap_ci_all.json") as f:
    ci = json.load(f)

# Table 2: BreastPairNet (stage2_breastpairnet in source_map)
with open("results/result_source_map.json") as f:
    sm = json.load(f)

print("=== Stage 2 (BreastPairNet) verification ===")
for entry in sm["stage2_breastpairnet"]:
    ckpt_stem = entry["model_checkpoint"].split("/")[-1].replace(".pt", "")
    row = next((r for r in ci["rows"] if ckpt_stem in r["checkpoint"]), None)
    if row:
        match_auc  = abs(row["point"]["auc"]  - entry["auc"])  < 0.0002
        match_f1   = abs(row["point"]["f1"]   - entry["f1"])   < 0.0002
        match_sens = abs(row["point"]["sens"] - entry["sens"])  < 0.0002
        match_s90  = abs(row["point"]["s90"]  - entry["s90"])   < 0.0002
        status = "OK" if all([match_auc, match_f1, match_sens, match_s90]) else "MISMATCH"
        print(f"[{status}] {entry['experiment']:35s}  AUC={entry['auc']:.4f}  F1={entry['f1']:.4f}  Sens={entry['sens']:.4f}  S@90={entry['s90']:.4f}")

print()
print("=== Stage 3 (CrossView Transformer) verification ===")
for entry in sm["stage3_crossview_transformer"]:
    ckpt_stem = entry["model_checkpoint"].split("/")[-1].replace(".pt", "")
    row = next((r for r in ci["rows"] if ckpt_stem in r["checkpoint"]), None)
    if row:
        match_auc  = abs(row["point"]["auc"]  - entry["auc"])  < 0.0002
        match_s90  = abs(row["point"]["s90"]  - entry["s90"])   < 0.0002
        status = "OK" if all([match_auc, match_s90]) else "MISMATCH"
        print(f"[{status}] {entry['experiment']:35s}  AUC={entry['auc']:.4f}  S@90={entry['s90']:.4f}")
```

All entries should print `[OK]`.

### Verifying the ensemble result

```python
import json

with open("results/per_model/stage5/ensemble_mean_top16_+asym_tta5_20260512_074729.json") as f:
    ens = json.load(f)

# Expected from paper Table 5b:
print(f"AUC   = {ens.get('auc', ens.get('ensemble_auc')):.4f}")   # 0.8458
print(f"S@90% = {ens.get('s90', ens.get('sens_at_90spec')):.4f}") # 0.6566
print(f"Sens  = {ens.get('sens_opt', ens.get('sensitivity')):.4f}") # 0.7374
```

### Verifying the RSNA external validation

```python
import json

with open("results/external_validation/rsna_external_val_20260520_165623.json") as f:
    rsna = json.load(f)

# Expected from paper Table 8:
# AUC=0.6008 [0.5577–0.6585], S@90=0.1636, Sens=0.3909, Spec=0.7598
```

### Re-running bootstrap CI from scratch

If you have the model checkpoints and preprocessed VinDr-Mammo PNGs, you can recompute all CIs:

```bash
# Computes per-model AUC, F1, Sens, Spec, S@90 + 95% BCa bootstrap CIs
# Output: results/bootstrap_ci_all_<timestamp>.json
python scripts/evaluation/compute_all_bootstrap_ci.py \
    --models_dir /path/to/checkpoints \
    --data_dir   /path/to/vindr_pngs \
    --n_boot 2000
```

Expected runtime: ~2–4 hours on a single GPU.

### Important note on metric provenance

**All F1, sensitivity, spec, and S@90% values in Tables 2–3 come from `results/bootstrap_ci_all.json`, NOT from the training logs.** The training script (`tools/multiview_tools.py`) logged F1/sens/S@90 from the last training epoch's validation split, while AUC was from the best epoch's validation AUC — a mismatch that made those logged metrics incorrect for the test split. The bootstrap script re-evaluates each best checkpoint on the held-out test split (2,000 pairs, 99 cancer-positive) and is the authoritative source. See `results/result_source_map.json` → `"metric_correction_note"` for the full explanation.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# PyTorch with CUDA (adjust cu128 to your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Configure paths

Edit `config.py` to point to your dataset locations:

```python
VINDR_DIR  = Path("/path/to/vindr-mammo/dicom")
MODELS_DIR = Path("/path/to/save/checkpoints")
```

### 3. Download and preprocess datasets

**VinDr-Mammo** — download from [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/), then:

```bash
python scripts/setup/setup_vindr.py
python scripts/preprocessing/preprocess_vindr.py
```

**RSNA 2022** (optional, for external validation):

```bash
python scripts/setup/download_rsna.py   # requires Kaggle API credentials
python scripts/preprocessing/preprocess_rsna.py
```

---

## Reproducing Results

Run stages in order:

```bash
# Stage 1 — single-view CNN baselines
python scripts/training/train_single_cnn.py --backbone densenet121

# Stage 2 — BreastPairNet multi-view fusion (best config)
python scripts/training/train_multiview.py --backbone efficientnet_b3 --fusion attention --reg

# Stage 3 — CrossView Transformer (best config)
python scripts/training/train_crossview.py --backbone convnext_base

# Stage 4 — MC Dropout uncertainty
python scripts/evaluation/evaluate_uncertainty.py --checkpoint <path/to/checkpoint>

# Stage 5 — Ensemble + TTA
python scripts/evaluation/run_ensemble.py --tta 5 --method mean

# Stage 6 — External validation (RSNA)
python scripts/evaluation/run_external_validation.py

# Bootstrap 95% CIs (all models, authoritative metrics for Tables 2–3)
python scripts/evaluation/compute_all_bootstrap_ci.py
```

---

## Datasets

| Dataset | Source | Size | Cancer rate |
|---------|--------|------|-------------|
| VinDr-Mammo | [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) | 20,000 images / 5,000 patients | 4.94% |
| RSNA 2022 | [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) | 54,706 images / 11,913 patients | ~2% |

> **Note:** Datasets are not included in this repository. Download links above require free registration.

---

## Hardware

All experiments were run on a single **NVIDIA RTX 5070 (12 GB VRAM)** with bfloat16 mixed precision. Training time per model: approximately 2–4 hours.

---

## Citation

```bibtex
@article{luo2026multiview,
  title   = {Multi-View Fusion, CrossView Transformer, and Uncertainty-Aware Triage
             for Breast Cancer Detection on VinDr-Mammo},
  author  = {Luo, Haoming and Hamilton, Matthew},
  year    = {2026},
  note    = {Preprint}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
