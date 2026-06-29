# Multi-View Breast Cancer Detection on VinDr-Mammo

**From 0.756 to 0.846 AUC** — a six-stage pipeline combining single-view CNN baselines, multi-view fusion, CrossView Transformer, bounding-box annotation ablation, MC Dropout uncertainty quantification, and hybrid triage simulation.

> Haoming Luo, Matthew Hamilton  
> Department of Computer Science, Memorial University, St. John's, NL, Canada  
> Preprint: [`paper/research_paper.pdf`](paper/research_paper.pdf)

---

## Results Summary

| Stage | Method | AUC | vs. single-view |
|-------|--------|-----|-----------------|
| Prior work | MamT4 (4-view Transformer) | 0.840 | — |
| Stage 1 | DenseNet-121 (single-view) | 0.756 | baseline |
| Stage 2 | EfficientNet-B3 +reg (BreastPairNet) | 0.826 | +9.1% |
| Stage 3 | ConvNeXt-Base (CrossView Transformer) | 0.832 | +10.0% |
| **Stage 5** | **16-model ensemble + TTA×5** | **0.846** | **+11.8%** ★ |
| External | RSNA 2022 (zero-shot transfer) | 0.601 | domain gap |

Key finding: bounding-box annotations consistently **hurt** AUC by 0.12–0.20 due to shortcut learning.

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
│       ├── compute_bootstrap_ci.py  # 95% BCa bootstrap CIs
│       └── generate_gradcam.py      # Grad-CAM visualisations
└── paper/
    ├── research_paper.tex
    └── research_paper.pdf
```

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

# Stage 2 — BreastPairNet multi-view fusion
python scripts/training/train_multiview.py --backbone efficientnet_b3 --fusion attention --reg

# Stage 3 — CrossView Transformer
python scripts/training/train_crossview.py --backbone convnext_base

# Stage 4 — MC Dropout uncertainty
python scripts/evaluation/evaluate_uncertainty.py --checkpoint <path>

# Stage 5 — Ensemble + TTA
python scripts/evaluation/run_ensemble.py --checkpoints <dir>

# Stage 6 — External validation (RSNA)
python scripts/evaluation/run_external_validation.py --checkpoint <path>

# Bootstrap 95% CIs
python scripts/evaluation/compute_bootstrap_ci.py --results <path>
```

---

## Datasets

| Dataset | Source | Size | Cancer rate |
|---------|--------|------|-------------|
| VinDr-Mammo | [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) | 20,000 images / 5,000 patients | 4.94% |
| RSNA 2022 | [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) | 54,706 images / 11,913 patients | ~0.02% |

> **Note:** Datasets are not included in this repository. Download links above require free registration.

---

## Hardware

All experiments were run on a single NVIDIA RTX 5070 (12 GB VRAM) with bfloat16 mixed precision. Training time per model: approximately 2–4 hours.

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
