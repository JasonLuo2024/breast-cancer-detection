# NECEC 2026 Extended Abstract
**Deadline:** September 18, 2026 | **Notification:** October 14, 2026

---

## Paper Title
MamNet-6: Multi-View Fusion, CrossView Transformer, and Uncertainty-Aware Triage for Breast Cancer Detection on VinDr-Mammo

## Authors
Haoming Luo, Matthew Hamilton, Oscar Meruvia-Pastor, Edward Kendall
Department of Computer Science, Memorial University, St. John's, NL, Canada
hluo@mun.ca

## Keywords
breast cancer detection, mammography, multi-view learning, uncertainty quantification, triage

---

## Extended Abstract

Breast cancer is the most commonly diagnosed cancer worldwide. Population screening with full-field digital mammography (FFDM) reduces mortality but imposes a heavy radiologist workload and substantial inter-reader variability of 10–30 percentage points (Lehman et al., 2017). Meta-analyses report pooled AUC of 0.87 for deep learning AI versus 0.81 for radiologists on digital mammography (Dembrower et al., 2020), motivating automated computer-aided diagnosis. This work develops and evaluates MamNet-6, a six-stage pipeline on the VinDr-Mammo benchmark (Nguyen et al., 2023), a 20,000-image Vietnamese FFDM dataset, addressing three gaps: single-view dominance in existing VinDr results, underexplored spatial cross-view attention, and lack of uncertainty-aware triage simulation.

MamNet-6 progresses through six stages. Stage 1 trains five single-view CNN backbones (ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-Tiny/Base) with and without regularisation (+reg: elastic deformation, mixup α=0.2, label smoothing 0.1). Stage 2 introduces BreastPairNet, a shared-backbone multi-view network evaluated across twelve backbone×fusion×resolution variants (concat vs. attention, 512 vs. 1024 px). Stage 3 adds a CrossView Transformer with spatial cross-attention between CC and MLO feature maps before global average pooling. Stage 3b performs a bounding-box annotation ablation to test whether coarse localisation annotations shortcut learning. Stage 4 applies Monte Carlo (MC) Dropout (N=20 stochastic passes) for predictive uncertainty quantification. Stage 5 assembles a 16-model ensemble with test-time augmentation (TTA×5). Stage 6 simulates a Verboom-style hybrid triage policy (Verboom et al., 2021) routing the most-confident AI reads autonomously and referring the remainder to a radiologist. All models use AdamW optimiser with CosineAnnealingLR, FP16 mixed-precision, and FocalLoss (α=0.25, γ=2.0). Confidence intervals are estimated via percentile bootstrap (n=2000). External zero-shot validation is performed on the RSNA 2022 screening mammography dataset.

DenseNet-121 achieved AUROC=0.756 at Stage 1. BreastPairNet improved to AUROC=0.826 [95% CI 0.776–0.871] (+9.1%), and the CrossView Transformer reached AUROC=0.832 [0.783–0.875] (+10.0%). Adjacent-stage differences lie within overlapping confidence intervals; the cumulative Stage 1→5 gain is the most statistically robust comparison. Bounding-box annotations consistently degraded AUROC by 0.12–0.20, indicating shortcut learning from coarse localisation. The 16-model ensemble achieved AUROC=0.846 (+11.8% vs. Stage 1) with sensitivity at 90% specificity (S@90%spec)=0.657 and PPV=0.184 at the triage threshold. Hybrid triage at 50% AI reads preserved 87.9% sensitivity under oracle conditions. Zero-shot transfer to RSNA 2022 yielded AUROC=0.601, with S@90%spec=0.164, highlighting a large domain gap between Vietnamese and North American screening populations.

MamNet-6 contributes a systematic ablation study across six complementary design axes — backbone, multi-view fusion strategy, spatial cross-attention, uncertainty quantification, ensemble construction, and clinical triage simulation — on a single large FFDM benchmark. The pipeline delivers a cumulative 11.8% AUROC gain over single-view CNN. Two key negative findings are reported: bounding-box shortcut learning degrades performance, and domain generalisation remains an open challenge. These results are directly relevant to the NECEC 2026 themes of signal processing, machine intelligence, and biomedical engineering.

---

## References
- Dembrower, K. et al. (2020). Effect of AI-supported mammography screening. *Lancet Oncology*.
- Lehman, C. et al. (2017). National performance benchmarks for screening mammography. *Radiology*.
- Nguyen, H. et al. (2023). VinDr-Mammo: A large-scale benchmark dataset. *Scientific Data*.
- Verboom, P. et al. (2021). Standalone AI for triage in breast cancer screening. *Radiology*.
