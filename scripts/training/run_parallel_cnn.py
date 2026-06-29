"""
Run CNN experiments in parallel groups, fitting within RTX 5070 VRAM (~12 GB).

Usage:
    python scripts/run_parallel_cnn.py                  # all pending baseline
    python scripts/run_parallel_cnn.py --workers 2      # limit to 2 at once
    python scripts/run_parallel_cnn.py --phase prep     # preprocessing ablation
    python scripts/run_parallel_cnn.py --phase hyper    # hyperparameter search
    python scripts/run_parallel_cnn.py --phase advanced  # advanced experiments
    python scripts/run_parallel_cnn.py --phase bilateral # bilateral 4-view cross-attention
    python scripts/run_parallel_cnn.py --phase asymmetry # asymmetry diff bilateral (simpler)
    python scripts/run_parallel_cnn.py --phase medical   # medical-pretrained backbone
    python scripts/run_parallel_cnn.py --phase ensemble  # ensemble of best ipsilateral models
    python scripts/run_parallel_cnn.py --phase all       # everything pending
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

PYTHON = r"C:\Users\JasonLuo2026\anaconda3\envs\medical\python.exe"
PROGRESS_FILE = BASE_DIR / "data" / "progress.json"
LOG_DIR = BASE_DIR / "data"

ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"}


# ── experiment definitions ────────────────────────────────────────────────────

# Phase 2: CNN baseline (15 epochs, standard preprocessing)
# VRAM estimates: VGG19~3GB, ResNet50~2.5GB, EffB3~2GB, ResNet18~1.5GB, Dense~2GB, Custom~1GB
BASELINE = [
    {"key": "cnn_vgg19_e15",          "arch": "vgg19",          "epochs": 15, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "standard"},
    {"key": "cnn_efficientnet_b3_e15", "arch": "efficientnet_b3","epochs": 15, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "standard"},
    {"key": "cnn_resnet18_e15",        "arch": "resnet18",       "epochs": 15, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "standard"},
    {"key": "cnn_densenet121_e15",     "arch": "densenet121",    "epochs": 15, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "standard"},
    {"key": "cnn_custom_cnn_e15",      "arch": "custom_cnn",     "epochs": 15, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "standard"},
]

# Phase 3: Preprocessing ablation on best arch from Phase 2
# (best_arch resolved at runtime from progress.json)
PREPROCESSING = [
    {"key": "cnn_{best}_clahe_e20",        "epochs": 20, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_{best}_zscore_clahe_e20", "epochs": 20, "lr": 1e-4, "batch": 16, "img": 224, "gamma": 2.0, "prep": "zscore_clahe"},
]

# Phase 4: Hyperparameter search on best arch + best preprocessing
HYPERPARAM = [
    {"key": "cnn_{best}_gamma1_5",  "epochs": 20, "lr": 1e-4, "batch": 64, "img": 224, "gamma": 1.5, "prep": "clahe"},
    {"key": "cnn_{best}_gamma2_5",  "epochs": 20, "lr": 1e-4, "batch": 64, "img": 224, "gamma": 2.5, "prep": "clahe"},
    {"key": "cnn_{best}_lr5e5",     "epochs": 25, "lr": 5e-5, "batch": 64, "img": 224, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_{best}_lr2e4",     "epochs": 20, "lr": 2e-4, "batch": 64, "img": 224, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_{best}_img256",    "epochs": 20, "lr": 1e-4, "batch": 32, "img": 256, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_{best}_e30",       "epochs": 30, "lr": 5e-5, "batch": 64, "img": 224, "gamma": 2.0, "prep": "clahe"},
]

# Phase 5: Advanced — top architectures with CLAHE + extended training
ADVANCED = [
    {"key": "cnn_vgg19_clahe_e25",    "arch": "vgg19",          "epochs": 25, "lr": 5e-5, "batch": 32, "img": 256, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_resnet50_clahe_e25", "arch": "resnet50",       "epochs": 25, "lr": 5e-5, "batch": 64, "img": 224, "gamma": 2.0, "prep": "clahe"},
    {"key": "cnn_effb3_clahe_e25",    "arch": "efficientnet_b3","epochs": 25, "lr": 1e-4, "batch": 64, "img": 224, "gamma": 2.5, "prep": "clahe"},
]

# Phase 6: DICOM-native training (kept for reference — superseded by Phase 7)
DICOM = [
    {"key": "dicom_resnet50_clahe_e20",    "arch": "resnet50",        "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.0, "prep": "clahe",        "dicom": True},
    {"key": "dicom_densenet121_clahe_e20", "arch": "densenet121",     "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.0, "prep": "clahe",        "dicom": True},
    {"key": "dicom_effb3_clahe_e20",       "arch": "efficientnet_b3", "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.5, "prep": "clahe",        "dicom": True},
    {"key": "dicom_resnet50_zscl_e20",     "arch": "resnet50",        "epochs": 20, "lr": 5e-5, "batch": 32, "img": 224, "gamma": 2.0, "prep": "zscore_clahe", "dicom": True},
    {"key": "dicom_resnet50_img320_e20",   "arch": "resnet50",        "epochs": 20, "lr": 1e-4, "batch": 16, "img": 320, "gamma": 2.0, "prep": "clahe",        "dicom": True},
]

# Phase 7: Full preprocessed dataset — 16k train / 4k test, patient-level split
# PNGs pre-converted from raw DICOMs by preprocess_full_dataset.py → G:\
# batch=64, workers=8 → keeps RTX 5070 saturated (500+ batches/epoch)
FULL = [
    {"key": "full_resnet50_e20",       "arch": "resnet50",        "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.0, "full": True},
    {"key": "full_densenet121_e20",    "arch": "densenet121",     "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.0, "full": True},
    {"key": "full_efficientb3_e20",    "arch": "efficientnet_b3", "epochs": 20, "lr": 1e-4, "batch": 32, "img": 224, "gamma": 2.5, "full": True},
    {"key": "full_vgg19_e20",          "arch": "vgg19",           "epochs": 20, "lr": 5e-5, "batch": 16, "img": 224, "gamma": 2.0, "full": True},
    {"key": "full_resnet50_img384_e20","arch": "resnet50",        "epochs": 20, "lr": 1e-4, "batch": 16, "img": 384, "gamma": 2.0, "full": True},
]

# Phase 9: Bilateral 4-view fusion — Phase A+B improvement
# Each sample = full patient quad (L-CC, L-MLO, R-CC, R-MLO).
# BilateralFusionNet: shared backbone + ipsilateral concat + cross-bilateral MHA.
# Phase A: label_smoothing=0.1, mixup=0.2, weight_decay=1e-3 (fixes overfitting).
# Phase B: bilateral cross-attention captures contralateral asymmetry.
# batch=6 → 24 images per step; workers=1 to avoid VRAM contention between runs.
BILATERAL = [
    # Best baseline backbone (DenseNet-121) — primary experiment
    {"key": "bl_densenet121_e30",  "arch": "densenet121",    "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "bilateral": True},
    # EfficientNet-B3 — best F1 in ipsilateral phase
    {"key": "bl_effb3_e30",       "arch": "efficientnet_b3", "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.5,
     "label_smoothing": 0.1, "mixup": 0.2, "bilateral": True},
    # ResNet-50 — baseline comparison
    {"key": "bl_resnet50_e30",    "arch": "resnet50",        "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "bilateral": True},
    # DenseNet-121 extended + stronger MixUp — ablation
    {"key": "bl_densenet121_e40", "arch": "densenet121",    "epochs": 40,
     "lr": 3e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.3, "bilateral": True},
]

# Phase 10: Asymmetry bilateral — |h_L - h_R| diff, no cross-attention
# Simpler than BilateralFusionNet: fewer params, less prone to overfitting 385 positives.
# Same PatientQuadDataset, same training loop. Run sequentially (workers=1).
ASYMMETRY = [
    # Best backbone (DenseNet-121) — primary
    {"key": "asym_densenet121_e30", "arch": "densenet121", "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "asymmetry": True},
    # EfficientNet-B3
    {"key": "asym_effb3_e30",       "arch": "efficientnet_b3", "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.5,
     "label_smoothing": 0.1, "mixup": 0.2, "asymmetry": True},
    # ResNet-50
    {"key": "asym_resnet50_e30",    "arch": "resnet50",        "epochs": 30,
     "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "asymmetry": True},
]

# Phase 11: Medical-pretrained backbone — DenseNet-121 from TorchXRayVision
# Pretrained on NIH+PC+CheXpert+MIMIC+Google+OpenI+Kaggle chest X-rays.
# Lower LR (3e-5) since medical features already closer to mammography.
MEDICAL = [
    # Medical pretrained + asymmetry head — primary
    {"key": "med_densenet121_e30", "arch": "densenet121", "epochs": 30,
     "lr": 3e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "medical": True},
    # Extended training — more epochs since features are already good
    {"key": "med_densenet121_e40", "arch": "densenet121", "epochs": 40,
     "lr": 2e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "label_smoothing": 0.1, "mixup": 0.2, "medical": True},
]

# Phase 12: Hi-res 512px ipsilateral fusion — AMP required to fit 12 GB VRAM
# Two backbone passes at 512px × batch=3 → 6 images; AMP halves activation memory.
# Lower LR (5e-5) and longer freeze (3 ep) for stable convergence at higher res.
HIRES = [
    {"key": "mv_densenet121_concat_512_e25", "arch": "densenet121",
     "fusion": "concat",    "epochs": 25, "lr": 5e-5, "batch": 3,
     "img": 512, "gamma": 2.0, "mv": True, "amp": True},
    {"key": "mv_effb3_attn_512_e25",         "arch": "efficientnet_b3",
     "fusion": "attention", "epochs": 25, "lr": 5e-5, "batch": 3,
     "img": 512, "gamma": 2.5, "mv": True, "amp": True},
    # Phase 12b: Additional 512px models with label smoothing (no mixup — OOM at 512px+mixup)
    {"key": "mv_densenet121_attn_512_e30", "arch": "densenet121",
     "fusion": "attention", "epochs": 30, "lr": 5e-5, "batch": 2,
     "img": 512, "gamma": 2.0, "mv": True, "amp": True,
     "label_smoothing": 0.1},
    {"key": "mv_resnet50_attn_512_e30",    "arch": "resnet50",
     "fusion": "attention", "epochs": 30, "lr": 5e-5, "batch": 2,
     "img": 512, "gamma": 2.0, "mv": True, "amp": True,
     "label_smoothing": 0.1},
]

# Phase 13: Improved 224px ipsilateral — label smoothing + mixup regularization.
# Retrain best-performing architectures; cosine annealing already in training loop.
IMPROVED_MV = [
    {"key": "mv_densenet121_attn_reg_e25", "arch": "densenet121",
     "fusion": "attention", "epochs": 25, "lr": 1e-4, "batch": 8,
     "img": 224, "gamma": 2.0, "mv": True,
     "label_smoothing": 0.1, "mixup": 0.2},
    {"key": "mv_effb3_attn_reg_e25",       "arch": "efficientnet_b3",
     "fusion": "attention", "epochs": 25, "lr": 1e-4, "batch": 8,
     "img": 224, "gamma": 2.5, "mv": True,
     "label_smoothing": 0.1, "mixup": 0.2},
    {"key": "mv_densenet121_concat_reg_e25", "arch": "densenet121",
     "fusion": "concat",    "epochs": 25, "lr": 1e-4, "batch": 8,
     "img": 224, "gamma": 2.0, "mv": True,
     "label_smoothing": 0.1, "mixup": 0.2},
]

# Phase 14: Cross-view transformer — spatial cross-attention between CC and MLO.
# CrossViewTransformerNet: shared spatial backbone + MHA cross-attention (49 tokens@224px).
# Runs sequentially (workers=1); each experiment needs ~6 GB VRAM at batch=6.
# Ordered: VinDr-only baseline → medical pretrained → +RSNA multi-dataset → medical+RSNA (best).
CROSSVIEW = [
    # Baseline: DenseNet-121 cross-view on VinDr only
    {"key": "cv_densenet121_e30", "arch": "densenet121",
     "epochs": 30, "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "crossview": True, "label_smoothing": 0.1, "mixup": 0.2},
    # Medical pretrained backbone (TorchXRayVision) + cross-view on VinDr
    {"key": "cv_densenet121_medical_e30", "arch": "densenet121",
     "epochs": 30, "lr": 3e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "crossview": True, "medical": True, "label_smoothing": 0.1, "mixup": 0.2},
    # Multi-dataset: VinDr + RSNA training split, standard backbone
    {"key": "cv_densenet121_rsna_e30", "arch": "densenet121",
     "epochs": 30, "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "crossview": True, "use_rsna": True, "label_smoothing": 0.1, "mixup": 0.2},
    # Best expected: medical pretrained + VinDr + RSNA (Mirai-style fine-tuning)
    {"key": "cv_densenet121_medical_rsna_e35", "arch": "densenet121",
     "epochs": 35, "lr": 3e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "crossview": True, "medical": True, "use_rsna": True,
     "label_smoothing": 0.1, "mixup": 0.2},
    # ConvNeXt-Base backbone (ImageNet-21k weights, stronger than ResNet/DenseNet)
    {"key": "cv_convnext_base_e30", "arch": "convnext_base",
     "epochs": 30, "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.0,
     "crossview": True, "label_smoothing": 0.1, "mixup": 0.2},
    # EfficientNet-B3 cross-view (1536-dim features, good balance of size/accuracy)
    {"key": "cv_effb3_e30", "arch": "efficientnet_b3",
     "epochs": 30, "lr": 5e-5, "batch": 6, "img": 224, "gamma": 2.5,
     "crossview": True, "label_smoothing": 0.1, "mixup": 0.2},
]

# Phase 8: Multi-view breast-pair fusion — novel contribution
# Each sample = (CC, MLO) pair for one breast; shared backbone + fusion head.
# batch=8 → 16 images in memory at once; fits comfortably in 12 GB VRAM.
# Run sequentially (workers=1) to avoid VRAM contention.
MULTIVIEW = [
    {"key": "mv_resnet50_concat_e25",       "arch": "resnet50",        "fusion": "concat",    "epochs": 25, "lr": 1e-4, "batch": 8, "img": 224, "gamma": 2.0, "mv": True},
    {"key": "mv_resnet50_attn_e25",         "arch": "resnet50",        "fusion": "attention", "epochs": 25, "lr": 1e-4, "batch": 8, "img": 224, "gamma": 2.0, "mv": True},
    # batch=2 (4 imgs at 384px full backbone); OOM at batch=4 on unfreeze — use gradient_checkpointing or just low batch
    {"key": "mv_resnet50_concat_384_e25",   "arch": "resnet50",        "fusion": "concat",    "epochs": 25, "lr": 1e-4, "batch": 2, "img": 384, "gamma": 2.0, "mv": True, "freeze": 1},
    {"key": "mv_effb3_attn_e25",            "arch": "efficientnet_b3", "fusion": "attention", "epochs": 25, "lr": 1e-4, "batch": 8, "img": 224, "gamma": 2.5, "mv": True},
    # DenseNet121 multiview — best single-view backbone (AUC 0.756)
    {"key": "mv_densenet121_concat_e25",    "arch": "densenet121",     "fusion": "concat",    "epochs": 25, "lr": 1e-4, "batch": 8, "img": 224, "gamma": 2.0, "mv": True},
    {"key": "mv_densenet121_attn_e25",      "arch": "densenet121",     "fusion": "attention", "epochs": 25, "lr": 1e-4, "batch": 8, "img": 224, "gamma": 2.0, "mv": True},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_completed() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text())["completed_experiments"])
    return set()


def best_cnn_arch() -> str:
    """Return the best arch from Phase 2 by val F1."""
    if not PROGRESS_FILE.exists():
        return "resnet50"
    data = json.loads(PROGRESS_FILE.read_text())
    best_key, best_f1 = "resnet50", 0.0
    for k, m in data.get("best_results", {}).items():
        if k.startswith("cnn_") and k.endswith("_e15"):
            f1 = m.get("f1_score", 0) or 0
            if f1 > best_f1:
                best_f1 = f1
                # extract arch from key like cnn_resnet50_e15
                best_key = k[4:-4]  # strip "cnn_" prefix and "_e15" suffix
    return best_key


def resolve_keys(exps: list, best: str) -> list:
    return [{**e, "key": e["key"].replace("{best}", best)} for e in exps]


def run_one(exp: dict) -> tuple[str, int]:
    key           = exp["key"]
    arch          = exp.get("arch") or best_cnn_arch()
    log_path      = LOG_DIR / f"{key}.log"
    is_dicom      = exp.get("dicom",     False)
    is_full       = exp.get("full",      False)
    is_mv         = exp.get("mv",        False)
    is_bilateral  = exp.get("bilateral", False)
    is_asymmetry  = exp.get("asymmetry", False)
    is_medical    = exp.get("medical",   False)
    is_crossview  = exp.get("crossview", False)

    if is_crossview:
        script = "scripts/run_single_crossview.py"
    elif is_medical:
        script = "scripts/run_single_medical.py"
    elif is_asymmetry:
        script = "scripts/run_single_asymmetry.py"
    elif is_bilateral:
        script = "scripts/run_single_bilateral.py"
    elif is_mv:
        script = "scripts/run_single_multiview.py"
    elif is_full:
        script = "scripts/run_single_full.py"
    elif is_dicom:
        script = "scripts/run_single_dicom.py"
    else:
        script = "scripts/run_single_cnn.py"

    cmd = [
        PYTHON, "-u", script,
        "--key",    key,
        "--arch",   arch,
        "--epochs", str(exp["epochs"]),
        "--lr",     str(exp["lr"]),
        "--batch",  str(exp["batch"]),
        "--img",    str(exp["img"]),
        "--gamma",  str(exp["gamma"]),
    ]
    if is_crossview:
        cmd += ["--label_smoothing", str(exp.get("label_smoothing", 0.1))]
        cmd += ["--mixup",           str(exp.get("mixup",           0.2))]
        if exp.get("medical"):
            cmd += ["--medical"]
        if exp.get("use_rsna"):
            cmd += ["--use_rsna"]
        if exp.get("amp"):
            cmd += ["--amp"]
    elif is_medical or is_asymmetry or is_bilateral:
        cmd += ["--label_smoothing", str(exp.get("label_smoothing", 0.1))]
        cmd += ["--mixup",           str(exp.get("mixup",           0.2))]
        if "freeze" in exp:
            cmd += ["--freeze", str(exp["freeze"])]
        if is_bilateral and "n_heads" in exp:
            cmd += ["--n_heads", str(exp["n_heads"])]
    elif is_mv:
        cmd += ["--fusion", exp.get("fusion", "concat")]
        if "freeze" in exp:
            cmd += ["--freeze", str(exp["freeze"])]
        if exp.get("amp"):
            cmd += ["--amp"]
        if "label_smoothing" in exp:
            cmd += ["--label_smoothing", str(exp["label_smoothing"])]
        if "mixup" in exp:
            cmd += ["--mixup", str(exp["mixup"])]
    elif not is_full:
        cmd += ["--prep", exp.get("prep", "clahe")]

    if is_crossview:
        tag   = "CROSSVIEW"
        extra = (f"arch={arch}  ls={exp.get('label_smoothing',0.1)}  "
                 f"{'med ' if exp.get('medical') else ''}"
                 f"{'rsna' if exp.get('use_rsna') else ''}")
    elif is_medical:
        tag   = "MEDICAL"
        extra = f"backbone=xrv-densenet121  label_smoothing={exp.get('label_smoothing', 0.1)}"
    elif is_asymmetry:
        tag   = "ASYMMETRY"
        extra = f"label_smoothing={exp.get('label_smoothing', 0.1)}  mixup={exp.get('mixup', 0.2)}"
    elif is_bilateral:
        tag   = "BILATERAL"
        extra = f"label_smoothing={exp.get('label_smoothing', 0.1)}  mixup={exp.get('mixup', 0.2)}"
    elif is_mv:
        tag   = "MV"
        ls = exp.get("label_smoothing", 0)
        extra = f"fusion={exp['fusion']}" + (f"  ls={ls}  mixup={exp.get('mixup',0)}" if ls else "")
    elif is_full:
        tag   = "FULL"
        extra = "prep=preprocessed-png"
    elif is_dicom:
        tag   = "DICOM"
        extra = f"prep={exp.get('prep', 'clahe')}"
    else:
        tag   = "CNN"
        extra = f"prep={exp.get('prep', 'clahe')}"
    ts  = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] LAUNCH [{tag}] {key}  (arch={arch}, epochs={exp['epochs']}, {extra})")

    with open(log_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), stdout=fout,
                              stderr=subprocess.STDOUT, env=ENV)
    ts2 = datetime.now().strftime("%H:%M:%S")
    try:
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        summary = lines[-1] if lines else "(no output)"
    except Exception:
        summary = ""
    print(f"[{ts2}] DONE   [{tag}] {key}  exit={proc.returncode}  {summary}")
    return key, proc.returncode


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3,
                        help="Max parallel experiments (default 3 for 12 GB VRAM)")
    parser.add_argument("--phase", choices=["baseline", "prep", "hyper", "advanced", "dicom",
                                            "full", "mv", "bilateral", "asymmetry", "medical",
                                            "ensemble", "hires", "improved_mv", "crossview", "all"],
                        default="baseline")
    args = parser.parse_args()

    completed = load_completed()
    best = best_cnn_arch()
    print(f"Best arch so far: {best}")
    print(f"Completed so far: {sorted(completed)}")

    # Build work queue
    if args.phase == "baseline":
        queue = BASELINE
    elif args.phase == "prep":
        queue = resolve_keys(PREPROCESSING, best)
    elif args.phase == "hyper":
        queue = resolve_keys(HYPERPARAM, best)
    elif args.phase == "advanced":
        queue = ADVANCED
    elif args.phase == "dicom":
        queue = DICOM
    elif args.phase == "full":
        queue = FULL
    elif args.phase == "mv":
        queue = MULTIVIEW
    elif args.phase == "bilateral":
        queue = BILATERAL
    elif args.phase == "asymmetry":
        queue = ASYMMETRY
    elif args.phase == "medical":
        queue = MEDICAL
    elif args.phase == "hires":
        queue = HIRES
    elif args.phase == "improved_mv":
        queue = IMPROVED_MV
    elif args.phase == "crossview":
        queue = CROSSVIEW
    elif args.phase == "ensemble":
        # Ensemble has no training — run directly and exit
        print("Running ensemble evaluation ...")
        import subprocess
        cmd = [PYTHON, "-u", "scripts/run_ensemble.py", "--method", "mean"]
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=ENV)
        cmd_w = [PYTHON, "-u", "scripts/run_ensemble.py", "--method", "weighted"]
        proc2 = subprocess.run(cmd_w, cwd=str(BASE_DIR), env=ENV)
        sys.exit(0 if (proc.returncode == 0 and proc2.returncode == 0) else 1)
    else:  # all
        queue = (BASELINE
                 + resolve_keys(PREPROCESSING, best)
                 + resolve_keys(HYPERPARAM, best)
                 + ADVANCED
                 + DICOM
                 + FULL
                 + MULTIVIEW
                 + BILATERAL
                 + ASYMMETRY
                 + MEDICAL
                 + HIRES
                 + IMPROVED_MV
                 + CROSSVIEW)

    pending = [e for e in queue if e["key"] not in completed]
    if not pending:
        print("Nothing pending — all experiments in this phase are done.")
        return

    print(f"\nRunning {len(pending)} experiments with --workers {args.workers}\n")
    for e in pending:
        print(f"  {e['key']}")
    print()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, exp): exp["key"] for exp in pending}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                k, code = fut.result()
                status = "OK" if code == 0 else f"FAILED (exit {code})"
                print(f"  >> {k}: {status}")
            except Exception as exc:
                print(f"  >> {key}: EXCEPTION {exc}")

    print("\n=== All done ===")
    completed2 = load_completed()
    print(f"Total completed: {len(completed2)}")
    # Exit 1 if any experiment in this run still didn't complete (triggers bat retry)
    still_pending = [e["key"] for e in pending if e["key"] not in completed2]
    if still_pending:
        print(f"  !! Still incomplete: {still_pending} — exiting 1 so bat retries")
        sys.exit(1)


if __name__ == "__main__":
    main()
