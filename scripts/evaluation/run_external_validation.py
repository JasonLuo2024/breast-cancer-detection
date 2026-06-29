"""
External validation of trained VinDr models on the RSNA 2022 Breast Cancer Detection dataset.

Loads every mv_*.pt / asymmetry_*.pt / cv_*.pt / dualscale_*.pt model found in
LARGE_MODELS_DIR, evaluates each on the rsna_val split with TTA-5, then reports
an ensemble AUC with 95 % bootstrap confidence intervals.

Usage:
    python scripts/run_external_validation.py
    python scripts/run_external_validation.py --tta 5 --models mv_densenet121_attn_20260512*.pt
    python scripts/run_external_validation.py --single_model path/to/model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR
from tools.multiview_tools import (
    BreastPairNet, AsymmetryNet, PatientQuadDataset,
    _val_transform, _compute_metrics, find_optimal_threshold,
)
from tools.crossview_tools import (
    CrossViewTransformerNet, RsnaBreastPairDataset,
)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RSNA_MANIFEST = Path("G:/breast-cancer-research/rsna_manifest.csv")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(pt_path: Path):
    """Load any saved checkpoint and return (model, img_size, model_type, arch)."""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)

    # Infer model type: checkpoint key > filename prefix
    name = pt_path.stem
    if ckpt.get("model_type"):
        model_type = ckpt["model_type"]
    elif name.startswith("asymmetry_"):
        model_type = "asymmetry"
    elif name.startswith("cv_"):
        model_type = "crossview"
    else:
        model_type = "breastpair"

    arch    = ckpt.get("arch", "densenet121")
    fusion  = ckpt.get("fusion", "attention")
    img_size = ckpt.get("img_size", 224)
    d_model = ckpt.get("d_model", 512)
    n_heads = ckpt.get("n_heads", 8)

    if model_type == "crossview":
        model = CrossViewTransformerNet(arch, pretrained=False,
                                         d_model=d_model, n_heads=n_heads,
                                         medical=False)
    elif model_type == "asymmetry":
        model = AsymmetryNet(arch, pretrained=False)
    else:
        model = BreastPairNet(arch, fusion, pretrained=False)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE), img_size, model_type, arch


# ── TTA inference ─────────────────────────────────────────────────────────────

def predict_with_tta(model, loader, n_tta: int = 5, model_type: str = "breastpair"):
    """Horizontal-flip TTA inference. Returns (labels, probs)."""
    model.eval()
    all_labels, run_probs = [], [[] for _ in range(n_tta)]

    with torch.no_grad():
        for batch in loader:
            cc, mlo, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]
            all_labels.extend(labels.tolist())
            for run in range(n_tta):
                flip = torch.rand(1).item() > 0.5
                cc_f  = torch.flip(cc,  [3]) if flip else cc
                mlo_f = torch.flip(mlo, [3]) if flip else mlo
                p = torch.sigmoid(model(cc_f, mlo_f)).cpu()
                run_probs[run].extend(p.tolist())

    avg_probs = np.mean(run_probs, axis=0).tolist()
    return all_labels, avg_probs


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_auc_ci(labels, probs, n_boot: int = 1000, ci: float = 0.95):
    labels = np.array(labels)
    probs  = np.array(probs)

    def _auc(y, p):
        try:
            return roc_auc_score(y, p)
        except Exception:
            return float("nan")

    boot_aucs = []
    rng = np.random.default_rng(42)
    n   = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boot_aucs.append(_auc(labels[idx], probs[idx]))

    lo = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta",          type=int, default=5)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--n_boot",       type=int, default=1000)
    parser.add_argument("--models",       nargs="*",
                        help="Glob patterns for model files (relative to LARGE_MODELS_DIR)")
    parser.add_argument("--single_model", type=str, default=None,
                        help="Evaluate a single .pt file")
    args = parser.parse_args()

    if not RSNA_MANIFEST.exists():
        print(f"ERROR: RSNA manifest not found: {RSNA_MANIFEST}")
        print("Run scripts/preprocess_rsna.py first.")
        return

    rsna_df = pd.read_csv(RSNA_MANIFEST)
    print(f"RSNA manifest: {len(rsna_df)} pairs  "
          f"(cancer={rsna_df.cancer.sum()}, "
          f"val={(rsna_df.rsna_split=='rsna_val').sum()})")

    # Collect model paths
    if args.single_model:
        model_paths = [Path(args.single_model)]
    elif args.models:
        model_paths = []
        for pat in args.models:
            model_paths.extend(LARGE_MODELS_DIR.glob(pat))
    else:
        model_paths = (
            list(LARGE_MODELS_DIR.glob("mv_*.pt"))
            + list(LARGE_MODELS_DIR.glob("asymmetry_*.pt"))
            + list(LARGE_MODELS_DIR.glob("cv_*.pt"))
            + list(LARGE_MODELS_DIR.glob("dualscale_*.pt"))
        )

    model_paths = sorted(set(model_paths))
    print(f"\nModels to evaluate: {len(model_paths)}\n")

    per_model_probs: list[np.ndarray] = []
    rsna_labels: list[int] | None     = None
    model_records: list[dict]         = []

    for pt in tqdm(model_paths, desc="Evaluating models"):
        try:
            model, img_size, model_type, arch = load_model(pt)
        except Exception as e:
            print(f"  SKIP {pt.name}: {e}")
            continue

        # AsymmetryNet needs 4-view bilateral input — skip for breast-pair evaluation
        if model_type == "asymmetry":
            print(f"  SKIP {pt.name}: asymmetry model needs 4-view input (not supported on RSNA)")
            del model
            continue

        val_ds = RsnaBreastPairDataset(rsna_df, "rsna_val",
                                        transform=_val_transform(img_size))
        if len(val_ds) == 0:
            print(f"  SKIP {pt.name}: no RSNA val pairs found (PNGs not ready?)")
            continue

        loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

        labels, probs = predict_with_tta(model, loader, args.tta, model_type)
        if rsna_labels is None:
            rsna_labels = labels
        else:
            assert labels == rsna_labels, "Label mismatch across models"

        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float("nan")

        m = _compute_metrics(labels, probs, find_optimal_threshold(labels, probs))
        lo, hi = bootstrap_auc_ci(labels, probs, args.n_boot)

        print(f"  {pt.name:<55} AUC={auc:.4f} [{lo:.4f}-{hi:.4f}]  "
              f"s@90={m.get('sens_at_90pct_spec')}  arch={arch}")
        per_model_probs.append(np.array(probs))
        model_records.append({
            "model": pt.name,
            "arch": arch,
            "model_type": model_type,
            "auc": round(auc, 4),
            "auc_ci_95_lo": round(lo, 4),
            "auc_ci_95_hi": round(hi, 4),
            "metrics_opt": m,
        })

        del model
        torch.cuda.empty_cache()

    if not per_model_probs or rsna_labels is None:
        print("\nNo models evaluated. Check paths and RSNA manifest.")
        return

    print(f"\n{'='*70}")
    print(f"  RSNA External Validation — {len(per_model_probs)} models, TTA={args.tta}")

    ensemble_probs = np.mean(per_model_probs, axis=0)
    ens_auc        = roc_auc_score(rsna_labels, ensemble_probs)
    ens_lo, ens_hi = bootstrap_auc_ci(rsna_labels, ensemble_probs.tolist(), args.n_boot)
    thresh         = find_optimal_threshold(rsna_labels, ensemble_probs.tolist())
    ens_m          = _compute_metrics(rsna_labels, ensemble_probs.tolist(), thresh)

    print(f"  Ensemble AUC = {ens_auc:.4f}  95%CI [{ens_lo:.4f}-{ens_hi:.4f}]")
    print(f"  F1@opt={ens_m['f1_score']:.4f} (t={thresh:.3f})  "
          f"sens@opt={ens_m['sensitivity']:.4f}  "
          f"s@90={ens_m['sens_at_90pct_spec']}")
    print(f"{'='*70}\n")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LARGE_RESULTS_DIR / f"rsna_external_val_{ts}.json"
    report = {
        "dataset":         "RSNA-2022-Breast-Cancer",
        "n_val_pairs":     len(rsna_labels),
        "n_cancer":        int(sum(rsna_labels)),
        "n_models":        len(per_model_probs),
        "tta":             args.tta,
        "ensemble_auc":    round(ens_auc, 4),
        "ensemble_auc_ci": [round(ens_lo, 4), round(ens_hi, 4)],
        "ensemble_metrics_opt": ens_m,
        "per_model":       model_records,
        "timestamp":       datetime.now().isoformat(),
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
