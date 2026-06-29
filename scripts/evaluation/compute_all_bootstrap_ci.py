"""
Compute 95% bootstrap CIs for ALL saved pair-level checkpoints on the
VinDr-Mammo test split (2000 breast pairs, 99 positives).

Handles: BreastPairNet (mv_*.pt), CrossView Transformer (cv_*.pt)
Also computes CIs for the saved ensemble predictions.

Outputs:
  data/reports/bootstrap_ci_all_<ts>.json   — full per-model CI table
  (prints LaTeX-ready table to stdout)

Usage:
    python scripts/compute_all_bootstrap_ci.py
    python scripts/compute_all_bootstrap_ci.py --n_boot 5000
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR
from tools.multiview_tools import (
    BreastPairNet, AsymmetryNet, BreastPairDataset,
    _val_transform, _build_backbone,
)
from tools.crossview_tools import CrossViewTransformerNet

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV = Path("G:/breast-cancer-research/split_manifest.csv")


# ── metrics ───────────────────────────────────────────────────────────────────

def _youden_threshold(labels, probs):
    fpr, tpr, thresh = roc_curve(labels, probs)
    return float(thresh[np.argmax(tpr - fpr)])


def _sens_at_90spec(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    mask = (1 - fpr) >= 0.90
    return float(tpr[mask].max()) if mask.any() else 0.0


def _point(labels, probs):
    labels = np.array(labels)
    probs  = np.array(probs)
    t      = _youden_threshold(labels, probs)
    preds  = (probs >= t).astype(int)
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    return {
        "auc":  float(roc_auc_score(labels, probs)),
        "f1":   f1,
        "sens": sens,
        "spec": spec,
        "s90":  _sens_at_90spec(labels, probs),
        "thresh": float(t),
    }


def bootstrap_ci(labels, probs, n_boot=2000, seed=42):
    rng    = np.random.default_rng(seed)
    labels = np.array(labels)
    probs  = np.array(probs)
    n      = len(labels)
    boot   = {k: [] for k in ["auc", "f1", "sens", "spec", "s90"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lb, pb = labels[idx], probs[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        m = _point(lb.tolist(), pb.tolist())
        for k in boot:
            boot[k].append(m[k])
    ci = {}
    for k, vals in boot.items():
        vals = np.array(vals)
        ci[k] = (round(float(np.percentile(vals, 2.5)), 4),
                 round(float(np.percentile(vals, 97.5)), 4))
    return ci


# ── model loading ─────────────────────────────────────────────────────────────

class _LegacyBreastPairNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained=False)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feat_dim * 2, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1),
        )
    def forward(self, cc, mlo):
        return self.head(torch.cat([self.backbone(cc), self.backbone(mlo)], 1)).squeeze(1)


def load_model(pt_path: Path):
    ckpt     = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd       = ckpt.get("state_dict", ckpt)
    arch     = ckpt.get("arch",       "densenet121")
    img_size = ckpt.get("img_size",   224)
    fusion   = ckpt.get("fusion",     "attention")
    d_model  = ckpt.get("d_model",    512)
    n_heads  = ckpt.get("n_heads",    8)
    medical  = ckpt.get("medical",    False)
    name     = pt_path.stem

    mtype = ckpt.get("model_type")
    if not mtype:
        if name.startswith("cv_"):
            mtype = "crossview"
        elif name.startswith("asymmetry_"):
            mtype = "asymmetry"
        else:
            mtype = "breastpair"

    if mtype == "crossview":
        model = CrossViewTransformerNet(arch, pretrained=False,
                                        d_model=d_model, n_heads=n_heads,
                                        medical=medical)
    elif mtype == "asymmetry":
        return None, mtype, arch, img_size  # skip — needs 4-view
    else:
        is_legacy = (fusion == "concat"
                     and "head.4.weight" in sd
                     and sd["head.4.weight"].shape[0] == 1)
        if is_legacy:
            model = _LegacyBreastPairNet(arch)
        else:
            model = BreastPairNet(arch, fusion, pretrained=False)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE), mtype, arch, img_size


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, manifest, img_size, batch=8, workers=4):
    ds     = BreastPairDataset(manifest, "test", _val_transform(img_size))
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
    labels_all, probs_all = [], []
    for cc, mlo, lbl in loader:
        cc, mlo = cc.to(DEVICE), mlo.to(DEVICE)
        logits  = model(cc, mlo)
        probs   = torch.sigmoid(logits).cpu().numpy().tolist()
        probs_all.extend(probs)
        labels_all.extend(lbl.tolist())
    return labels_all, probs_all


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--workers",type=int, default=4)
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found: {MANIFEST_CSV}"); sys.exit(1)
    manifest = pd.read_csv(MANIFEST_CSV)

    # Collect checkpoints (mv_* + cv_*, skip asymmetry_*)
    ckpt_paths = sorted(LARGE_MODELS_DIR.glob("mv_*.pt")) + \
                 sorted(LARGE_MODELS_DIR.glob("cv_*.pt"))

    print(f"\nBootstrap CI computation  (n_boot={args.n_boot}, test_pairs=2000)")
    print(f"Device: {DEVICE}  |  Checkpoints: {len(ckpt_paths)}\n")

    rows = []
    for pt in ckpt_paths:
        model, mtype, arch, img_size = load_model(pt)
        if model is None:
            print(f"  SKIP {pt.name} (asymmetry — needs 4-view)")
            continue

        print(f"  {pt.name} ({mtype}/{arch}/{img_size}px)  ", end="", flush=True)
        try:
            labels, probs = run_inference(model, manifest, img_size,
                                          args.batch, args.workers)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        finally:
            del model
            torch.cuda.empty_cache()

        pt_m = _point(labels, probs)
        ci   = bootstrap_ci(labels, probs, args.n_boot)

        print(f"AUC={pt_m['auc']:.4f} [{ci['auc'][0]:.4f}–{ci['auc'][1]:.4f}]  "
              f"S@90={pt_m['s90']:.4f} [{ci['s90'][0]:.4f}–{ci['s90'][1]:.4f}]")

        rows.append({
            "checkpoint": pt.name,
            "model_type": mtype,
            "arch":       arch,
            "img_size":   img_size,
            "n_pairs":    len(labels),
            "n_pos":      int(sum(labels)),
            "n_boot":     args.n_boot,
            "point":      {k: round(v, 6) for k, v in pt_m.items()},
            "ci_95":      {k: list(v) for k, v in ci.items()},
        })

    # Also compute CI for the saved ensemble predictions
    ens_json = sorted(LARGE_RESULTS_DIR.glob("ensemble_mean_top16_+asym_tta5_*.json"))
    if ens_json:
        ens_path = ens_json[-1]
        print(f"\n  Ensemble ({ens_path.name})  ", end="", flush=True)
        ens_data = json.loads(ens_path.read_text())
        # Ensemble JSON doesn't store raw preds — report from paper values with note
        print("(raw predictions not stored — CI must be computed from re-run)")

    # Summary table
    print(f"\n{'═'*90}")
    print(f"  {'Checkpoint':<50} {'AUC':>6}  {'95% CI AUC':>18}  {'S@90':>6}  {'95% CI S@90':>18}")
    print(f"{'═'*90}")
    for r in rows:
        lo, hi   = r["ci_95"]["auc"]
        s90, s90lo, s90hi = r["point"]["s90"], r["ci_95"]["s90"][0], r["ci_95"]["s90"][1]
        print(f"  {r['checkpoint']:<50} {r['point']['auc']:>6.4f}  [{lo:.4f}–{hi:.4f}]"
              f"  {s90:>6.4f}  [{s90lo:.4f}–{s90hi:.4f}]")
    print(f"{'═'*90}\n")

    # Save
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out     = BASE_DIR / "data" / "reports" / f"bootstrap_ci_all_{ts}.json"
    out.write_text(json.dumps({"generated": ts, "n_boot": args.n_boot, "rows": rows}, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
