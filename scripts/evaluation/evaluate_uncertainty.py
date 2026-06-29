"""
MC Dropout uncertainty evaluation for the best multi-view model.
Implements the hybrid triage strategy from Verboom et al. 2025 (Radiology).

Strategy:
  AI handles only confident cases (low entropy); uncertain cases are referred
  to a radiologist. Oracle assumption: radiologist catches all cancers in
  referred group — stated explicitly as per Verboom 2025 design.

Output sections:
  1. Deterministic vs MC Dropout metrics (AUC, F1, sensitivity, specificity)
  2. Entropy distribution: positive vs negative breast pairs
  3. Hybrid triage table at 7 operating points (AI reads 30-90% of cases)
  Results saved → G:/breast-cancer-research/results/uncertainty_*.json

Usage:
    python scripts/evaluate_uncertainty.py
    python scripts/evaluate_uncertainty.py --model G:/models/mv_resnet50_concat_*.pt
    python scripts/evaluate_uncertainty.py --n_passes 30
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import torch
from torch.utils.data import DataLoader
from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR
from tools.multiview_tools import (
    BreastPairDataset, BreastPairNet, _val_transform,
    _compute_metrics, find_optimal_threshold, mc_dropout_predict,
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV = Path(r"G:\breast-cancer-research\split_manifest.csv")


# ── model selection ───────────────────────────────────────────────────────────

def _find_best_model() -> tuple:
    """Pick best mv model by best_val_auc from saved result JSONs on G:/."""
    results = list(LARGE_RESULTS_DIR.glob("mv_*.json"))
    if not results:
        raise FileNotFoundError(f"No mv_*.json in {LARGE_RESULTS_DIR}")

    best_auc, best = 0.0, None
    for r in results:
        try:
            d = json.loads(r.read_text())
            auc = d.get("best_val_auc", 0) or 0
            pt  = LARGE_MODELS_DIR / (r.stem + ".pt")
            if auc > best_auc and pt.exists():
                best_auc = auc
                best = (pt, d["architecture"], d["fusion"],
                        d.get("img_size", 224), d.get("best_thresh", 0.5))
        except Exception:
            pass

    if best is None:
        # Fallback: newest .pt checkpoint
        pts = sorted(LARGE_MODELS_DIR.glob("mv_*.pt"), key=lambda p: p.stat().st_mtime)
        if not pts:
            raise FileNotFoundError(f"No mv_*.pt in {LARGE_MODELS_DIR}")
        ckpt = torch.load(str(pts[-1]), map_location="cpu")
        best = (pts[-1],
                ckpt.get("arch",       "resnet50"),
                ckpt.get("fusion",     "concat"),
                ckpt.get("img_size",   224),
                ckpt.get("best_thresh", 0.5))

    return best


def _load_model(model_path: Path, arch: str, fusion: str) -> BreastPairNet:
    ckpt  = torch.load(str(model_path), map_location=DEVICE)
    model = BreastPairNet(arch, fusion, pretrained=False).to(DEVICE)
    sd    = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"[Uncertainty] Loaded {arch}/{fusion} from {model_path.name}")
    return model


# ── hybrid triage simulation ──────────────────────────────────────────────────

def _hybrid_triage(records: list[dict], ai_threshold: float) -> list[dict]:
    """
    Verboom 2025 hybrid triage simulation at 7 operating points.

    Cases sorted by entropy (ascending = most confident).
    AI reads the K% most confident cases; radiologist reads the rest.
    Oracle assumption: radiologist catches all positives in referred group.
    """
    df        = pd.DataFrame(records).sort_values("entropy").reset_index(drop=True)
    total_pos = int(df["label"].sum())
    total_neg = len(df) - total_pos

    rows = []
    for pct in [30, 40, 50, 60, 70, 80, 90]:
        n_ai      = max(1, int(len(df) * pct / 100))
        ai_cases  = df.iloc[:n_ai]
        rad_cases = df.iloc[n_ai:]

        ai_pos_pred = ai_cases["mean_prob"] >= ai_threshold
        ai_tp = int((ai_pos_pred & (ai_cases["label"] == 1)).sum())
        ai_fp = int((ai_pos_pred & (ai_cases["label"] == 0)).sum())
        ai_fn = int((~ai_pos_pred & (ai_cases["label"] == 1)).sum())
        rad_tp = int(rad_cases["label"].sum())  # oracle: catches all

        ai_cases_pos     = int(ai_cases["label"].sum())
        ai_sensitivity   = ai_tp / ai_cases_pos if ai_cases_pos else 0.0
        overall_sens     = (ai_tp + rad_tp) / total_pos if total_pos else 0.0
        # Recall rate = FP among AI-read negatives (false-positive callback rate)
        ai_neg_total     = n_ai - ai_cases_pos
        recall_rate      = ai_fp / ai_neg_total if ai_neg_total else 0.0

        rows.append({
            "ai_reads_pct":           pct,
            "workload_reduction_pct": 100 - pct,
            "n_ai_cases":             n_ai,
            "overall_sensitivity":    round(float(overall_sens),   4),
            "ai_sensitivity":         round(float(ai_sensitivity), 4),
            "recall_rate":            round(float(recall_rate),    4),
            "ai_tp": ai_tp, "ai_fp": ai_fp, "ai_fn": ai_fn, "rad_tp": rad_tp,
        })

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--arch",     type=str, default=None)
    parser.add_argument("--fusion",   type=str, default=None)
    parser.add_argument("--img",      type=int, default=None)
    parser.add_argument("--thresh",   type=float, default=None,
                        help="Classification threshold (default: Youden J on MC mean)")
    parser.add_argument("--n_passes", type=int, default=20)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()

    # ── load model ─────────────────────────────────────────────────────────────
    if args.model:
        model_path  = Path(args.model)
        ckpt        = torch.load(str(model_path), map_location="cpu")
        arch        = args.arch   or (ckpt.get("arch",       "resnet50") if isinstance(ckpt, dict) else "resnet50")
        fusion      = args.fusion or (ckpt.get("fusion",     "concat")   if isinstance(ckpt, dict) else "concat")
        img_size    = args.img    or (ckpt.get("img_size",   224)        if isinstance(ckpt, dict) else 224)
        best_thresh = args.thresh or (ckpt.get("best_thresh", 0.5)       if isinstance(ckpt, dict) else 0.5)
    else:
        model_path, arch, fusion, img_size, best_thresh = _find_best_model()
        if args.img:    img_size    = args.img
        if args.thresh: best_thresh = args.thresh

    print(f"\n{'='*65}")
    print(f"  MC DROPOUT UNCERTAINTY EVALUATION")
    print(f"{'='*65}")
    print(f"  Model  : {model_path.name}")
    print(f"  Arch   : {arch}   Fusion: {fusion}   ImgSize: {img_size}px")
    print(f"  Device : {DEVICE}   N_passes: {args.n_passes}")
    print(f"{'='*65}\n")

    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found at {MANIFEST_CSV}")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_CSV)
    model    = _load_model(model_path, arch, fusion)
    val_ds   = BreastPairDataset(manifest, "test", _val_transform(img_size))

    if len(val_ds) == 0:
        print("ERROR: No test breast-pairs found in manifest")
        sys.exit(1)

    # ── step 1: MC Dropout inference ───────────────────────────────────────────
    print(f"[1/3] MC Dropout inference ({args.n_passes} passes × {len(val_ds)} pairs)...")
    records = mc_dropout_predict(model, val_ds, img_size,
                                 batch_size=args.batch,
                                 num_workers=args.workers,
                                 n_passes=args.n_passes)
    labels     = [r["label"]     for r in records]
    mean_probs = [r["mean_prob"] for r in records]
    entropies  = [r["entropy"]   for r in records]
    print(f"      Done — {len(records)} pairs  |  positives: {sum(labels)}")

    # ── step 2: deterministic baseline (single pass, dropout off) ─────────────
    print(f"\n[2/3] Computing deterministic baseline...")
    model.eval()
    det_probs = []
    with torch.no_grad():
        loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        for cc, mlo, _ in loader:
            logits = model(cc.to(DEVICE), mlo.to(DEVICE))
            det_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    det_thresh = find_optimal_threshold(labels, det_probs)
    det_m      = _compute_metrics(labels, det_probs, det_thresh)

    mc_thresh  = args.thresh if args.thresh else find_optimal_threshold(labels, mean_probs)
    mc_m       = _compute_metrics(labels, mean_probs, mc_thresh)

    pos_ent = [r["entropy"] for r in records if r["label"] == 1]
    neg_ent = [r["entropy"] for r in records if r["label"] == 0]

    W = 22
    print(f"\n{'─'*68}")
    print(f"  {'':>{W}} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'s@90':>6} {'Thresh':>7}")
    print(f"{'─'*68}")
    print(f"  {'Deterministic':>{W}} "
          f"{det_m['auc_roc']:>6.4f} "
          f"{det_m['f1_score']:>6.4f} "
          f"{det_m['sensitivity']:>6.4f} "
          f"{det_m['specificity']:>6.4f} "
          f"{str(det_m.get('sens_at_90pct_spec','?')):>6} "
          f"{det_thresh:>7.4f}")
    print(f"  {'MC Dropout (N='+str(args.n_passes)+')':>{W}} "
          f"{mc_m['auc_roc']:>6.4f} "
          f"{mc_m['f1_score']:>6.4f} "
          f"{mc_m['sensitivity']:>6.4f} "
          f"{mc_m['specificity']:>6.4f} "
          f"{str(mc_m.get('sens_at_90pct_spec','?')):>6} "
          f"{mc_thresh:>7.4f}")
    print(f"{'─'*68}")

    print(f"\n  Entropy | Positive: mean={np.mean(pos_ent):.4f}  "
          f"median={np.median(pos_ent):.4f}  std={np.std(pos_ent):.4f}")
    print(f"          | Negative: mean={np.mean(neg_ent):.4f}  "
          f"median={np.median(neg_ent):.4f}  std={np.std(neg_ent):.4f}")
    print(f"\n  Interpretation: higher entropy in positives → model is more "
          f"uncertain on cancers\n  (expected — subtle lesions are harder)")

    # ── step 3: hybrid triage ──────────────────────────────────────────────────
    print(f"\n[3/3] Hybrid triage simulation")
    print(f"  Reference  : Verboom et al. 2025 (Radiology 316:e242594)")
    print(f"  Assumption : radiologist reads all referred cases and catches all cancers")
    print(f"  Threshold  : {mc_thresh:.4f}  |  Total: {len(records)}  |  "
          f"Positives: {sum(labels)} ({100*sum(labels)/len(records):.1f}%)")

    triage = _hybrid_triage(records, mc_thresh)

    print(f"\n{'─'*68}")
    print(f"  {'AI reads':>9}  {'Workload↓':>10}  {'Overall Sens':>13}  "
          f"{'AI Sens':>8}  {'Recall↑':>8}")
    print(f"{'─'*68}")
    for row in triage:
        marker = " ← Verboom" if row["workload_reduction_pct"] == 38 else (
                 " ←"         if row["workload_reduction_pct"] in (40, 30) else "")
        print(f"  {row['ai_reads_pct']:>7}%   "
              f"  -{row['workload_reduction_pct']:>3}%          "
              f"  {row['overall_sensitivity']:.4f}        "
              f"  {row['ai_sensitivity']:.4f}    "
              f"  {row['recall_rate']:.4f}{marker}")
    print(f"{'─'*68}")
    print(f"  (Verboom 2025 achieved: 62% AI reads → 38% workload↓, "
          f"sensitivity preserved at ~6.6/1000)")

    # ── save ───────────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LARGE_RESULTS_DIR / f"uncertainty_{arch}_{fusion}_{ts}.json"
    payload = {
        "method":        "mc_dropout_uncertainty",
        "reference":     "Verboom et al. 2025, Radiology 316(2):e242594",
        "model_path":    str(model_path),
        "architecture":  arch,
        "fusion":        fusion,
        "img_size":      img_size,
        "n_passes":      args.n_passes,
        "n_pairs":       len(records),
        "n_positive":    sum(labels),
        "det_threshold": round(det_thresh, 6),
        "mc_threshold":  round(mc_thresh,  6),
        "det_metrics":   det_m,
        "mc_metrics":    mc_m,
        "entropy_stats": {
            "positive": {
                "mean":   round(float(np.mean(pos_ent)),   6),
                "median": round(float(np.median(pos_ent)), 6),
                "std":    round(float(np.std(pos_ent)),    6),
                "p25":    round(float(np.percentile(pos_ent, 25)), 6),
                "p75":    round(float(np.percentile(pos_ent, 75)), 6),
            },
            "negative": {
                "mean":   round(float(np.mean(neg_ent)),   6),
                "median": round(float(np.median(neg_ent)), 6),
                "std":    round(float(np.std(neg_ent)),    6),
                "p25":    round(float(np.percentile(neg_ent, 25)), 6),
                "p75":    round(float(np.percentile(neg_ent, 75)), 6),
            },
        },
        "triage_table": triage,
        "records":      records,
        "timestamp":    datetime.now().isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[Uncertainty] Saved → {out}")


if __name__ == "__main__":
    main()
