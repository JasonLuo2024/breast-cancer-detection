"""
Bootstrap 95% confidence intervals for AUC, F1, sensitivity, specificity,
and sensitivity@90%-specificity on the best model (DenseNet-121/concat).

Loads the test-set predictions saved by evaluate_uncertainty.py
(G:/breast-cancer-research/results/uncertainty_densenet121_concat_*.json)
and runs 2,000 bootstrap resamples.

Usage:
    python scripts/compute_bootstrap_ci.py
    python scripts/compute_bootstrap_ci.py --n_boot 5000
    python scripts/compute_bootstrap_ci.py --result_json G:/breast-cancer-research/results/uncertainty_densenet121_concat_20260508_070336.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── metrics ──────────────────────────────────────────────────────────────────

def _youden_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def _sens_at_90spec(labels, probs):
    """Sensitivity when specificity >= 0.90 (highest achievable)."""
    fpr, tpr, _ = roc_curve(labels, probs)
    spec = 1 - fpr
    mask = spec >= 0.90
    return float(tpr[mask].max()) if mask.any() else 0.0


def _compute_all(labels, probs):
    thresh = _youden_threshold(labels, probs)
    preds  = (np.array(probs) >= thresh).astype(int)
    labels = np.array(labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0
    auc  = float(roc_auc_score(labels, probs))
    s90  = _sens_at_90spec(labels, probs)

    return {"auc": auc, "f1": f1, "sens": sens, "spec": spec, "s90": s90}


def bootstrap_ci(labels, probs, n_boot=2000, alpha=0.05, seed=42):
    rng    = np.random.default_rng(seed)
    n      = len(labels)
    labels = np.array(labels)
    probs  = np.array(probs)

    boot = {k: [] for k in ["auc", "f1", "sens", "spec", "s90"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lb, pb = labels[idx], probs[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        m = _compute_all(lb.tolist(), pb.tolist())
        for k in boot:
            boot[k].append(m[k])

    lo, hi = alpha / 2, 1 - alpha / 2
    ci = {}
    for k, vals in boot.items():
        vals = np.array(vals)
        ci[k] = (float(np.percentile(vals, lo * 100)),
                 float(np.percentile(vals, hi * 100)))
    return ci


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not _HAS_SKLEARN:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", type=str, default=None,
                        help="Path to uncertainty_*.json with per-sample records")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--use_mc", action="store_true", default=True,
                        help="Use MC Dropout mean_prob (default) vs det_prob")
    args = parser.parse_args()

    # locate result JSON
    if args.result_json:
        result_path = Path(args.result_json)
    else:
        candidates = sorted(
            Path("G:/breast-cancer-research/results").glob("uncertainty_densenet121_concat_*.json"),
            key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            # fall back: look anywhere
            candidates = sorted(
                Path("G:/breast-cancer-research/results").glob("uncertainty_*.json"),
                key=lambda p: p.stat().st_mtime
            )
        if not candidates:
            print("ERROR: No uncertainty_*.json found. Run evaluate_uncertainty.py first.")
            sys.exit(1)
        result_path = candidates[-1]

    print(f"Loading records from: {result_path}")
    data    = json.loads(result_path.read_text())
    records = data.get("records", [])
    if not records:
        print("ERROR: 'records' key missing or empty in JSON.")
        sys.exit(1)

    labels = [r["label"]     for r in records]
    probs  = [r["mean_prob"] for r in records]  # MC mean probability

    print(f"  Total pairs   : {len(labels)}")
    print(f"  Positives     : {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Bootstrap runs: {args.n_boot}")
    print()

    # Point estimates
    point = _compute_all(labels, probs)
    print("Point estimates (MC Dropout mean_prob, Youden threshold):")
    print(f"  AUC  = {point['auc']:.4f}")
    print(f"  F1   = {point['f1']:.4f}")
    print(f"  Sens = {point['sens']:.4f}")
    print(f"  Spec = {point['spec']:.4f}")
    print(f"  S@90 = {point['s90']:.4f}")
    print()

    print(f"Running {args.n_boot} bootstrap resamples...")
    ci = bootstrap_ci(labels, probs, n_boot=args.n_boot)

    W = 6
    print(f"\n{'─'*55}")
    print(f"  {'Metric':<8} {'Point':>{W}} {'95% CI':>18}")
    print(f"{'─'*55}")
    for k in ["auc", "f1", "sens", "spec", "s90"]:
        lo, hi = ci[k]
        print(f"  {k.upper():<8} {point[k]:>{W}.4f}   [{lo:.4f}, {hi:.4f}]")
    print(f"{'─'*55}")
    print()

    # Save
    out = {
        "model":        data.get("model_path", str(result_path)),
        "architecture": data.get("architecture", "densenet121"),
        "fusion":       data.get("fusion",       "concat"),
        "n_pairs":      len(labels),
        "n_positive":   sum(labels),
        "n_bootstrap":  args.n_boot,
        "point_estimates": {k: round(point[k], 6) for k in point},
        "confidence_intervals_95pct": {k: [round(ci[k][0], 6), round(ci[k][1], 6)] for k in ci},
    }
    ts      = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = BASE_DIR / "data" / "reports" / f"bootstrap_ci_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved → {out_path}")

    # Print LaTeX-friendly summary line
    auc_lo, auc_hi = ci["auc"]
    s90_lo, s90_hi = ci["s90"]
    print(f"\nLaTeX-ready:")
    print(f"  AUC = {point['auc']:.4f} (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"  S@90 = {point['s90']:.4f} (95% CI: {s90_lo:.4f}–{s90_hi:.4f})")


if __name__ == "__main__":
    main()
