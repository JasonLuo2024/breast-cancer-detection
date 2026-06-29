"""
Asymmetry bilateral 4-view experiment entry point.
Called by run_parallel_cnn.py --phase asymmetry.

Architecture: AsymmetryNet — ipsilateral ipsi_proj + |h_L - h_R| diff head.
Simpler than BilateralFusionNet (no cross-attention), fewer params, less prone
to overfitting the limited ~385 positive patients.

Usage:
    python scripts/run_single_asymmetry.py \
        --key asym_densenet121_e30 --arch densenet121 \
        --epochs 30 --lr 5e-5 --batch 6 --img 224 \
        --gamma 2.0 --label_smoothing 0.1 --mixup 0.2
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

from tools.multiview_tools import run_asymmetry_experiment

PROGRESS_FILE = BASE_DIR / "data" / "progress.json"


def _mark_done(key: str, result: dict):
    try:
        data = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
        data.setdefault("completed_experiments", [])
        data.setdefault("best_results", {})
        if key not in data["completed_experiments"]:
            data["completed_experiments"].append(key)
        data["best_results"][key] = {
            "auc_roc":            result.get("best_val_auc"),
            "f1_score":           (result.get("final_metrics_opt_thresh") or {}).get("f1_score"),
            "sensitivity":        (result.get("final_metrics_thresh05")   or {}).get("sensitivity"),
            "sens_at_90pct_spec": (result.get("final_metrics_thresh05")   or {}).get("sens_at_90pct_spec"),
        }
        PROGRESS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[warn] Could not update progress.json: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key",             required=True)
    p.add_argument("--arch",            default="densenet121")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--batch",           type=int,   default=6)
    p.add_argument("--img",             type=int,   default=224)
    p.add_argument("--gamma",           type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup",           type=float, default=0.2)
    p.add_argument("--freeze",          type=int,   default=3)
    p.add_argument("--workers",         type=int,   default=6)
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"  ASYMMETRY BILATERAL: {args.key}")
    print(f"  arch={args.arch}  epochs={args.epochs}  lr={args.lr}")
    print(f"  batch={args.batch}  img={args.img}px  gamma={args.gamma}")
    print(f"  label_smoothing={args.label_smoothing}  mixup={args.mixup}")
    print(f"  freeze={args.freeze} epochs  [NO cross-attention]")
    print(f"{'='*65}\n")

    result = run_asymmetry_experiment(
        architecture         = args.arch,
        epochs               = args.epochs,
        batch_size           = args.batch,
        learning_rate        = args.lr,
        img_size             = args.img,
        focal_gamma          = args.gamma,
        label_smoothing      = args.label_smoothing,
        mixup_alpha          = args.mixup,
        medical_pretrained   = False,
        freeze_backbone_epochs = args.freeze,
        num_workers          = args.workers,
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    _mark_done(args.key, result)
    sys.exit(0)


if __name__ == "__main__":
    main()
