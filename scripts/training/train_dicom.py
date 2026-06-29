"""
Run one DICOM-native CNN experiment. Called by run_parallel_cnn.py --phase dicom.
Writes metrics to progress.json on success.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

from tools.dicom_cnn_tools import run_dicom_cnn_experiment

PROGRESS_FILE = BASE_DIR / "data" / "progress.json"
LOG_FILE = BASE_DIR / "data" / "autonomous_log.jsonl"


def _log(event: str, data: dict):
    from datetime import datetime
    entry = {"timestamp": datetime.now().isoformat(), "event": event, **data}
    for _ in range(5):
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            break
        except OSError:
            time.sleep(0.2)


def _mark_done(key: str, metrics: dict):
    for _ in range(10):
        try:
            data = json.loads(PROGRESS_FILE.read_text())
            if key not in data["completed_experiments"]:
                data["completed_experiments"].append(key)
                data["best_results"][key] = metrics
                PROGRESS_FILE.write_text(json.dumps(data, indent=2))
            return
        except (OSError, json.JSONDecodeError):
            time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",      required=True)
    parser.add_argument("--arch",     required=True)
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--img",      type=int,   default=224)
    parser.add_argument("--gamma",    type=float, default=2.0)
    parser.add_argument("--prep",     default="clahe")
    parser.add_argument("--freeze",   type=int,   default=3)
    parser.add_argument("--workers",  type=int,   default=4)
    args = parser.parse_args()

    print(f"[{args.key}] DICOM-CNN: arch={args.arch} epochs={args.epochs} "
          f"prep={args.prep} lr={args.lr} batch={args.batch} img={args.img}")

    result = run_dicom_cnn_experiment(
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        img_size=args.img,
        focal_gamma=args.gamma,
        pretrained=True,
        freeze_backbone_epochs=args.freeze,
        preprocessing=args.prep,
        num_workers=args.workers,
    )

    if "error" in result:
        print(f"[{args.key}] ERROR: {result['error']}")
        _log("experiment_error", {"key": args.key, "arch": args.arch, "error": result["error"]})
        sys.exit(1)

    metrics = result.get("final_metrics", {})
    print(f"[{args.key}] DONE  F1={metrics.get('f1_score', 0):.4f} "
          f"AUC={metrics.get('auc_roc', 0):.4f} "
          f"sens={metrics.get('sensitivity', 0):.4f} "
          f"s@90={metrics.get('sens_at_90pct_spec', 0)} "
          f"elapsed={result.get('elapsed_seconds', 0):.0f}s")

    _log("cnn_experiment_done", {
        "key": args.key,
        "arch": args.arch,
        "dataset": "VinDr-Mammo-DICOM",
        "best_val_f1": result.get("best_val_f1"),
        "metrics": metrics,
        "elapsed": result.get("elapsed_seconds"),
    })
    _mark_done(args.key, metrics)


if __name__ == "__main__":
    main()
