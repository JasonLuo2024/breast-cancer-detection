"""
Run one CNN experiment from the command line. Called by run_parallel_cnn.py.
Writes results and updates progress.json on success.
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

from tools.cnn_tools import run_cnn_experiment

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
    parser.add_argument("--key",    required=True)
    parser.add_argument("--arch",   required=True)
    parser.add_argument("--epochs", type=int,   default=15)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--img",    type=int,   default=224)
    parser.add_argument("--gamma",  type=float, default=2.0)
    parser.add_argument("--prep",   default="standard")
    parser.add_argument("--freeze", type=int,   default=3)
    args = parser.parse_args()

    print(f"[{args.key}] Starting: arch={args.arch} epochs={args.epochs} "
          f"prep={args.prep} lr={args.lr} batch={args.batch} img={args.img}")

    result = run_cnn_experiment(
        architecture=args.arch,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        img_size=args.img,
        focal_gamma=args.gamma,
        pretrained=True,
        freeze_backbone_epochs=args.freeze,
        preprocessing=args.prep,
    )

    if "error" in result:
        print(f"[{args.key}] ERROR: {result['error']}")
        _log("experiment_error", {"key": args.key, "arch": args.arch, "error": result["error"]})
        sys.exit(1)

    metrics = result.get("final_metrics", {})
    print(f"[{args.key}] DONE  F1={metrics.get('f1_score', 0):.4f} "
          f"AUC={metrics.get('auc_roc', 0):.4f} "
          f"sens={metrics.get('sensitivity', 0):.4f} "
          f"elapsed={result.get('elapsed_seconds', 0):.0f}s")

    _log("cnn_experiment_done", {
        "key": args.key,
        "arch": args.arch,
        "best_val_f1": result.get("best_val_f1"),
        "metrics": metrics,
        "elapsed": result.get("elapsed_seconds"),
    })
    _mark_done(args.key, metrics)


if __name__ == "__main__":
    main()
