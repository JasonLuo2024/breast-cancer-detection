"""Run one full-dataset CNN experiment. Called by run_parallel_cnn.py --phase full*."""
import argparse, json, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

from tools.full_dataset_tools import run_full_cnn_experiment

PROGRESS_FILE = BASE_DIR / "data" / "progress.json"
LOG_FILE      = BASE_DIR / "data" / "autonomous_log.jsonl"


def _log(event, data):
    from datetime import datetime
    entry = {"timestamp": datetime.now().isoformat(), "event": event, **data}
    for _ in range(5):
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            break
        except OSError:
            time.sleep(0.2)


def _mark_done(key, metrics):
    for _ in range(10):
        try:
            d = json.loads(PROGRESS_FILE.read_text())
            if key not in d["completed_experiments"]:
                d["completed_experiments"].append(key)
                d["best_results"][key] = metrics
                PROGRESS_FILE.write_text(json.dumps(d, indent=2))
            return
        except (OSError, json.JSONDecodeError):
            time.sleep(0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key",     required=True)
    p.add_argument("--arch",    required=True)
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--img",     type=int,   default=224)
    p.add_argument("--gamma",   type=float, default=2.0)
    p.add_argument("--freeze",  type=int,   default=3)
    p.add_argument("--workers", type=int,   default=8)
    args = p.parse_args()

    print(f"[{args.key}] FullCNN: arch={args.arch} epochs={args.epochs} "
          f"batch={args.batch} img={args.img} lr={args.lr}")

    result = run_full_cnn_experiment(
        architecture=args.arch, epochs=args.epochs,
        batch_size=args.batch, learning_rate=args.lr,
        img_size=args.img, focal_gamma=args.gamma,
        pretrained=True, freeze_backbone_epochs=args.freeze,
        num_workers=args.workers,
    )

    if "error" in result:
        print(f"[{args.key}] ERROR: {result['error']}")
        _log("experiment_error", {"key": args.key, "error": result["error"]})
        sys.exit(1)

    m = result.get("final_metrics", {})
    print(f"[{args.key}] DONE  F1={m.get('f1_score',0):.4f} "
          f"AUC={m.get('auc_roc',0):.4f} sens={m.get('sensitivity',0):.4f} "
          f"s@90={m.get('sens_at_90pct_spec',0)} elapsed={result.get('elapsed_seconds',0):.0f}s")
    _log("cnn_experiment_done", {"key": args.key, "arch": args.arch,
                                  "dataset": "VinDr-Full-20k",
                                  "best_val_f1": result.get("best_val_f1"),
                                  "metrics": m, "elapsed": result.get("elapsed_seconds")})
    _mark_done(args.key, m)


if __name__ == "__main__":
    main()
