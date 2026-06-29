"""Run one multi-view breast-pair fusion experiment. Called by run_parallel_cnn.py --phase mv."""
import argparse, json, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

from tools.multiview_tools import run_multiview_experiment

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
    p.add_argument("--fusion",  required=True, choices=["concat", "attention"])
    p.add_argument("--epochs",  type=int,   default=25)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--img",     type=int,   default=224)
    p.add_argument("--gamma",           type=float, default=2.0)
    p.add_argument("--freeze",          type=int,   default=3)
    p.add_argument("--workers",         type=int,   default=6)
    p.add_argument("--amp",             action="store_true",
                   help="Enable AMP (FP16) mixed precision — required for 512px")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup",           type=float, default=0.0)
    args = p.parse_args()

    print(f"[{args.key}] MultiView: arch={args.arch} fusion={args.fusion} "
          f"epochs={args.epochs} batch={args.batch} img={args.img} lr={args.lr}"
          f"{' AMP' if args.amp else ''}"
          f"{f' ls={args.label_smoothing} mixup={args.mixup}' if args.label_smoothing else ''}")

    result = run_multiview_experiment(
        architecture=args.arch, fusion=args.fusion,
        epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, img_size=args.img,
        focal_gamma=args.gamma, pretrained=True,
        freeze_backbone_epochs=args.freeze,
        num_workers=args.workers,
        use_amp=args.amp,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup,
    )

    if "error" in result:
        print(f"[{args.key}] ERROR: {result['error']}")
        _log("experiment_error", {"key": args.key, "error": result["error"]})
        sys.exit(1)

    m = result.get("final_metrics_opt_thresh", {})
    print(f"[{args.key}] DONE  "
          f"AUC={result.get('best_val_auc',0):.4f}  "
          f"F1={m.get('f1_score',0):.4f}  "
          f"sens={m.get('sensitivity',0):.4f}  "
          f"s@90={m.get('sens_at_90pct_spec')}  "
          f"elapsed={result.get('elapsed_seconds',0):.0f}s")
    _log("multiview_experiment_done", {
        "key": args.key, "arch": args.arch, "fusion": args.fusion,
        "best_val_auc": result.get("best_val_auc"),
        "metrics": m, "elapsed": result.get("elapsed_seconds"),
    })
    _mark_done(args.key, m)


if __name__ == "__main__":
    main()
