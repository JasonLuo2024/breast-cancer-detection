"""Run one CrossViewTransformerNet experiment. Called by run_parallel_cnn.py --phase crossview."""
import argparse, json, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from tools.crossview_tools import run_crossview_experiment

PROGRESS_FILE = BASE_DIR / "data" / "progress.json"


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
    p.add_argument("--key",             required=True)
    p.add_argument("--arch",            default="densenet121")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--batch",           type=int,   default=6)
    p.add_argument("--img",             type=int,   default=224)
    p.add_argument("--gamma",           type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup",           type=float, default=0.2)
    p.add_argument("--d_model",         type=int,   default=512)
    p.add_argument("--n_heads",         type=int,   default=8)
    p.add_argument("--freeze",          type=int,   default=3)
    p.add_argument("--workers",         type=int,   default=6)
    p.add_argument("--amp",             action="store_true")
    p.add_argument("--medical",         action="store_true",
                   help="Use TorchXRayVision medical-pretrained DenseNet-121 backbone")
    p.add_argument("--use_rsna",        action="store_true",
                   help="Include RSNA training split in combined training set")
    p.add_argument("--annotated",       action="store_true",
                   help="Use YOLO/GT bounding-box annotated PNGs (annotated manifests)")
    args = p.parse_args()

    print(f"[{args.key}] CrossView: arch={args.arch} "
          f"epochs={args.epochs} batch={args.batch} img={args.img} lr={args.lr} "
          f"ls={args.label_smoothing} mixup={args.mixup}"
          f"{' [medical]' if args.medical else ''}"
          f"{' [+RSNA]' if args.use_rsna else ''}"
          f"{' [annotated]' if args.annotated else ''}"
          f"{' AMP' if args.amp else ''}")

    result = run_crossview_experiment(
        architecture         = args.arch,
        epochs               = args.epochs,
        batch_size           = args.batch,
        learning_rate        = args.lr,
        img_size             = args.img,
        focal_gamma          = args.gamma,
        label_smoothing      = args.label_smoothing,
        mixup_alpha          = args.mixup,
        d_model              = args.d_model,
        n_heads              = args.n_heads,
        pretrained           = True,
        medical              = args.medical,
        use_rsna             = args.use_rsna,
        annotated            = args.annotated,
        freeze_backbone_epochs = args.freeze,
        num_workers          = args.workers,
        use_amp              = args.amp,
    )

    if "error" in result:
        print(f"[{args.key}] ERROR: {result['error']}")
        sys.exit(1)

    m = result.get("final_metrics_opt_thresh", {})
    print(f"[{args.key}] DONE  "
          f"AUC={result.get('best_val_auc',0):.4f}  "
          f"F1={m.get('f1_score',0):.4f}  "
          f"sens={m.get('sensitivity',0):.4f}  "
          f"s@90={m.get('sens_at_90pct_spec')}  "
          f"elapsed={result.get('elapsed_seconds',0):.0f}s")
    _mark_done(args.key, m)


if __name__ == "__main__":
    main()
