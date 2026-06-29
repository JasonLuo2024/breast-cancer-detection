"""
Late-fusion evaluation: aggregates image-level probabilities from a trained
image-level model into breast-level and patient-level predictions.

This requires NO retraining — it shows how much multi-view aggregation alone
improves over single-image prediction, forming Table 1 of the paper.

Usage:
    python scripts/evaluate_late_fusion.py --model G:/breast-cancer-research/models/full_resnet50_YYYYMMDD_HHMMSS.pt --arch resnet50 --img 224
    python scripts/evaluate_late_fusion.py --model G:/breast-cancer-research/models/full_resnet50_YYYYMMDD_HHMMSS.pt --arch resnet50 --img 384
    python scripts/evaluate_late_fusion.py  # auto-selects best model from G:/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from tools.multiview_tools import aggregate_late_fusion, find_optimal_threshold, _compute_metrics
from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV = Path(r"G:\breast-cancer-research\split_manifest.csv")


# ── image-level dataset (same as full_dataset_tools, no dependency) ───────────

class _ImageDataset(Dataset):
    def __init__(self, manifest_df, split="test", transform=None):
        self.transform = transform
        sub = manifest_df[manifest_df["split"] == split].copy()
        self.samples = []
        for _, row in sub.iterrows():
            p = Path(row["png_path"])
            if p.exists():
                self.samples.append({
                    "path": p,
                    "label": int(row["label"]),
                    "study_id": row["study_id"],
                    "laterality": row["laterality"],
                    "view_position": row["view_position"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(s["label"], dtype=torch.float32), idx

    def get_meta(self, idx):
        return self.samples[idx]


# ── model loader ──────────────────────────────────────────────────────────────

def _load_model(model_path: Path, arch: str, img_size: int):
    """Load a saved full-dataset image-level model."""
    from tools.full_dataset_tools import ARCHITECTURES
    model = ARCHITECTURES[arch]().to(DEVICE)
    state = torch.load(str(model_path), map_location=DEVICE)
    # Checkpoint may be raw state_dict or wrapped dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[LateFusion] Loaded {arch} from {model_path}")
    return model


def _auto_best_model() -> tuple[Path, str, int]:
    """Pick the best saved model from G:/models/ by filename timestamp."""
    candidates = list(LARGE_MODELS_DIR.glob("full_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No full_*.pt models found in {LARGE_MODELS_DIR}")
    # Prefer img384 ResNet50 if available (highest AUC in training)
    preferred = [c for c in candidates if "resnet50" in c.name and "img384" not in c.name]
    chosen    = sorted(preferred or candidates, key=lambda p: p.stat().st_mtime)[-1]
    # Infer arch from filename: full_{arch}_{ts}.pt
    parts = chosen.stem.split("_")
    arch  = "_".join(parts[1:-2]) if len(parts) >= 4 else "resnet50"
    if arch not in ("resnet50", "densenet121", "efficientnet_b3", "vgg19", "resnet18"):
        arch = "resnet50"
    img_size = 384 if "384" in chosen.name else 224
    return chosen, arch, img_size


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, manifest_df, img_size: int, batch_size: int = 32,
                  num_workers: int = 6) -> list[dict]:
    """Run model on all test images, return per-image record list."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = _ImageDataset(manifest_df, split="test", transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    records = []
    for imgs, labels, idxs in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        for i, idx in enumerate(idxs.numpy()):
            meta = ds.get_meta(int(idx))
            records.append({
                "study_id":    meta["study_id"],
                "laterality":  meta["laterality"],
                "view":        meta["view_position"],
                "label":       meta["label"],
                "prob":        float(probs[i]),
            })

    print(f"[LateFusion] Inference done: {len(records)} images")
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default=None,
                        help="Path to .pt checkpoint (auto-selects best if omitted)")
    parser.add_argument("--arch",    type=str, default=None,
                        help="Architecture name (auto-detected if omitted)")
    parser.add_argument("--img",     type=int, default=None,
                        help="Image size (auto-detected if omitted)")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
        arch       = args.arch or "resnet50"
        img_size   = args.img  or 224
    else:
        model_path, arch, img_size = _auto_best_model()
        if args.arch: arch     = args.arch
        if args.img:  img_size = args.img

    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found at {MANIFEST_CSV}")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_CSV)
    model    = _load_model(model_path, arch, img_size)
    records  = run_inference(model, manifest, img_size, args.batch, args.workers)

    # Compute late-fusion at all aggregation levels
    fusion_results = aggregate_late_fusion(records)

    print("\n" + "="*70)
    print(f"LATE FUSION EVALUATION  |  arch={arch}  img={img_size}")
    print("="*70)
    rows = [
        ("Image-level",        fusion_results["image_level"]),
        ("Breast (max)",       fusion_results["breast_level_max"]),
        ("Breast (mean)",      fusion_results["breast_level_mean"]),
        ("Patient (max)",      fusion_results["patient_level_max"]),
        ("Patient (mean)",     fusion_results["patient_level_mean"]),
    ]
    header = f"{'Level':<18} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'s@90':>6} {'s@95':>6} {'Thresh':>7}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<18} "
              f"{str(m.get('auc_roc','?')):>6} "
              f"{m.get('f1_score',0):.4f} "
              f"{m.get('sensitivity',0):.4f} "
              f"{m.get('specificity',0):.4f} "
              f"{str(m.get('sens_at_90pct_spec','?')):>6} "
              f"{str(m.get('sens_at_95pct_spec','?')):>6} "
              f"{m.get('threshold',0.5):.3f}")
    print("="*70)

    # Save results
    from datetime import datetime
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = LARGE_RESULTS_DIR / f"late_fusion_{arch}_img{img_size}_{ts}.json"
    payload = {
        "method":      "late_fusion",
        "architecture": arch,
        "model_path":  str(model_path),
        "img_size":    img_size,
        "n_images":    len(records),
        "results":     fusion_results,
        "timestamp":   datetime.now().isoformat(),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[LateFusion] Results saved → {out}")


if __name__ == "__main__":
    main()
