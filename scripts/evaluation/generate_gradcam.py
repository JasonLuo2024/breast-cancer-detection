"""
Grad-CAM saliency maps for the best multi-view model (DenseNet-121/concat).

For each breast pair (CC + MLO), produces a side-by-side PNG showing the
raw image and the Grad-CAM heatmap overlay. Saves top-K true positives,
top-K false negatives (highest-confidence misses), and top-K true negatives
for qualitative analysis in the paper.

Usage:
    python scripts/generate_gradcam.py
    python scripts/generate_gradcam.py --n_samples 10 --split test
    python scripts/generate_gradcam.py --model G:/breast-cancer-research/models/mv_densenet121_concat_20260508_034005.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV = Path(r"G:\breast-cancer-research\split_manifest.csv")


# ── Grad-CAM implementation ───────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for a DenseNet-style backbone where the final feature map
    is produced by the last dense block / transition layer.

    target_layer: the nn.Module whose output activations are used.
    For DenseNet-121, pass model.backbone.features.denseblock4.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._fwd_hook    = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook    = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, cc: torch.Tensor, mlo: torch.Tensor) -> tuple:
        """
        Returns (cam_cc, cam_mlo, prob) where cam_* are H×W numpy arrays
        in [0, 1] and prob is the sigmoid probability.
        """
        self.model.eval()
        cc  = cc.to(DEVICE).unsqueeze(0)   # (1,3,H,W)
        mlo = mlo.to(DEVICE).unsqueeze(0)

        # Forward — gradients for the CC branch
        cc.requires_grad_(True)
        self._activations = None
        self._gradients   = None
        logit = self.model(cc, mlo)        # scalar logit
        prob  = float(torch.sigmoid(logit).item())

        self.model.zero_grad()
        logit.backward()

        # Grad-CAM weights: global average of gradients over spatial dims
        grads = self._gradients            # (1, C, H', W')
        acts  = self._activations          # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_cc = cam

        # Repeat for MLO branch
        self._activations = None
        self._gradients   = None
        mlo.requires_grad_(True)
        logit2 = self.model(cc, mlo)
        self.model.zero_grad()
        logit2.backward()
        grads2 = self._gradients
        acts2  = self._activations
        weights2 = grads2.mean(dim=(2, 3), keepdim=True)
        cam2 = F.relu((weights2 * acts2).sum(dim=1, keepdim=True))
        cam2 = cam2.squeeze().cpu().numpy()
        if cam2.max() > 0:
            cam2 = cam2 / cam2.max()
        cam_mlo = cam2

        return cam_cc, cam_mlo, prob

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _overlay(img_tensor: torch.Tensor, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """Return H×W×3 uint8 numpy array with Grad-CAM overlay."""
    import cv2
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


def _save_panel(cc_raw, mlo_raw, cam_cc, cam_mlo, prob, label,
                study_id, laterality, out_path: Path):
    """Save a 2×2 panel: raw CC | CAM CC | raw MLO | CAM MLO."""
    try:
        import cv2
        from PIL import Image
    except ImportError:
        print("  [WARN] cv2 or PIL not available — skipping PNG save")
        return

    def _to_uint8(t):
        img = t.permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    cc_img   = _to_uint8(cc_raw)
    mlo_img  = _to_uint8(mlo_raw)
    cc_cam   = _overlay(cc_raw,  cam_cc)
    mlo_cam  = _overlay(mlo_raw, cam_mlo)

    H, W = cc_img.shape[:2]
    panel = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    panel[:H, :W]  = cc_img
    panel[:H, W:]  = cc_cam
    panel[H:, :W]  = mlo_img
    panel[H:, W:]  = mlo_cam

    # Annotate
    title = (f"Study={study_id}  Lat={laterality}  "
             f"Label={'POS' if label else 'NEG'}  P={prob:.3f}")
    cv2_img = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(cv2_img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(cv2_img, "CC raw | CC Grad-CAM", (10, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(cv2_img, "MLO raw | MLO Grad-CAM", (10, 2 * H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2_img)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples per category (TP, FN, TN)")
    parser.add_argument("--split",     type=str, default="test")
    parser.add_argument("--img",       type=int, default=224)
    parser.add_argument("--thresh",    type=float, default=0.3138,
                        help="Youden threshold from MC Dropout evaluation")
    args = parser.parse_args()

    # deferred imports (fail gracefully if cv2/PIL not installed)
    try:
        import pandas as pd
        from tools.multiview_tools import BreastPairDataset, BreastPairNet, _val_transform
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # locate model
    if args.model:
        model_path = Path(args.model)
    else:
        models = sorted(
            Path("G:/breast-cancer-research/models").glob("mv_densenet121_concat_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        if not models:
            print("ERROR: No mv_densenet121_concat_*.pt found in G:/models/")
            sys.exit(1)
        model_path = models[-1]

    print(f"Model: {model_path.name}")
    ckpt  = torch.load(str(model_path), map_location=DEVICE)
    model = BreastPairNet("densenet121", "concat", pretrained=False).to(DEVICE)
    sd    = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    # locate the target layer (last denseblock of backbone)
    target_layer = model.backbone.features.denseblock4
    gcam = GradCAM(model, target_layer)

    # dataset
    if not MANIFEST_CSV.exists():
        print(f"ERROR: {MANIFEST_CSV} not found")
        sys.exit(1)
    manifest = pd.read_csv(MANIFEST_CSV)
    transform = _val_transform(args.img)
    ds = BreastPairDataset(manifest, args.split, transform)
    print(f"Dataset: {len(ds)} breast pairs  |  threshold: {args.thresh}")

    # collect predictions
    records = []
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    meta   = ds.get_meta()

    print(f"Running inference on {len(ds)} pairs...")
    with torch.no_grad():
        for i, (cc, mlo, label) in enumerate(loader):
            prob  = float(torch.sigmoid(model(cc.to(DEVICE), mlo.to(DEVICE))).item())
            study_id, lat, lbl = meta[i]
            records.append({
                "idx": i, "prob": prob, "label": int(lbl),
                "pred": int(prob >= args.thresh),
                "study_id": study_id, "lat": lat,
                "cc": cc.squeeze(0), "mlo": mlo.squeeze(0),
            })

    tp_recs = sorted([r for r in records if r["label"]==1 and r["pred"]==1],
                     key=lambda x: x["prob"], reverse=True)
    fn_recs = sorted([r for r in records if r["label"]==1 and r["pred"]==0],
                     key=lambda x: x["prob"], reverse=True)
    tn_recs = sorted([r for r in records if r["label"]==0 and r["pred"]==0],
                     key=lambda x: x["prob"])

    out_dir = BASE_DIR / "data" / "gradcam"

    for tag, recs in [("true_positive", tp_recs), ("false_negative", fn_recs),
                      ("true_negative", tn_recs)]:
        print(f"\nGenerating Grad-CAM: {tag} (n={min(args.n_samples, len(recs))})")
        for i, r in enumerate(recs[:args.n_samples]):
            cam_cc, cam_mlo, prob = gcam(r["cc"], r["mlo"])
            fname = out_dir / tag / f"{tag}_{i:02d}_{r['study_id']}_{r['lat']}.png"
            _save_panel(r["cc"], r["mlo"], cam_cc, cam_mlo, prob,
                        r["label"], r["study_id"], r["lat"], fname)
            print(f"  [{i+1}/{min(args.n_samples, len(recs))}] {fname.name}  P={prob:.3f}")

    gcam.remove_hooks()
    print(f"\nDone. Panels saved to {out_dir}/")
    print("Categories: true_positive/ | false_negative/ | true_negative/")


if __name__ == "__main__":
    main()
