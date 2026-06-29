"""
Cross-view transformer, dual-scale YOLO-ROI, and multi-dataset tools.

New architectures:
  CrossViewTransformerNet  — spatial cross-attention between CC and MLO feature maps
  DualScaleBreastPairNet   — global + YOLO-detected ROI crop dual-scale model (post-YOLO)

New datasets:
  RsnaBreastPairDataset    — reads rsna_manifest.csv (cc_png/mlo_png/cancer columns)
  CombinedBreastPairDataset— VinDr training + RSNA training concatenated

Entry points:
  run_crossview_experiment()     — train CrossViewTransformerNet on VinDr (±RSNA)
  run_dualscale_experiment()     — train DualScaleBreastPairNet (needs YOLO done first)
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def _worker_init_fn(worker_id):
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def _safe_open(path) -> Image.Image:
    """Open an image, returning a black 224×224 RGB fallback on any read error."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), 0)

from tools.multiview_tools import (
    FocalLoss,
    _train_transform,
    _val_transform,
    _make_sampler,
    _compute_metrics,
    find_optimal_threshold,
    BreastPairDataset,
)
from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VINDR_MANIFEST           = Path("G:/breast-cancer-research/split_manifest.csv")
VINDR_MANIFEST_ANNOTATED = Path("G:/breast-cancer-research/split_manifest_annotated.csv")
RSNA_MANIFEST            = Path("G:/breast-cancer-research/rsna_manifest.csv")
RSNA_MANIFEST_ANNOTATED  = Path("G:/breast-cancer-research/rsna_manifest_annotated.csv")
RSNA_YOLO_CSV            = Path("G:/breast-cancer-research/rsna_yolo_detections.csv")


# ── spatial backbone wrappers ─────────────────────────────────────────────────
# These return 4-D feature maps [B, C, H, W] before global pooling,
# giving the cross-attention module spatial context to work with.

class _SpatialResNet50(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=w)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

    def forward(self, x):
        return self.features(x)  # [B, 2048, H/32, W/32]


class _SpatialDenseNet121(nn.Module):
    def __init__(self, pretrained: bool = True, medical: bool = False):
        super().__init__()
        if medical:
            import torchxrayvision as xrv
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            xrv_sd    = dict(xrv_model.state_dict())
            base      = models.densenet121(weights=None)
            tv_sd     = base.state_dict()
            transferred = 0
            for k in list(tv_sd.keys()):
                if k in xrv_sd:
                    if tv_sd[k].shape == xrv_sd[k].shape:
                        tv_sd[k] = xrv_sd[k]; transferred += 1
                    elif k == "features.conv0.weight":
                        tv_sd[k] = xrv_sd[k].repeat(1, 3, 1, 1) / 3.0; transferred += 1
            base.load_state_dict(tv_sd, strict=True)
            print(f"[MedicalSpatialBackbone] Transferred {transferred} XRV tensors")
        else:
            w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.densenet121(weights=w)
        self.features = base.features
        self.act      = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.features(x))  # [B, 1024, H/32, W/32]


class _SpatialEfficientNetB3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.efficientnet_b3(weights=w).features

    def forward(self, x):
        return self.features(x)  # [B, 1536, H/32, W/32]


class _SpatialConvNeXtBase(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.convnext_base(weights=w).features

    def forward(self, x):
        return self.features(x)  # [B, 1024, H/32, W/32]


_SPATIAL_FEAT_DIMS = {
    "resnet50":       2048,
    "densenet121":    1024,
    "efficientnet_b3":1536,
    "convnext_base":  1024,
}


def _build_spatial_backbone(arch: str, pretrained: bool = True,
                             medical: bool = False):
    """Returns (spatial_encoder, feat_dim)."""
    if arch == "resnet50":
        return _SpatialResNet50(pretrained), 2048
    elif arch == "densenet121":
        return _SpatialDenseNet121(pretrained, medical=medical), 1024
    elif arch == "efficientnet_b3":
        return _SpatialEfficientNetB3(pretrained), 1536
    elif arch == "convnext_base":
        return _SpatialConvNeXtBase(pretrained), 1024
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: resnet50, densenet121, "
                         "efficientnet_b3, convnext_base")


# ── cross-view transformer ────────────────────────────────────────────────────

class CrossViewTransformerNet(nn.Module):
    """
    Shared spatial CNN backbone + cross-attention between CC and MLO.

    Each CC spatial token attends to all MLO tokens (and vice versa), learning
    which regions of MLO are relevant to each region of CC.  Residual connections
    preserve local backbone features alongside the cross-view context.

    For 224px input: 7×7 = 49 spatial tokens per view (stride-32 backbone).
    For 512px input: 16×16 = 256 tokens — use AMP to keep VRAM under 12 GB.
    """
    def __init__(self, arch: str = "densenet121", pretrained: bool = True,
                 d_model: int = 512, n_heads: int = 8, dropout: float = 0.3,
                 medical: bool = False):
        super().__init__()
        self.spatial_enc, feat_dim = _build_spatial_backbone(arch, pretrained, medical)

        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, d_model, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.cross_cc2mlo = nn.MultiheadAttention(d_model, n_heads,
                                                   dropout=dropout, batch_first=True)
        self.cross_mlo2cc = nn.MultiheadAttention(d_model, n_heads,
                                                   dropout=dropout, batch_first=True)
        self.norm_cc  = nn.LayerNorm(d_model)
        self.norm_mlo = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.spatial_enc(x)          # [B, C, H, W]
        feat = self.proj(feat)              # [B, d, H, W]
        B, d, H, W = feat.shape
        return feat.flatten(2).permute(0, 2, 1)  # [B, HW, d]

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        seq_cc  = self._to_tokens(cc)   # [B, HW, d]
        seq_mlo = self._to_tokens(mlo)  # [B, HW, d]

        # Cross-attend with residual + layer norm
        ctx_cc,  _ = self.cross_cc2mlo(seq_cc,  seq_mlo, seq_mlo)
        ctx_mlo, _ = self.cross_mlo2cc(seq_mlo, seq_cc,  seq_cc)

        feat_cc  = self.norm_cc( seq_cc  + ctx_cc ).mean(1)  # [B, d]
        feat_mlo = self.norm_mlo(seq_mlo + ctx_mlo).mean(1)  # [B, d]

        fused = torch.cat([feat_cc, feat_mlo], dim=1)        # [B, 2d]
        return self.head(fused).squeeze(1)                   # [B]


# ── dual-scale model (for after YOLO inference) ───────────────────────────────

class DualScaleBreastPairNet(nn.Module):
    """
    Two-scale breast pair classifier:
      Global branch  : full mammogram (CC + MLO) encoded by shared backbone
      Local branch   : YOLO-detected lesion ROI crop encoded by second backbone
    Features fused per view, then cross-view concatenated → classifier.

    If no YOLO detection for a view, the full image is used as the local crop too.
    """
    def __init__(self, arch: str = "densenet121", pretrained: bool = True,
                 dropout: float = 0.3, medical: bool = False):
        super().__init__()
        from tools.multiview_tools import _build_backbone
        self.global_enc, feat_dim = _build_backbone(arch, pretrained, medical)
        self.local_enc,  _        = _build_backbone(arch, pretrained, medical)

        self.view_fuse = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

    def forward(self, cc_g, cc_l, mlo_g, mlo_l):
        # global + local features per view
        f_cc  = self.view_fuse(torch.cat([self.global_enc(cc_g),  self.local_enc(cc_l)],  dim=1))
        f_mlo = self.view_fuse(torch.cat([self.global_enc(mlo_g), self.local_enc(mlo_l)], dim=1))
        return self.head(torch.cat([f_cc, f_mlo], dim=1)).squeeze(1)


# ── RSNA dataset ──────────────────────────────────────────────────────────────

class RsnaBreastPairDataset(Dataset):
    """
    Reads rsna_manifest.csv built by scripts/preprocess_rsna.py.
    Columns: patient_id, laterality, cc_png, mlo_png, cancer, rsna_split.
    """
    def __init__(self, manifest_df: pd.DataFrame, split: str = "rsna_val",
                 transform=None):
        self.transform = transform
        sub = manifest_df[manifest_df["rsna_split"] == split].copy()
        self.samples: list[tuple[Path, Path, int]] = []

        for _, row in sub.iterrows():
            cc  = Path(row["cc_png"])
            mlo = Path(row["mlo_png"])
            if cc.exists() and mlo.exists():
                self.samples.append((cc, mlo, int(row["cancer"])))

        missing = len(sub) - len(self.samples)
        if missing:
            print(f"[RsnaBreastPairDataset] WARNING: {missing} pairs missing PNGs")
        pos = sum(s[2] for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[RsnaBreastPairDataset] split={split}  pairs={len(self.samples)} "
              f"(neg={neg}, pos={pos}, rate={100*pos/max(len(self.samples),1):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]
        cc  = _safe_open(cc_path)
        mlo = _safe_open(mlo_path)
        if self.transform:
            cc  = self.transform(cc)
            mlo = self.transform(mlo)
        return cc, mlo, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[2] for s in self.samples]


class CombinedBreastPairDataset(Dataset):
    """
    Concatenates VinDr training pairs and RSNA training pairs.
    RSNA positives are oversampled to match VinDr's cancer prevalence
    (~5%) rather than RSNA's 2.1%.
    """
    def __init__(self, vindr_df: pd.DataFrame, rsna_df: pd.DataFrame,
                 transform=None, rsna_oversample: int = 2):
        self.transform = transform
        self.samples: list[tuple[Path, Path, int]] = []

        # VinDr training pairs
        for (study_id, lat), grp in vindr_df[vindr_df.split == "training"].groupby(
                ["study_id", "laterality"]):
            cc_rows  = grp[grp.view_position == "CC"]
            mlo_rows = grp[grp.view_position == "MLO"]
            cc_path  = Path(cc_rows.iloc[0].png_path)  if len(cc_rows)  else None
            mlo_path = Path(mlo_rows.iloc[0].png_path) if len(mlo_rows) else None
            if cc_path is None or mlo_path is None:
                continue
            if not cc_path.exists() and not mlo_path.exists():
                continue
            cc_path  = cc_path  if cc_path  and cc_path.exists()  else mlo_path
            mlo_path = mlo_path if mlo_path and mlo_path.exists() else cc_path
            label = int(grp["label"].max())
            self.samples.append((cc_path, mlo_path, label))

        # RSNA training pairs (oversampled positives)
        rsna_train = rsna_df[rsna_df.rsna_split == "rsna_train"].copy()
        for _, row in rsna_train.iterrows():
            cc  = Path(row["cc_png"])
            mlo = Path(row["mlo_png"])
            if not cc.exists() or not mlo.exists():
                continue
            label = int(row["cancer"])
            reps  = rsna_oversample if label == 1 else 1
            for _ in range(reps):
                self.samples.append((cc, mlo, label))

        pos = sum(s[2] for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[CombinedBreastPairDataset] total={len(self.samples)} "
              f"(neg={neg}, pos={pos}, rate={100*pos/max(len(self.samples),1):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]
        cc  = _safe_open(cc_path)
        mlo = _safe_open(mlo_path)
        if self.transform:
            cc  = self.transform(cc)
            mlo = self.transform(mlo)
        return cc, mlo, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[2] for s in self.samples]


# ── ROI crop dataset for dual-scale (needs YOLO inference first) ─────────────

class RoiBreastPairDataset(Dataset):
    """
    Each sample: (cc_global, cc_roi, mlo_global, mlo_roi, label).
    ROI is the YOLO-detected lesion crop (padded 10%).
    Falls back to full image if no detection.
    """
    def __init__(self, manifest_df: pd.DataFrame, det_df: pd.DataFrame,
                 split: str = "rsna_val", transform=None, roi_size: int = 224):
        self.transform = transform
        self.roi_size  = roi_size
        sub = manifest_df[manifest_df.rsna_split == split].copy()

        det_map = {}
        for _, row in det_df.iterrows():
            if row["det_count"] > 0 and row["det_boxes"]:
                boxes = json.loads(row["det_boxes"])
                best  = max(boxes, key=lambda b: b["conf"])
                det_map[str(row["png_path"])] = best

        self.samples = []
        for _, row in sub.iterrows():
            cc  = Path(row["cc_png"])
            mlo = Path(row["mlo_png"])
            if not cc.exists() or not mlo.exists():
                continue
            cc_box  = det_map.get(str(cc))
            mlo_box = det_map.get(str(mlo))
            self.samples.append((cc, cc_box, mlo, mlo_box, int(row["cancer"])))

        pos = sum(s[4] for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[RoiBreastPairDataset] split={split}  pairs={len(self.samples)} "
              f"(neg={neg}, pos={pos})  "
              f"cc_detections={(sum(s[1] is not None for s in self.samples))}  "
              f"mlo_detections={(sum(s[3] is not None for s in self.samples))}")

    def _crop_roi(self, img: Image.Image, box: dict | None,
                  pad: float = 0.10) -> Image.Image:
        if box is None:
            return img
        W, H  = img.size
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - pad * bw)
        y1 = max(0, y1 - pad * bh)
        x2 = min(W, x2 + pad * bw)
        y2 = min(H, y2 + pad * bh)
        return img.crop((x1, y1, x2, y2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, cc_box, mlo_path, mlo_box, label = self.samples[idx]
        cc_img  = Image.open(cc_path).convert("RGB")
        mlo_img = Image.open(mlo_path).convert("RGB")

        cc_roi  = self._crop_roi(cc_img,  cc_box)
        mlo_roi = self._crop_roi(mlo_img, mlo_box)

        if self.transform:
            cc_img  = self.transform(cc_img)
            cc_roi  = self.transform(cc_roi)
            mlo_img = self.transform(mlo_img)
            mlo_roi = self.transform(mlo_roi)

        return cc_img, cc_roi, mlo_img, mlo_roi, torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        return [s[4] for s in self.samples]


# ── training / eval loops ─────────────────────────────────────────────────────

def _train_cv_epoch(model, loader, optimizer, criterion,
                    mixup_alpha: float = 0.0, scaler=None):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []

    for cc, mlo, labels in loader:
        cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)

        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(cc.size(0), device=DEVICE)
            cc     = lam * cc    + (1 - lam) * cc[idx]
            mlo    = lam * mlo   + (1 - lam) * mlo[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(cc, mlo)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(cc, mlo)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.round().int().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

    m = _compute_metrics(all_labels, all_probs)
    return total_loss / max(len(loader.dataset), 1), m


def _eval_cv_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    with torch.no_grad():
        for cc, mlo, labels in loader:
            cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)
            logits = model(cc, mlo)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())

    return total_loss / max(len(loader.dataset), 1), all_labels, all_probs


def _tta_predict(model, loader, n_tta: int = 5):
    """Horizontal-flip TTA: average over n_tta forward passes."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for cc, mlo, labels in loader:
            cc, mlo = cc.to(DEVICE), mlo.to(DEVICE)
            probs_runs = []
            for _ in range(n_tta):
                # random horizontal flip per pass
                if torch.rand(1).item() > 0.5:
                    cc_f  = torch.flip(cc,  [3])
                    mlo_f = torch.flip(mlo, [3])
                else:
                    cc_f, mlo_f = cc, mlo
                p = torch.sigmoid(model(cc_f, mlo_f)).cpu()
                probs_runs.append(p)
            all_probs.extend(torch.stack(probs_runs, 0).mean(0).tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_probs


# ── main experiment: CrossViewTransformerNet ──────────────────────────────────

def run_crossview_experiment(
    architecture: str  = "densenet121",
    epochs: int        = 30,
    batch_size: int    = 6,
    learning_rate: float = 5e-5,
    img_size: int      = 224,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    d_model: int       = 512,
    n_heads: int       = 8,
    dropout: float     = 0.3,
    pretrained: bool   = True,
    medical: bool      = False,
    use_rsna: bool     = False,
    annotated: bool    = False,
    freeze_backbone_epochs: int = 3,
    num_workers: int   = 6,
    use_amp: bool      = False,
) -> dict:
    """Train CrossViewTransformerNet on VinDr (and optionally RSNA training split)."""
    vindr_mf      = VINDR_MANIFEST_ANNOTATED if annotated else VINDR_MANIFEST
    vindr_test_mf = VINDR_MANIFEST           # test set always uses original PNGs (no GT boxes)
    rsna_mf       = RSNA_MANIFEST_ANNOTATED  if annotated else RSNA_MANIFEST

    if not vindr_mf.exists():
        return {"error": f"VinDr manifest not found: {vindr_mf}"}

    vindr_df      = pd.read_csv(vindr_mf)
    vindr_test_df = pd.read_csv(vindr_test_mf)

    if use_rsna:
        if not rsna_mf.exists():
            return {"error": f"RSNA manifest not found: {rsna_mf}. "
                    "Run scripts/preprocess_rsna.py + annotate_pngs.py first."}
        rsna_df  = pd.read_csv(rsna_mf)
        train_ds = CombinedBreastPairDataset(vindr_df, rsna_df,
                                             transform=_train_transform(img_size))
    else:
        train_ds = BreastPairDataset(vindr_df, "training", _train_transform(img_size))

    # Test set uses original unannotated PNGs — prevents data leakage from GT boxes
    val_ds = BreastPairDataset(vindr_test_df, "test", _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No training pairs found. Check manifest paths."}

    sampler      = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2,
                              worker_init_fn=_worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2,
                              worker_init_fn=_worker_init_fn)

    model     = CrossViewTransformerNet(architecture, pretrained, d_model, n_heads,
                                        dropout, medical).to(DEVICE)
    criterion = FocalLoss(0.25, focal_gamma, label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = (torch.amp.GradScaler("cuda")
                 if (use_amp and DEVICE.type == "cuda") else None)

    if scaler:
        print(f"[CrossView] AMP enabled (FP16)")

    def _set_backbone_grad(req: bool):
        for p in model.spatial_enc.parameters():
            p.requires_grad = req

    history, best_auc, best_state, best_thresh = [], 0.0, None, 0.5
    start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[CrossView] Epoch {epoch}: backbone unfrozen")

        tr_loss, tr_m  = _train_cv_epoch(model, train_loader, optimizer, criterion,
                                          mixup_alpha, scaler)
        va_loss, va_labels, va_probs = _eval_cv_epoch(model, val_loader, criterion)

        thresh   = find_optimal_threshold(va_labels, va_probs)
        va_m_05  = _compute_metrics(va_labels, va_probs, 0.5)
        va_m_opt = _compute_metrics(va_labels, va_probs, thresh)
        scheduler.step()

        cur_auc = va_m_05.get("auc_roc") or 0.0
        if cur_auc > best_auc:
            best_auc    = cur_auc
            best_thresh = thresh
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        med_tag = " [medical]" if medical else ""
        rsna_tag = " [+RSNA]" if use_rsna else ""
        print(f"  Ep {epoch:02d}/{epochs}{med_tag}{rsna_tag} | "
              f"tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va AUC={va_m_05['auc_roc']}  "
              f"f1@opt={va_m_opt['f1_score']:.4f}(t={thresh:.3f})  "
              f"s@90={va_m_05['sens_at_90pct_spec']}")
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "val_loss": round(va_loss, 4),
            "train": tr_m,
            "val_thresh_05":  va_m_05,
            "val_thresh_opt": va_m_opt,
            "opt_threshold":  round(thresh, 4),
        })

    elapsed = round(time.time() - start, 1)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = f"cv_{architecture}{'_med' if medical else ''}{'_rsna' if use_rsna else ''}"

    if best_state:
        model.load_state_dict(best_state)
        ckpt = LARGE_MODELS_DIR / f"{tag}_{ts}.pt"
        torch.save({
            "state_dict": best_state,
            "arch": architecture, "medical": medical, "use_rsna": use_rsna,
            "d_model": d_model, "n_heads": n_heads,
            "img_size": img_size, "best_thresh": best_thresh,
            "model_type": "crossview",
        }, str(ckpt))
        print(f"[CrossView] Checkpoint -> {ckpt}")

    result = {
        "method": "crossview_transformer",
        "architecture": architecture, "medical": medical, "use_rsna": use_rsna,
        "d_model": d_model, "n_heads": n_heads,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "img_size": img_size,
        "use_amp": use_amp,
        "elapsed_seconds": elapsed,
        "best_val_auc": round(best_auc, 4),
        "best_thresh":  round(best_thresh, 4),
        "final_metrics_thresh05":  history[-1]["val_thresh_05"]  if history else {},
        "final_metrics_opt_thresh": history[-1]["val_thresh_opt"] if history else {},
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"{tag}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[CrossView] Done {elapsed:.0f}s  best_AUC={best_auc:.4f}  -> {out}")
    return result


# ── dual-scale experiment (run after YOLO inference is complete) ──────────────

def run_dualscale_experiment(
    architecture: str  = "densenet121",
    epochs: int        = 30,
    batch_size: int    = 4,
    learning_rate: float = 5e-5,
    img_size: int      = 224,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.1,
    pretrained: bool   = True,
    medical: bool      = False,
    freeze_backbone_epochs: int = 3,
    num_workers: int   = 4,
    use_amp: bool      = False,
) -> dict:
    """Train DualScaleBreastPairNet on RSNA with YOLO ROI crops."""
    for f in [RSNA_MANIFEST, RSNA_YOLO_CSV]:
        if not f.exists():
            return {"error": f"Required file not found: {f}"}

    rsna_df = pd.read_csv(RSNA_MANIFEST)
    det_df  = pd.read_csv(RSNA_YOLO_CSV)

    train_ds = RoiBreastPairDataset(rsna_df, det_df, "rsna_train", _train_transform(img_size))
    val_ds   = RoiBreastPairDataset(rsna_df, det_df, "rsna_val",   _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No training pairs found."}

    labels   = np.array(train_ds.get_labels())
    counts   = np.bincount(labels)
    weights  = 1.0 / counts[labels]
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True,
                              worker_init_fn=_worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True,
                              worker_init_fn=_worker_init_fn)

    model     = DualScaleBreastPairNet(architecture, pretrained,
                                        dropout=0.3, medical=medical).to(DEVICE)
    criterion = FocalLoss(0.25, focal_gamma, label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = (torch.amp.GradScaler("cuda")
                 if (use_amp and DEVICE.type == "cuda") else None)

    def _set_grad(req: bool):
        for p in list(model.global_enc.parameters()) + list(model.local_enc.parameters()):
            p.requires_grad = req

    history, best_auc, best_state, best_thresh = [], 0.0, None, 0.5
    start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_grad(True)
            print(f"[DualScale] Epoch {epoch}: backbones unfrozen")

        model.train()
        tr_loss, all_labels, all_probs = 0.0, [], []

        for cc_g, cc_l, mlo_g, mlo_l, labels in train_loader:
            cc_g, cc_l = cc_g.to(DEVICE), cc_l.to(DEVICE)
            mlo_g, mlo_l, labels = mlo_g.to(DEVICE), mlo_l.to(DEVICE), labels.to(DEVICE)

            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(cc_g.size(0), device=DEVICE)
                cc_g = lam*cc_g + (1-lam)*cc_g[idx]; cc_l = lam*cc_l + (1-lam)*cc_l[idx]
                mlo_g = lam*mlo_g + (1-lam)*mlo_g[idx]; mlo_l = lam*mlo_l + (1-lam)*mlo_l[idx]
                labels = lam*labels + (1-lam)*labels[idx]

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(cc_g, cc_l, mlo_g, mlo_l)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(cc_g, cc_l, mlo_g, mlo_l)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tr_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.round().int().cpu().tolist())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

        tr_m = _compute_metrics(all_labels, all_probs)
        tr_loss /= max(len(train_ds), 1)

        model.eval()
        va_labels, va_probs = [], []
        with torch.no_grad():
            for cc_g, cc_l, mlo_g, mlo_l, labels in val_loader:
                cc_g, cc_l = cc_g.to(DEVICE), cc_l.to(DEVICE)
                mlo_g, mlo_l = mlo_g.to(DEVICE), mlo_l.to(DEVICE)
                p = torch.sigmoid(model(cc_g, cc_l, mlo_g, mlo_l))
                va_labels.extend(labels.tolist())
                va_probs.extend(p.cpu().tolist())

        thresh   = find_optimal_threshold(va_labels, va_probs)
        va_m_05  = _compute_metrics(va_labels, va_probs, 0.5)
        va_m_opt = _compute_metrics(va_labels, va_probs, thresh)
        scheduler.step()

        cur_auc = va_m_05.get("auc_roc") or 0.0
        if cur_auc > best_auc:
            best_auc   = cur_auc; best_thresh = thresh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d}/{epochs} | tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va AUC={va_m_05['auc_roc']}  f1@opt={va_m_opt['f1_score']:.4f}(t={thresh:.3f})")
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "val_thresh_05":  va_m_05,
            "val_thresh_opt": va_m_opt,
        })

    elapsed = round(time.time() - start, 1)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = f"dualscale_{architecture}{'_med' if medical else ''}"

    if best_state:
        model.load_state_dict(best_state)
        ckpt = LARGE_MODELS_DIR / f"{tag}_{ts}.pt"
        torch.save({
            "state_dict": best_state, "arch": architecture, "medical": medical,
            "img_size": img_size, "best_thresh": best_thresh,
            "model_type": "dualscale",
        }, str(ckpt))
        print(f"[DualScale] Checkpoint -> {ckpt}")

    result = {
        "method": "dualscale_breastpair",
        "architecture": architecture, "medical": medical,
        "epochs": epochs, "batch_size": batch_size, "img_size": img_size,
        "elapsed_seconds": elapsed,
        "best_val_auc": round(best_auc, 4),
        "best_thresh":  round(best_thresh, 4),
        "final_metrics_opt_thresh": history[-1]["val_thresh_opt"] if history else {},
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"{tag}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[DualScale] Done {elapsed:.0f}s  best_AUC={best_auc:.4f}  -> {out}")
    return result
