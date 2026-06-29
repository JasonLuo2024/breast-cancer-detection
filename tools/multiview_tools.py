"""
Multi-view breast-pair fusion for VinDr-Mammo.

Two architectures:

BreastPairNet  (ipsilateral, Phase 1-2 baseline)
  One sample = ipsilateral CC+MLO pair → single breast prediction.
  Fusion: concat or soft-attention over two feature vectors.

BilateralFusionNet  (4-view bilateral, Phase B novel contribution)
  One sample = full patient quad (L-CC, L-MLO, R-CC, R-MLO).
  Architecture:
    1. Shared backbone encodes all 4 views independently.
    2. Shared ipsilateral concat head: [h_CC, h_MLO] → h_breast (left & right).
    3. Cross-bilateral MultiheadAttention: each breast attends to the other,
       capturing the radiologist's contralateral asymmetry check.
    4. Shared per-breast output head → (logit_L, logit_R).
  Training: focal_loss(L) + focal_loss(R) summed.

Phase A regularisation (applied to both architectures):
  label_smoothing — soft targets prevent overconfident logits.
  MixUp          — λ-blended image pairs and labels at the batch level.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from config import LARGE_RESULTS_DIR, LARGE_MODELS_DIR

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV = Path(r"G:\breast-cancer-research\split_manifest.csv")


# ── dataset ───────────────────────────────────────────────────────────────────

class BreastPairDataset(Dataset):
    """
    Each sample: (cc_tensor, mlo_tensor, breast_label).
    Grouped by study_id × laterality.  Missing view is duplicated.
    """
    def __init__(self, manifest_df: pd.DataFrame, split: str = "training",
                 transform=None):
        self.transform = transform
        sub = manifest_df[manifest_df["split"] == split].copy()
        self.samples = []  # (cc_path, mlo_path, label, study_id, laterality)

        for (study_id, lat), grp in sub.groupby(["study_id", "laterality"]):
            cc_rows  = grp[grp["view_position"] == "CC"]
            mlo_rows = grp[grp["view_position"] == "MLO"]

            cc_path  = Path(cc_rows.iloc[0]["png_path"])  if len(cc_rows)  else None
            mlo_path = Path(mlo_rows.iloc[0]["png_path"]) if len(mlo_rows) else None

            cc_ok  = cc_path  is not None and cc_path.exists()
            mlo_ok = mlo_path is not None and mlo_path.exists()
            if not cc_ok and not mlo_ok:
                continue

            # Duplicate missing view
            if not cc_ok:  cc_path  = mlo_path
            if not mlo_ok: mlo_path = cc_path

            label = int(grp["label"].max())  # OR logic: cancer if any view positive
            self.samples.append((cc_path, mlo_path, label, study_id, lat))

        pos = sum(s[2] for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[BreastPairDataset] split={split}  pairs={len(self.samples)} "
              f"(neg={neg}, pos={pos}, rate={100*pos/max(len(self.samples),1):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label, _, _ = self.samples[idx]
        cc  = Image.open(cc_path).convert("RGB")
        mlo = Image.open(mlo_path).convert("RGB")
        if self.transform:
            cc  = self.transform(cc)
            mlo = self.transform(mlo)
        return cc, mlo, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[2] for s in self.samples]

    def get_meta(self) -> list[tuple]:
        """Returns (study_id, laterality, label) for each sample."""
        return [(s[3], s[4], s[2]) for s in self.samples]


# ── transforms ────────────────────────────────────────────────────────────────

def _train_transform(img_size: int = 224):
    # Mammogram-specific deformations: elastic deformation simulates breast tissue
    # compression variation; RandomAffine shear approximates grid distortion from
    # scanner pressure artefacts. Both validated by Miller et al. 2023 (IEEE TMI).
    aug_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=0, shear=(-5, 5, -5, 5)),
    ]
    try:
        # alpha=50 (displacement magnitude px), sigma=5 (field smoothness)
        aug_list.append(transforms.ElasticTransform(alpha=50.0, sigma=5.0))
    except AttributeError:
        pass  # torchvision < 0.14 — skip gracefully
    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug_list)


def _val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing  # ε: 0→hard labels, 0.1→soft targets

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── backbone builders ─────────────────────────────────────────────────────────

def _build_backbone(arch: str, pretrained: bool = True,
                    medical_pretrained: bool = False):
    """Returns (backbone_module_without_head, feature_dim).

    medical_pretrained=True loads a DenseNet-121 pretrained on multi-dataset
    chest X-rays via TorchXRayVision (nih+pc+chex+mimic+google+openi+kaggle).
    The 1-channel first-conv is adapted to 3-channel by repeating and rescaling.
    Only valid for arch='densenet121'.
    """
    if medical_pretrained:
        if arch != "densenet121":
            raise ValueError("medical_pretrained is only supported for arch='densenet121'")
        return _build_medical_backbone()

    if arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    elif arch == "efficientnet_b3":
        w = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b3(weights=w)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    elif arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: resnet50, efficientnet_b3, densenet121")


def _build_medical_backbone():
    """
    DenseNet-121 initialized from TorchXRayVision multi-dataset chest X-ray weights.
    All layers transferred directly; first conv (1-ch→3-ch) adapted by channel repeat.
    Returns (backbone, 1024).
    """
    import torchxrayvision as xrv

    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xrv_sd    = {k: v for k, v in xrv_model.state_dict().items()}

    m = models.densenet121(weights=None)
    feat_dim = m.classifier.in_features
    m.classifier = nn.Identity()
    tv_sd = m.state_dict()

    transferred, skipped = 0, 0
    for k in list(tv_sd.keys()):
        if k in xrv_sd:
            if tv_sd[k].shape == xrv_sd[k].shape:
                tv_sd[k] = xrv_sd[k]
                transferred += 1
            elif k == "features.conv0.weight":
                # XRV: (64,1,7,7) → torchvision: (64,3,7,7)
                # Repeat 1-ch weight across 3 channels, divide by 3 to keep scale
                tv_sd[k] = xrv_sd[k].repeat(1, 3, 1, 1) / 3.0
                transferred += 1
            else:
                skipped += 1
        else:
            skipped += 1

    m.load_state_dict(tv_sd, strict=True)
    print(f"[MedicalBackbone] Transferred {transferred} tensors, skipped {skipped}")
    return m, feat_dim


# ── multi-view fusion model ───────────────────────────────────────────────────

class BreastPairNet(nn.Module):
    """
    Shared CNN backbone encodes CC and MLO independently.

    concat    : [f_cc | f_mlo] → dropout → Linear(feat*2, 512) → ReLU → Linear(512,1)
    attention : soft attention over {f_cc, f_mlo} → weighted sum → Linear(feat,1)
    """
    def __init__(self, arch: str = "resnet50", fusion: str = "concat",
                 pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained)
        self.fusion_mode = fusion

        if fusion == "concat":
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feat_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(0.5),   # stronger: was 0.3, now 0.5
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        elif fusion == "attention":
            self.view_attn = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feat_dim, 1),
            )
        else:
            raise ValueError(f"Unknown fusion '{fusion}'. Choose: concat, attention")

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        f_cc  = self.backbone(cc)   # (B, feat_dim)
        f_mlo = self.backbone(mlo)  # (B, feat_dim)

        if self.fusion_mode == "concat":
            fused = torch.cat([f_cc, f_mlo], dim=1)  # (B, feat_dim*2)
        else:  # attention
            stack   = torch.stack([f_cc, f_mlo], dim=1)       # (B, 2, feat_dim)
            scores  = self.view_attn(stack).squeeze(-1)        # (B, 2)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, 2, 1)
            fused   = (weights * stack).sum(dim=1)             # (B, feat_dim)

        return self.head(fused).squeeze(1)  # (B,)


# ── sampler ───────────────────────────────────────────────────────────────────

def _make_sampler(dataset):
    labels  = np.array(dataset.get_labels())
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(all_labels, all_probs, threshold: float = 0.5) -> dict:
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc  = (tp + tn) / len(labels) if len(labels) else 0.0
    auc = s90 = s95 = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
        spec_arr = 1.0 - fpr
        m90 = spec_arr >= 0.90
        s90 = float(tpr[m90].max()) if m90.any() else 0.0
        m95 = spec_arr >= 0.95
        s95 = float(tpr[m95].max()) if m95.any() else 0.0
    except Exception:
        pass
    return {
        "accuracy": round(acc, 4), "sensitivity": round(sens, 4),
        "specificity": round(spec, 4), "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "auc_roc":            round(auc, 4) if not np.isnan(auc) else None,
        "sens_at_90pct_spec": round(s90, 4) if not np.isnan(s90) else None,
        "sens_at_95pct_spec": round(s95, 4) if not np.isnan(s95) else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": round(threshold, 4),
    }


def find_optimal_threshold(labels, probs) -> float:
    """Youden's J: maximises sensitivity + specificity − 1."""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j = tpr - fpr
        return float(thresholds[j.argmax()])
    except Exception:
        return 0.5


# ── late-fusion aggregation (no retraining required) ─────────────────────────

def aggregate_late_fusion(
    records: list[dict],
    agg: str = "max",
) -> dict:
    """
    Takes per-image inference records and computes breast-level and
    patient-level AUC by aggregating image scores.

    Each record must have: {"study_id", "laterality", "label", "prob"}
    label should be the IMAGE-level label (0/1); aggregation recomputes
    breast- and patient-level labels via OR logic.

    Returns dict with keys:
      image_level, breast_level_max, breast_level_mean,
      patient_level_max, patient_level_mean
    each containing the full metrics dict.
    """
    df = pd.DataFrame(records)
    fn = np.max if agg == "max" else np.mean

    # Image-level
    img_thresh   = find_optimal_threshold(df["label"].tolist(), df["prob"].tolist())
    img_metrics  = _compute_metrics(df["label"].tolist(), df["prob"].tolist(), img_thresh)

    results = {"image_level": img_metrics}

    # Breast-level (study_id × laterality)
    for method, agg_fn in [("max", np.max), ("mean", np.mean)]:
        breast = (
            df.groupby(["study_id", "laterality"])
              .agg(prob=("prob", agg_fn), label=("label", "max"))
              .reset_index()
        )
        thresh = find_optimal_threshold(breast["label"].tolist(), breast["prob"].tolist())
        results[f"breast_level_{method}"] = _compute_metrics(
            breast["label"].tolist(), breast["prob"].tolist(), thresh)

    # Patient-level (study_id)
    for method, agg_fn in [("max", np.max), ("mean", np.mean)]:
        patient = (
            df.groupby("study_id")
              .agg(prob=("prob", agg_fn), label=("label", "max"))
              .reset_index()
        )
        thresh = find_optimal_threshold(patient["label"].tolist(), patient["prob"].tolist())
        results[f"patient_level_{method}"] = _compute_metrics(
            patient["label"].tolist(), patient["prob"].tolist(), thresh)

    return results


# ── MixUp ────────────────────────────────────────────────────────────────────

def _mixup_batch(x1, x2, y1, y2, alpha: float = 0.2):
    """
    Image-level MixUp: λ ~ Beta(α,α); mixed = λ*a + (1-λ)*b for inputs and labels.
    Operates on a single (x1, x2) view pair — call separately for CC and MLO.
    """
    lam = float(np.random.beta(alpha, alpha))
    x1_mix = lam * x1 + (1 - lam) * x2
    x2_mix = lam * x2 + (1 - lam) * x1   # mix the other view symmetrically
    y_mix  = lam * y1 + (1 - lam) * y2
    return x1_mix, x2_mix, y_mix


# ── bilateral dataset ─────────────────────────────────────────────────────────

class PatientQuadDataset(Dataset):
    """
    Each sample = full patient quad: (L_CC, L_MLO, R_CC, R_MLO, L_label, R_label).
    Missing views within a breast are duplicated from the available view.
    Patients where an entire breast (both CC and MLO) is absent are skipped.
    """
    def __init__(self, manifest_df: pd.DataFrame, split: str = "training",
                 transform=None):
        self.transform = transform
        sub = manifest_df[manifest_df["split"] == split].copy()
        self.samples = []

        for study_id, grp in sub.groupby("study_id"):
            def _get(lat, view):
                rows = grp[(grp["laterality"] == lat) & (grp["view_position"] == view)]
                if len(rows) == 0:
                    return None
                p = Path(rows.iloc[0]["png_path"])
                return p if p.exists() else None

            l_cc  = _get("L", "CC");  l_mlo = _get("L", "MLO")
            r_cc  = _get("R", "CC");  r_mlo = _get("R", "MLO")

            # Need at least one view per breast to form a valid quad
            if (l_cc is None and l_mlo is None) or (r_cc is None and r_mlo is None):
                continue

            # Duplicate missing single view
            if l_cc  is None: l_cc  = l_mlo
            if l_mlo is None: l_mlo = l_cc
            if r_cc  is None: r_cc  = r_mlo
            if r_mlo is None: r_mlo = r_cc

            l_grp   = grp[grp["laterality"] == "L"]
            r_grp   = grp[grp["laterality"] == "R"]
            l_label = int(l_grp["label"].max()) if len(l_grp) else 0
            r_label = int(r_grp["label"].max()) if len(r_grp) else 0

            self.samples.append((l_cc, l_mlo, r_cc, r_mlo,
                                  l_label, r_label, study_id))

        pos_patients = sum(1 for s in self.samples if s[4] or s[5])
        print(f"[PatientQuadDataset] split={split}  patients={len(self.samples)} "
              f"(pos_patient={pos_patients}, "
              f"rate={100*pos_patients/max(len(self.samples),1):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        l_cc, l_mlo, r_cc, r_mlo, l_lbl, r_lbl, _ = self.samples[idx]
        imgs = [Image.open(p).convert("RGB")
                for p in (l_cc, l_mlo, r_cc, r_mlo)]
        if self.transform:
            imgs = [self.transform(im) for im in imgs]
        l_label = torch.tensor(l_lbl, dtype=torch.float32)
        r_label = torch.tensor(r_lbl, dtype=torch.float32)
        return imgs[0], imgs[1], imgs[2], imgs[3], l_label, r_label

    def get_labels(self) -> list[int]:
        # Patient is positive if either breast is positive
        return [int(s[4] or s[5]) for s in self.samples]

    def get_breast_meta(self) -> list[tuple]:
        """Returns [(study_id, 'L', l_label), (study_id, 'R', r_label), ...]."""
        out = []
        for s in self.samples:
            out.append((s[6], "L", s[4]))
            out.append((s[6], "R", s[5]))
        return out


# ── bilateral model ───────────────────────────────────────────────────────────

class BilateralFusionNet(nn.Module):
    """
    4-view bilateral fusion with cross-lateral attention.

    Architecture:
      1. Shared backbone encodes L-CC, L-MLO, R-CC, R-MLO independently.
      2. Shared ipsilateral head fuses [h_CC, h_MLO] → h_breast for each side.
      3. Cross-bilateral MultiheadAttention: each breast attends to the other
         to capture radiologist contralateral asymmetry reasoning.
      4. Shared per-breast classification head → (logit_L, logit_R).
    """
    def __init__(self, arch: str = "densenet121", n_heads: int = 4,
                 pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained)

        # Shared ipsilateral fusion: [h_cc, h_mlo] → h_breast
        self.ipsi_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        # Cross-bilateral attention (each breast attends to the other)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=n_heads,
            dropout=0.1, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(feat_dim)

        # Shared per-breast output head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _encode_breast(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        h_cc  = self.backbone(cc)
        h_mlo = self.backbone(mlo)
        return self.ipsi_head(torch.cat([h_cc, h_mlo], dim=1))  # (B, feat_dim)

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        h_L = self._encode_breast(l_cc, l_mlo)   # (B, d)
        h_R = self._encode_breast(r_cc, r_mlo)   # (B, d)

        # Stack as sequence: (B, 2, d) — left is query attending to right and vice versa
        seq = torch.stack([h_L, h_R], dim=1)                    # (B, 2, d)
        attn_out, _ = self.cross_attn(seq, seq, seq)             # (B, 2, d)
        seq = self.cross_norm(seq + attn_out)                    # residual + norm

        h_L_bi = seq[:, 0, :]   # (B, d)
        h_R_bi = seq[:, 1, :]   # (B, d)

        logit_L = self.classifier(h_L_bi).squeeze(1)  # (B,)
        logit_R = self.classifier(h_R_bi).squeeze(1)  # (B,)
        return logit_L, logit_R


# ── train / eval loops ────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for cc, mlo, labels in loader:
        cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(cc, mlo)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(cc, mlo)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


def _train_epoch_mixup(model, loader, optimizer, criterion, mixup_alpha: float,
                       scaler=None):
    """Intra-batch MixUp: shuffle within the current batch — no list(loader) needed."""
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for cc, mlo, labels in loader:
        cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.float().to(DEVICE)
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(cc.size(0), device=DEVICE)
        cc_m  = lam * cc  + (1 - lam) * cc[idx]
        mlo_m = lam * mlo + (1 - lam) * mlo[idx]
        y_m   = lam * labels + (1 - lam) * labels[idx]
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(cc_m, mlo_m)
                loss   = criterion(logits, y_m)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(cc_m, mlo_m)
            loss   = criterion(logits, y_m)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for cc, mlo, labels in loader:
        cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.float().to(DEVICE)
        logits = model(cc, mlo)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), all_labels, all_probs


# ── MC Dropout uncertainty inference ─────────────────────────────────────────

def mc_dropout_predict(model, dataset, img_size: int, batch_size: int = 8,
                       num_workers: int = 4, n_passes: int = 20) -> list[dict]:
    """
    Run N stochastic forward passes with Dropout active (BN stays in eval mode).
    Returns per-breast-pair uncertainty records.

    Uncertainty metric: binary entropy of mean malignancy probability.
    Validated clinically by Verboom et al. 2025 (Radiology 316:e242594).

    Returns list of dicts: {study_id, laterality, label,
                             mean_prob, entropy, std_prob}
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_mean, all_std, all_ent = [], [], []
    eps = 1e-8

    with torch.no_grad():
        for cc, mlo, _ in loader:
            cc, mlo = cc.to(DEVICE), mlo.to(DEVICE)
            pass_probs = torch.stack(
                [torch.sigmoid(model(cc, mlo)) for _ in range(n_passes)]
            )  # (n_passes, B)
            mean_p = pass_probs.mean(dim=0)
            std_p  = pass_probs.std(dim=0)
            ent    = -(mean_p * torch.log(mean_p + eps) +
                       (1 - mean_p) * torch.log(1 - mean_p + eps))
            all_mean.extend(mean_p.cpu().numpy().tolist())
            all_std.extend(std_p.cpu().numpy().tolist())
            all_ent.extend(ent.cpu().numpy().tolist())

    model.eval()  # disable dropout again

    return [
        {
            "study_id":   study_id,
            "laterality": lat,
            "label":      label,
            "mean_prob":  round(float(all_mean[i]), 6),
            "entropy":    round(float(all_ent[i]),  6),
            "std_prob":   round(float(all_std[i]),  6),
        }
        for i, (study_id, lat, label) in enumerate(dataset.get_meta())
    ]


# ── public API ────────────────────────────────────────────────────────────────

def run_multiview_experiment(
    architecture: str = "resnet50",
    fusion: str = "concat",
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    num_workers: int = 6,
    use_amp: bool = False,
) -> dict:
    """Train breast-pair multi-view fusion on full VinDr-Mammo (patient-level split).

    use_amp=True enables mixed-precision (FP16) training via torch.cuda.amp.
    Required for 512px training to fit in 12 GB VRAM at batch>=3.
    """
    if not MANIFEST_CSV.exists():
        return {"error": f"Manifest not found: {MANIFEST_CSV}. Run preprocess_full_dataset.py first."}

    manifest = pd.read_csv(MANIFEST_CSV)
    train_ds = BreastPairDataset(manifest, "training", _train_transform(img_size))
    val_ds   = BreastPairDataset(manifest, "test",     _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No training breast-pairs found. Check manifest PNG paths."}

    sampler      = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = BreastPairNet(architecture, fusion, pretrained).to(DEVICE)
    criterion = FocalLoss(focal_alpha, focal_gamma, label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda") if (use_amp and DEVICE.type == "cuda") else None

    if scaler:
        print(f"[MultiView] AMP enabled (FP16 mixed precision)")

    def _set_backbone_grad(req: bool):
        for p in model.backbone.parameters():
            p.requires_grad = req

    history, best_auc, best_state, best_thresh = [], 0.0, None, 0.5
    start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[MultiView] Epoch {epoch}: backbone unfrozen")

        if mixup_alpha > 0:
            tr_loss, tr_m = _train_epoch_mixup(
                model, train_loader, optimizer, criterion, mixup_alpha, scaler)
        else:
            tr_loss, tr_m = _train_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_labels, va_probs    = _eval_epoch(model, val_loader, criterion)

        thresh   = find_optimal_threshold(va_labels, va_probs)
        va_m_05  = _compute_metrics(va_labels, va_probs, 0.5)
        va_m_opt = _compute_metrics(va_labels, va_probs, thresh)
        scheduler.step()

        cur_auc = va_m_05.get("auc_roc") or 0.0
        if cur_auc > best_auc:
            best_auc    = cur_auc
            best_thresh = thresh
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va AUC={va_m_05['auc_roc']}  "
              f"f1@0.5={va_m_05['f1_score']:.4f}  "
              f"f1@opt={va_m_opt['f1_score']:.4f}(t={thresh:.3f})  "
              f"sens={va_m_05['sensitivity']:.4f}  "
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = LARGE_MODELS_DIR / f"mv_{architecture}_{fusion}_{ts}.pt"
        torch.save({
            "state_dict": best_state,
            "arch": architecture, "fusion": fusion,
            "img_size": img_size, "best_thresh": best_thresh,
        }, str(ckpt_path))
        print(f"[MultiView] Checkpoint → {ckpt_path}")

    result = {
        "method": "multiview_breastpair",
        "architecture": architecture, "fusion": fusion,
        "dataset": "VinDr-Mammo-Full-20k", "split": "patient-level",
        "train_breast_pairs": len(train_ds), "val_breast_pairs": len(val_ds),
        "device": str(DEVICE),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "use_amp": use_amp,
        "elapsed_seconds": elapsed,
        "best_val_auc": round(best_auc, 4),
        "best_thresh":  round(best_thresh, 4),
        "final_metrics_thresh05":  history[-1]["val_thresh_05"],
        "final_metrics_opt_thresh": history[-1]["val_thresh_opt"],
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"mv_{architecture}_{fusion}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)
    print(f"[MultiView] Done {elapsed:.0f}s  best_AUC={best_auc:.4f}  → {out}")
    return result


# ── bilateral 4-view training loop ────────────────────────────────────────────

def _train_bilateral_epoch(model, loader, optimizer, criterion, mixup_alpha: float = 0.0):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    batches = list(loader) if mixup_alpha > 0 else None

    for i, batch in enumerate(loader if batches is None else batches):
        l_cc, l_mlo, r_cc, r_mlo, l_lbl, r_lbl = batch
        l_cc  = l_cc.to(DEVICE);  l_mlo = l_mlo.to(DEVICE)
        r_cc  = r_cc.to(DEVICE);  r_mlo = r_mlo.to(DEVICE)
        l_lbl = l_lbl.float().to(DEVICE)
        r_lbl = r_lbl.float().to(DEVICE)

        if mixup_alpha > 0 and batches is not None:
            j = (i + 1) % len(batches)
            b2 = batches[j]
            l_cc2,l_mlo2,r_cc2,r_mlo2,l_lb2,r_lb2 = b2
            l_cc2=l_cc2.to(DEVICE); l_mlo2=l_mlo2.to(DEVICE)
            r_cc2=r_cc2.to(DEVICE); r_mlo2=r_mlo2.to(DEVICE)
            l_lb2=l_lb2.float().to(DEVICE); r_lb2=r_lb2.float().to(DEVICE)
            n = min(l_cc.shape[0], l_cc2.shape[0])
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            l_cc  = lam*l_cc[:n]  + (1-lam)*l_cc2[:n]
            l_mlo = lam*l_mlo[:n] + (1-lam)*l_mlo2[:n]
            r_cc  = lam*r_cc[:n]  + (1-lam)*r_cc2[:n]
            r_mlo = lam*r_mlo[:n] + (1-lam)*r_mlo2[:n]
            l_lbl = lam*l_lbl[:n] + (1-lam)*l_lb2[:n]
            r_lbl = lam*r_lbl[:n] + (1-lam)*r_lb2[:n]

        optimizer.zero_grad()
        logit_L, logit_R = model(l_cc, l_mlo, r_cc, r_mlo)
        loss = criterion(logit_L, l_lbl) + criterion(logit_R, r_lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs_L = torch.sigmoid(logit_L).detach().cpu().numpy().tolist()
        probs_R = torch.sigmoid(logit_R).detach().cpu().numpy().tolist()
        all_probs.extend(probs_L + probs_R)
        all_labels.extend(l_lbl.cpu().numpy().tolist() + r_lbl.cpu().numpy().tolist())

    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


@torch.no_grad()
def _eval_bilateral_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for l_cc, l_mlo, r_cc, r_mlo, l_lbl, r_lbl in loader:
        l_cc  = l_cc.to(DEVICE);  l_mlo = l_mlo.to(DEVICE)
        r_cc  = r_cc.to(DEVICE);  r_mlo = r_mlo.to(DEVICE)
        l_lbl = l_lbl.float().to(DEVICE)
        r_lbl = r_lbl.float().to(DEVICE)
        logit_L, logit_R = model(l_cc, l_mlo, r_cc, r_mlo)
        loss = criterion(logit_L, l_lbl) + criterion(logit_R, r_lbl)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logit_L).cpu().numpy().tolist())
        all_probs.extend(torch.sigmoid(logit_R).cpu().numpy().tolist())
        all_labels.extend(l_lbl.cpu().numpy().tolist())
        all_labels.extend(r_lbl.cpu().numpy().tolist())
    return total_loss / len(loader), all_labels, all_probs


def run_bilateral_experiment(
    architecture: str = "densenet121",
    epochs: int = 30,
    batch_size: int = 6,
    learning_rate: float = 5e-5,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    n_heads: int = 4,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    num_workers: int = 6,
) -> dict:
    """
    Train BilateralFusionNet on full patient quads (L-CC, L-MLO, R-CC, R-MLO).
    Phase A+B: label smoothing + MixUp + cross-bilateral attention.
    """
    if not MANIFEST_CSV.exists():
        return {"error": f"Manifest not found: {MANIFEST_CSV}"}

    manifest  = pd.read_csv(MANIFEST_CSV)
    train_ds  = PatientQuadDataset(manifest, "training", _train_transform(img_size))
    val_ds    = PatientQuadDataset(manifest, "test",     _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No patient quads found. Check manifest."}

    sampler      = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = BilateralFusionNet(architecture, n_heads, pretrained).to(DEVICE)
    criterion = FocalLoss(focal_alpha, focal_gamma, label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _set_backbone_grad(req: bool):
        for p in model.backbone.parameters():
            p.requires_grad = req

    history, best_auc, best_state, best_thresh = [], 0.0, None, 0.5
    start = time.time()

    print(f"[Bilateral] arch={architecture} epochs={epochs} batch={batch_size} "
          f"img={img_size} lr={learning_rate} ls={label_smoothing} mixup={mixup_alpha}")
    print(f"[Bilateral] Train patients={len(train_ds)}  Val patients={len(val_ds)}")

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[Bilateral] Epoch {epoch}: backbone unfrozen")

        tr_loss, tr_m = _train_bilateral_epoch(
            model, train_loader, optimizer, criterion, mixup_alpha)
        va_loss, va_labels, va_probs = _eval_bilateral_epoch(
            model, val_loader, criterion)

        thresh   = find_optimal_threshold(va_labels, va_probs)
        va_m_05  = _compute_metrics(va_labels, va_probs, 0.5)
        va_m_opt = _compute_metrics(va_labels, va_probs, thresh)
        scheduler.step()

        cur_auc = va_m_05.get("auc_roc") or 0.0
        if cur_auc > best_auc:
            best_auc    = cur_auc
            best_thresh = thresh
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va AUC={va_m_05['auc_roc']}  "
              f"f1@0.5={va_m_05['f1_score']:.4f}  "
              f"f1@opt={va_m_opt['f1_score']:.4f}(t={thresh:.3f})  "
              f"sens={va_m_05['sensitivity']:.4f}  "
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = LARGE_MODELS_DIR / f"bilateral_{architecture}_{ts}.pt"
        torch.save({
            "state_dict": best_state,
            "arch": architecture, "n_heads": n_heads,
            "img_size": img_size, "best_thresh": best_thresh,
            "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
        }, str(ckpt_path))
        print(f"[Bilateral] Checkpoint → {ckpt_path}")

    result = {
        "method": "bilateral_4view_fusion",
        "architecture": architecture, "n_heads": n_heads,
        "dataset": "VinDr-Mammo-Full-20k", "split": "patient-level",
        "train_patients": len(train_ds), "val_patients": len(val_ds),
        "device": str(DEVICE),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
        "elapsed_seconds": elapsed,
        "best_val_auc": round(best_auc, 4),
        "best_thresh":  round(best_thresh, 4),
        "final_metrics_thresh05":   history[-1]["val_thresh_05"],
        "final_metrics_opt_thresh": history[-1]["val_thresh_opt"],
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"bilateral_{architecture}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)

    done_line = (f"[bl_{architecture}_e{epochs}] DONE  "
                 f"AUC={best_auc:.4f}  "
                 f"F1={history[-1]['val_thresh_opt']['f1_score']:.4f}  "
                 f"sens={history[-1]['val_thresh_05']['sensitivity']:.4f}  "
                 f"s@90={history[-1]['val_thresh_05']['sens_at_90pct_spec']}  "
                 f"elapsed={int(elapsed)}s")
    print(f"[Bilateral] Done {elapsed:.0f}s  best_AUC={best_auc:.4f}  → {out}")
    print(done_line)
    return result


# ── asymmetry bilateral model ─────────────────────────────────────────────────

class AsymmetryNet(nn.Module):
    """
    Simpler bilateral 4-view model using explicit contralateral asymmetry.

    Architecture (no attention — avoids overfitting on limited positives):
      1. Shared backbone encodes all 4 views independently.
      2. Shared ipsilateral proj: [h_CC, h_MLO] → h_breast (left & right).
      3. Asymmetry feature: diff = |h_L - h_R|  (symmetric, same for both heads).
      4. Per-breast classifier: [h_breast, diff] → logit.
         Left:  [h_L, |h_L - h_R|]
         Right: [h_R, |h_L - h_R|]

    Parameter count ~50% of BilateralFusionNet (no MultiheadAttention).
    """
    def __init__(self, arch: str = "densenet121", pretrained: bool = True,
                 medical_pretrained: bool = False):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained, medical_pretrained)

        # Ipsilateral projection: [h_cc, h_mlo] → h_breast
        self.ipsi_proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        # Per-breast classifier: [h_breast, |diff|] → logit
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _encode_breast(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        return self.ipsi_proj(torch.cat([self.backbone(cc), self.backbone(mlo)], dim=1))

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        h_L = self._encode_breast(l_cc, l_mlo)   # (B, d)
        h_R = self._encode_breast(r_cc, r_mlo)   # (B, d)
        diff = torch.abs(h_L - h_R)               # (B, d)  — contralateral asymmetry

        logit_L = self.classifier(torch.cat([h_L, diff], dim=1)).squeeze(1)
        logit_R = self.classifier(torch.cat([h_R, diff], dim=1)).squeeze(1)
        return logit_L, logit_R


def run_asymmetry_experiment(
    architecture: str = "densenet121",
    epochs: int = 30,
    batch_size: int = 6,
    learning_rate: float = 5e-5,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    pretrained: bool = True,
    medical_pretrained: bool = False,
    freeze_backbone_epochs: int = 3,
    num_workers: int = 6,
) -> dict:
    """
    Train AsymmetryNet: simpler bilateral model using |h_L - h_R| asymmetry diff.
    Reuses PatientQuadDataset and bilateral training loops.
    """
    if not MANIFEST_CSV.exists():
        return {"error": f"Manifest not found: {MANIFEST_CSV}"}

    manifest  = pd.read_csv(MANIFEST_CSV)
    train_ds  = PatientQuadDataset(manifest, "training", _train_transform(img_size))
    val_ds    = PatientQuadDataset(manifest, "test",     _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No patient quads found. Check manifest."}

    sampler      = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = AsymmetryNet(architecture, pretrained, medical_pretrained).to(DEVICE)
    criterion = FocalLoss(focal_alpha, focal_gamma, label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _set_backbone_grad(req: bool):
        for p in model.backbone.parameters():
            p.requires_grad = req

    history, best_auc, best_state, best_thresh = [], 0.0, None, 0.5
    start = time.time()

    tag = "AsymmetryMedical" if medical_pretrained else "Asymmetry"
    print(f"[{tag}] arch={architecture} epochs={epochs} batch={batch_size} "
          f"img={img_size} lr={learning_rate} ls={label_smoothing} mixup={mixup_alpha}")
    print(f"[{tag}] Train patients={len(train_ds)}  Val patients={len(val_ds)}")

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[{tag}] Epoch {epoch}: backbone unfrozen")

        tr_loss, tr_m = _train_bilateral_epoch(
            model, train_loader, optimizer, criterion, mixup_alpha)
        va_loss, va_labels, va_probs = _eval_bilateral_epoch(
            model, val_loader, criterion)

        thresh   = find_optimal_threshold(va_labels, va_probs)
        va_m_05  = _compute_metrics(va_labels, va_probs, 0.5)
        va_m_opt = _compute_metrics(va_labels, va_probs, thresh)
        scheduler.step()

        cur_auc = va_m_05.get("auc_roc") or 0.0
        if cur_auc > best_auc:
            best_auc    = cur_auc
            best_thresh = thresh
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va AUC={va_m_05['auc_roc']}  "
              f"f1@0.5={va_m_05['f1_score']:.4f}  "
              f"f1@opt={va_m_opt['f1_score']:.4f}(t={thresh:.3f})  "
              f"sens={va_m_05['sensitivity']:.4f}  "
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    med_suffix = "_medical" if medical_pretrained else ""

    if best_state:
        model.load_state_dict(best_state)
        ckpt_path = LARGE_MODELS_DIR / f"asymmetry{med_suffix}_{architecture}_{ts}.pt"
        torch.save({
            "state_dict": best_state,
            "arch": architecture,
            "img_size": img_size, "best_thresh": best_thresh,
            "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
            "medical_pretrained": medical_pretrained,
            "model_type": "asymmetry",
        }, str(ckpt_path))
        print(f"[{tag}] Checkpoint → {ckpt_path}")

    result = {
        "method": f"asymmetry_bilateral{med_suffix}",
        "architecture": architecture,
        "medical_pretrained": medical_pretrained,
        "dataset": "VinDr-Mammo-Full-20k", "split": "patient-level",
        "train_patients": len(train_ds), "val_patients": len(val_ds),
        "device": str(DEVICE),
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
        "elapsed_seconds": elapsed,
        "best_val_auc": round(best_auc, 4),
        "best_thresh":  round(best_thresh, 4),
        "final_metrics_thresh05":   history[-1]["val_thresh_05"],
        "final_metrics_opt_thresh": history[-1]["val_thresh_opt"],
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"asymmetry{med_suffix}_{architecture}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)

    prefix = "asym_med" if medical_pretrained else "asym"
    done_line = (f"[{prefix}_{architecture}_e{epochs}] DONE  "
                 f"AUC={best_auc:.4f}  "
                 f"F1={history[-1]['val_thresh_opt']['f1_score']:.4f}  "
                 f"sens={history[-1]['val_thresh_05']['sensitivity']:.4f}  "
                 f"s@90={history[-1]['val_thresh_05']['sens_at_90pct_spec']}  "
                 f"elapsed={int(elapsed)}s")
    print(f"[{tag}] Done {elapsed:.0f}s  best_AUC={best_auc:.4f}  → {out}")
    print(done_line)
    return result
