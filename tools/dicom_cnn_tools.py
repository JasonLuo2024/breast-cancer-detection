"""
DICOM-native CNN training pipeline for VinDr-Mammo.

Key difference from cnn_tools.py: loads directly from
F:\\BreastCancerDetection\\Dataset\\Dicom\\Vindir\\images\\{study_id}\\{image_id}.dicom
using ALL 16,000 training images (vs 1,414 PNGs previously).

Class distribution (train split, approximate):
  negative (BI-RADS 1-3): ~15,200  positive (BI-RADS 4-5): ~800
  imbalance ratio: ~19:1  →  WeightedRandomSampler + FocalLoss

Model checkpoints and result JSON files are saved to G:\\ (552 GB free).
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

from config import VINDR_DIR, LARGE_RESULTS_DIR, LARGE_MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANCER_BIRADS = {"BI-RADS 4", "BI-RADS 5"}
DICOM_DIR = VINDR_DIR / "images"
ANN_CSV = VINDR_DIR / "breast-level_annotations.csv"


# ── DICOM loading ──────────────────────────────────────────────────────────────

def _load_dicom_as_pil(dicom_path: Path, preprocessing: str = "clahe") -> Image.Image:
    """
    Load one DICOM file → apply preprocessing → return RGB PIL Image.
    Pipeline: VOI-LUT windowing → MONOCHROME1 inversion → CLAHE → 3-channel RGB.
    """
    import pydicom
    from tools.preprocessing import PIPELINES, dicom_to_array

    dcm = pydicom.dcmread(str(dicom_path), stop_before_pixels=False)
    arr = dicom_to_array(dcm)           # float32 [0,1], shape (H, W)

    pipeline_fn = PIPELINES.get(preprocessing, PIPELINES["clahe"])
    arr = pipeline_fn(arr)              # CLAHE / zscore_clahe / standard

    uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    rgb = np.stack([uint8, uint8, uint8], axis=2)
    return Image.fromarray(rgb, mode="RGB")


# ── dataset ───────────────────────────────────────────────────────────────────

class VinDrDicomDataset(Dataset):
    """
    Loads VinDr-Mammo DICOMs on the fly from F:\\.

    Each sample is one breast image (single view — CC or MLO, L or R).
    Label: 1 if BI-RADS 4 or 5, else 0.
    """
    def __init__(self, ann_df: pd.DataFrame, split: str = "training",
                 transform=None, preprocessing: str = "clahe"):
        self.transform = transform
        self.preprocessing = preprocessing

        df = ann_df[ann_df["split"] == split].copy()
        self.samples: list[tuple[Path, int]] = []
        missing = 0
        for _, row in df.iterrows():
            dicom_path = DICOM_DIR / row["study_id"] / f"{row['image_id']}.dicom"
            if dicom_path.exists():
                label = 1 if row["breast_birads"] in CANCER_BIRADS else 0
                self.samples.append((dicom_path, label))
            else:
                missing += 1
        if missing:
            print(f"[VinDrDicom] WARNING: {missing} DICOM files not found (skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = _load_dicom_as_pil(path, self.preprocessing)
        except Exception as e:
            # Return blank image on rare corrupt DICOM
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        return [s[1] for s in self.samples]


# ── transforms ────────────────────────────────────────────────────────────────

def _train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── architectures (same as cnn_tools.py) ──────────────────────────────────────

def _build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    return model


def _build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def _build_vgg19(pretrained: bool = True) -> nn.Module:
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)
    model.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 1))
    return model


def _build_efficientnet_b3(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 1))
    return model


def _build_densenet121(pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


ARCHITECTURES = {
    "resnet50":        _build_resnet50,
    "resnet18":        _build_resnet18,
    "vgg19":           _build_vgg19,
    "efficientnet_b3": _build_efficientnet_b3,
    "densenet121":     _build_densenet121,
}


# ── weighted sampler ──────────────────────────────────────────────────────────

def _make_sampler(dataset: VinDrDicomDataset) -> WeightedRandomSampler:
    labels = np.array(dataset.get_labels())
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(all_labels, all_probs, threshold=0.5):
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0

    auc = sens_at_90 = sens_at_95 = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
        spec_arr = 1.0 - fpr
        mask90 = spec_arr >= 0.90
        sens_at_90 = float(tpr[mask90].max()) if mask90.any() else 0.0
        mask95 = spec_arr >= 0.95
        sens_at_95 = float(tpr[mask95].max()) if mask95.any() else 0.0
    except Exception:
        pass

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
        "sens_at_90pct_spec": round(sens_at_90, 4) if not np.isnan(sens_at_90) else None,
        "sens_at_95pct_spec": round(sens_at_95, 4) if not np.isnan(sens_at_95) else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── training / evaluation loops ───────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


# ── public API ────────────────────────────────────────────────────────────────

def run_dicom_cnn_experiment(
    architecture: str = "resnet50",
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    preprocessing: str = "clahe",
    num_workers: int = 2,
) -> dict:
    """
    Train a CNN on full VinDr-Mammo DICOMs (16,000 train / 4,000 test).
    Saves model weights and result JSON to G:\\.
    """
    if architecture not in ARCHITECTURES:
        return {"error": f"Unknown architecture. Choose from: {list(ARCHITECTURES.keys())}"}

    if not ANN_CSV.exists():
        return {"error": f"Annotations not found at {ANN_CSV}"}

    if not DICOM_DIR.exists():
        return {"error": f"DICOM directory not found at {DICOM_DIR}"}

    ann_df = pd.read_csv(ANN_CSV)

    train_ds = VinDrDicomDataset(ann_df, split="training",
                                  transform=_train_transform(img_size),
                                  preprocessing=preprocessing)
    val_ds = VinDrDicomDataset(ann_df, split="test",
                                transform=_val_transform(img_size),
                                preprocessing=preprocessing)

    if len(train_ds) == 0:
        return {"error": "No labelled DICOM training images found"}

    train_counts = np.bincount(train_ds.get_labels())
    print(f"[DICOM-CNN] Train: {len(train_ds)} images  "
          f"(neg={train_counts[0]}, pos={train_counts[1] if len(train_counts) > 1 else 0})")
    print(f"[DICOM-CNN] Val:   {len(val_ds)} images")
    print(f"[DICOM-CNN] Device: {DEVICE}  |  arch={architecture}  |  prep={preprocessing}")

    sampler = _make_sampler(train_ds)
    # persistent_workers keeps worker processes alive between epochs (no respawn overhead)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers > 0), prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=(num_workers > 0), prefetch_factor=2)

    model = ARCHITECTURES[architecture](pretrained).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _set_backbone_grad(req_grad: bool):
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = req_grad

    history = []
    best_val_f1 = -1.0
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[DICOM-CNN] Epoch {epoch}: backbone unfrozen")

        train_loss, train_m = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_m = _eval_epoch(model, val_loader, criterion)
        scheduler.step()

        if val_m["f1_score"] > best_val_f1:
            best_val_f1 = val_m["f1_score"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} f1={train_m['f1_score']:.4f} | "
            f"val_loss={val_loss:.4f} f1={val_m['f1_score']:.4f} "
            f"AUC={val_m['auc_roc']} "
            f"sens={val_m['sensitivity']:.4f} s@90={val_m['sens_at_90pct_spec']}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train": train_m,
            "val": val_m,
        })

    elapsed = round(time.time() - start, 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = LARGE_MODELS_DIR / f"dicom_{architecture}_{ts}.pt"
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, weights_path)

    final_val = history[-1]["val"]
    result = {
        "method": "dicom_cnn",
        "architecture": architecture,
        "dataset": "VinDr-Mammo-DICOM",
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "device": str(DEVICE),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "preprocessing": preprocessing,
        "elapsed_seconds": elapsed,
        "best_val_f1": round(best_val_f1, 4),
        "final_metrics": final_val,
        "history": history,
        "weights_saved": str(weights_path),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = LARGE_RESULTS_DIR / f"dicom_{architecture}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out_path)
    print(f"\n[DICOM-CNN] Done in {elapsed:.0f}s. "
          f"Best val F1={best_val_f1:.4f}. Saved → {out_path}")
    return result
