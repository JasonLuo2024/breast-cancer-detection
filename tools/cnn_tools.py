"""
CNN experiments for breast cancer detection on VinDr-Mammo.

# Suppress OpenMP duplicate-library warning from mixing conda/pip torch installs.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


Architectures: CustomCNN, ResNet18, EfficientNetB0, DenseNet121
Loss: FocalLoss (handles severe class imbalance)
Input: PNG mammograms from yolov5_dataset/images/{train,val}
Labels: breast-level BI-RADS (binary: 0=normal, 1=cancer/BI-RADS4+5)
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from config import RESULTS_DIR, VINDR_DIR, LARGE_RESULTS_DIR, LARGE_MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANCER_BIRADS = {"BI-RADS 4", "BI-RADS 5"}
IMG_DIR = VINDR_DIR / "yolov5_dataset" / "images"


# ── label loading ─────────────────────────────────────────────────────────────

def _load_labels(ann_path: Path) -> dict[str, int]:
    """Map image_id → binary label (1=cancer)."""
    labels = {}
    with open(ann_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["image_id"]] = 1 if row["breast_birads"] in CANCER_BIRADS else 0
    return labels


# ── dataset ───────────────────────────────────────────────────────────────────

class VinDrDataset(Dataset):
    def __init__(self, image_dir: Path, labels: dict[str, int],
                 transform=None, preprocessing: str = "standard"):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        self.preprocessing = preprocessing

        for png in image_dir.glob("*.png"):
            img_id = png.stem
            if img_id in labels:
                self.samples.append((png, labels[img_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)

        # Apply paper-recommended preprocessing before augmentation
        if self.preprocessing != "none":
            try:
                from tools.preprocessing import preprocess_png
                img = preprocess_png(img, pipeline=self.preprocessing)
            except Exception:
                img = img.convert("RGB")
        else:
            img = img.convert("RGB")

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
    """Binary focal loss — down-weights easy negatives, focuses on hard cases."""
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
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ── architectures ─────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """Lightweight 4-block CNN built from scratch."""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))  # [B, 1] — squeezed in training loop


def _build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def _build_resnet50(pretrained: bool = True) -> nn.Module:
    """ResNet-50: strongest single backbone in Añez et al. 2025 (AUC 0.95 on CBIS-DDSM)."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


def _build_vgg19(pretrained: bool = True) -> nn.Module:
    """VGG19: topped the 6-dataset multi-view benchmark at AUC 0.9051 (Abdikenov et al. 2025)."""
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 1),
    )
    return model


def _build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def _build_efficientnet_b3(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B3: best parameter-efficiency Pareto (Añez et al. 2025)."""
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )
    return model


def _build_densenet121(pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


ARCHITECTURES = {
    "custom_cnn":      lambda: CustomCNN(),
    "resnet18":        _build_resnet18,
    "resnet50":        _build_resnet50,       # SOTA backbone per literature
    "vgg19":           _build_vgg19,           # Best multi-view AUC 0.9051 per literature
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet_b3": _build_efficientnet_b3, # Best efficiency per literature
    "densenet121":     _build_densenet121,
}


# ── weighted sampler ──────────────────────────────────────────────────────────

def _make_sampler(dataset: VinDrDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── training / evaluation ─────────────────────────────────────────────────────

def _compute_metrics(all_labels, all_probs, threshold=0.5):
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / len(labels) if len(labels) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0

    auc = float(np.nan)
    sens_at_90spec = float(np.nan)
    sens_at_95spec = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels, probs))
        # Sensitivity at fixed specificity operating points (clinical standard)
        fpr, tpr, _ = roc_curve(labels, probs)
        spec_arr = 1.0 - fpr
        # At ≥90% specificity: highest sensitivity achievable
        mask_90 = spec_arr >= 0.90
        sens_at_90spec = float(tpr[mask_90].max()) if mask_90.any() else 0.0
        mask_95 = spec_arr >= 0.95
        sens_at_95spec = float(tpr[mask_95].max()) if mask_95.any() else 0.0
    except Exception:
        pass

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
        "sens_at_90pct_spec": round(sens_at_90spec, 4) if not np.isnan(sens_at_90spec) else None,
        "sens_at_95pct_spec": round(sens_at_95spec, 4) if not np.isnan(sens_at_95spec) else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)  # [B,1] → [B]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        logits = model(images).squeeze(1)  # [B,1] → [B]
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), _compute_metrics(all_labels, all_probs)


# ── public API ────────────────────────────────────────────────────────────────

def get_vindr_cnn_info() -> dict:
    """Return info about the CNN experiment setup."""
    ann = VINDR_DIR / "breast-level_annotations.csv"
    train_dir = IMG_DIR / "train"
    val_dir = IMG_DIR / "val"
    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "train_images": len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0,
        "val_images": len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0,
        "annotations_found": ann.exists(),
        "available_architectures": list(ARCHITECTURES.keys()),
        "loss_function": "FocalLoss(alpha=0.25, gamma=2) — designed for class imbalance",
        "imbalance_handling": "WeightedRandomSampler + FocalLoss + class-weighted metrics",
    }


def run_cnn_experiment(
    architecture: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    preprocessing: str = "standard",
) -> dict:
    """
    Train a CNN model on VinDr-Mammo PNGs for binary cancer classification.

    Args:
        architecture: one of custom_cnn, resnet18, efficientnet_b0, densenet121
        epochs: total training epochs
        batch_size: images per batch
        learning_rate: initial Adam LR
        img_size: resize images to this square size
        focal_alpha: FocalLoss alpha (positive class weight)
        focal_gamma: FocalLoss gamma (focusing parameter)
        pretrained: use ImageNet weights for transfer learning
        freeze_backbone_epochs: freeze pretrained layers for first N epochs
    """
    if architecture not in ARCHITECTURES:
        return {"error": f"Unknown architecture. Choose from: {list(ARCHITECTURES.keys())}"}

    ann = VINDR_DIR / "breast-level_annotations.csv"
    if not ann.exists():
        return {"error": f"Annotations not found at {ann}"}

    train_dir = IMG_DIR / "train"
    val_dir = IMG_DIR / "val"
    if not train_dir.exists():
        return {"error": f"Train images not found at {train_dir}"}

    labels = _load_labels(ann)

    train_ds = VinDrDataset(train_dir, labels, transform=_train_transform(img_size),
                            preprocessing=preprocessing)
    val_ds = VinDrDataset(val_dir, labels, transform=_val_transform(img_size),
                          preprocessing=preprocessing) if val_dir.exists() else None

    if len(train_ds) == 0:
        return {"error": "No labelled training images found"}

    sampler = _make_sampler(train_ds)
    # num_workers=4: keep GPU fed with parallel preprocessing on 32-core i9-13900K
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=torch.cuda.is_available(),
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=torch.cuda.is_available(),
                            persistent_workers=True) if val_ds and len(val_ds) else None

    model = ARCHITECTURES[architecture]().to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Freeze backbone for first N epochs (transfer learning warmup)
    def _set_backbone_grad(requires_grad: bool):
        if architecture == "custom_cnn":
            return
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = requires_grad

    history = []
    best_val_f1 = -1.0
    best_state = None

    print(f"\n[CNN] Architecture: {architecture}  |  Device: {DEVICE}  |  Epochs: {epochs}")
    print(f"[CNN] Train: {len(train_ds)} images  |  Val: {len(val_ds) if val_ds else 0} images")

    train_label_counts = np.bincount(train_ds.get_labels())
    print(f"[CNN] Train label dist → negative: {train_label_counts[0] if len(train_label_counts) > 0 else 0}, "
          f"positive: {train_label_counts[1] if len(train_label_counts) > 1 else 0}")

    start = time.time()

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == freeze_backbone_epochs + 1 and architecture != "custom_cnn":
            _set_backbone_grad(True)
            print(f"[CNN] Epoch {epoch}: backbone unfrozen")
        elif epoch == 1 and architecture != "custom_cnn":
            _set_backbone_grad(False)

        train_loss, train_metrics = _train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train": train_metrics,
        }

        if val_loader:
            val_loss, val_metrics = _eval_epoch(model, val_loader, criterion)
            entry["val_loss"] = round(val_loss, 4)
            entry["val"] = val_metrics

            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} f1={train_metrics['f1_score']:.4f} | "
                f"val_loss={val_loss:.4f} f1={val_metrics['f1_score']:.4f} "
                f"sens={val_metrics['sensitivity']:.4f} spec={val_metrics['specificity']:.4f}"
            )
        else:
            print(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} f1={train_metrics['f1_score']:.4f} "
                f"sens={train_metrics['sensitivity']:.4f}"
            )

        history.append(entry)

    elapsed = round(time.time() - start, 1)

    # Save best model weights to G:\ (large files)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = LARGE_MODELS_DIR / f"cnn_{architecture}_{ts}.pt"
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, weights_path)

    # Final validation metrics
    final_val = history[-1].get("val", history[-1].get("train"))

    result = {
        "method": "cnn",
        "architecture": architecture,
        "pretrained": pretrained,
        "dataset": "VinDr-Mammo",
        "device": str(DEVICE),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "preprocessing": preprocessing,
        "train_images": len(train_ds),
        "val_images": len(val_ds) if val_ds else 0,
        "elapsed_seconds": elapsed,
        "best_val_f1": round(best_val_f1, 4),
        "final_metrics": final_val,
        "history": history,
        "weights_saved": str(weights_path),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = LARGE_RESULTS_DIR / f"cnn_{architecture}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out_path)
    print(f"\n[CNN] Done in {elapsed}s. Best val F1={best_val_f1:.4f}. Saved → {out_path}")
    return result


def run_cnn_benchmark(
    epochs: int = 15,
    batch_size: int = 16,
) -> dict:
    """
    Benchmark all CNN architectures and return a ranked comparison.
    Runs: CustomCNN, ResNet18, EfficientNetB0, DenseNet121.
    """
    results = {}
    ranking = []

    for arch in ARCHITECTURES:
        print(f"\n{'='*60}")
        print(f"RUNNING: {arch}")
        print(f"{'='*60}")
        res = run_cnn_experiment(architecture=arch, epochs=epochs, batch_size=batch_size)
        if "error" not in res:
            results[arch] = res["final_metrics"]
            ranking.append({
                "architecture": arch,
                "f1_score": res.get("best_val_f1", res["final_metrics"].get("f1_score", 0)),
                "sensitivity": res["final_metrics"].get("sensitivity", 0),
                "specificity": res["final_metrics"].get("specificity", 0),
                "auc_roc": res["final_metrics"].get("auc_roc"),
                "elapsed_seconds": res.get("elapsed_seconds"),
            })
        else:
            print(f"  [ERROR] {res['error']}")

    ranking.sort(key=lambda x: x["f1_score"], reverse=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "method": "cnn_benchmark",
        "dataset": "VinDr-Mammo",
        "epochs": epochs,
        "results": results,
        "ranking": ranking,
        "best_architecture": ranking[0]["architecture"] if ranking else None,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / f"cnn_benchmark_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    summary["saved_to"] = str(out_path)
    return summary
