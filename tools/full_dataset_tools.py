"""
CNN training on the full VinDr-Mammo dataset using preprocessed PNGs from G:\.

Prerequisites:
  Run scripts/preprocess_full_dataset.py first to generate:
    G:\breast-cancer-research\preprocessed_png\{image_id}.png  (20,000 files)
    G:\breast-cancer-research\split_manifest.csv               (patient-level split)

Key differences from cnn_tools.py (biased PNG subset):
  - 16,000 train / 4,000 test  (vs 1,414 / 354)
  - Patient-level split — no leakage between train and test
  - True cancer rate ~4.9%  (vs artificial ~54% in yolov5 subset)
  - Full original resolution images, already CLAHE-preprocessed
  - num_workers=8 to keep RTX 5070 fed on 32-core i9-13900K
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

class VinDrFullDataset(Dataset):
    """
    Loads preprocessed PNGs from G:\ using the patient-level split manifest.
    Images are already CLAHE-processed and saved at original DICOM resolution.
    The DataLoader resize transform handles the final size for the model.
    """
    def __init__(self, manifest_df: pd.DataFrame, split: str = "training",
                 transform=None):
        self.transform = transform
        sub = manifest_df[manifest_df["split"] == split].copy()
        self.samples = [
            (Path(row["png_path"]), int(row["label"]))
            for _, row in sub.iterrows()
            if Path(row["png_path"]).exists()
        ]
        missing = len(sub) - len(self.samples)
        if missing:
            print(f"[VinDrFull] WARNING: {missing} PNGs missing from manifest")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
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

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── architectures ─────────────────────────────────────────────────────────────

def _build_resnet50(pretrained=True):
    w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 1))
    return m

def _build_resnet18(pretrained=True):
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m

def _build_vgg19(pretrained=True):
    w = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.vgg19(weights=w)
    m.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 1))
    return m

def _build_efficientnet_b3(pretrained=True):
    w = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b3(weights=w)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 1))
    return m

def _build_densenet121(pretrained=True):
    w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=w)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    return m

ARCHITECTURES = {
    "resnet50":        _build_resnet50,
    "resnet18":        _build_resnet18,
    "vgg19":           _build_vgg19,
    "efficientnet_b3": _build_efficientnet_b3,
    "densenet121":     _build_densenet121,
}


# ── sampler ───────────────────────────────────────────────────────────────────

def _make_sampler(dataset):
    labels = np.array(dataset.get_labels())
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(all_labels, all_probs, threshold=0.5):
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    acc  = (tp + tn) / len(labels) if len(labels) else 0
    auc = s90 = s95 = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
        spec_arr = 1.0 - fpr
        m90 = spec_arr >= 0.90; s90 = float(tpr[m90].max()) if m90.any() else 0.0
        m95 = spec_arr >= 0.95; s95 = float(tpr[m95].max()) if m95.any() else 0.0
    except Exception:
        pass
    return {
        "accuracy": round(acc, 4), "sensitivity": round(sens, 4),
        "specificity": round(spec, 4), "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
        "sens_at_90pct_spec": round(s90, 4) if not np.isnan(s90) else None,
        "sens_at_95pct_spec": round(s95, 4) if not np.isnan(s95) else None,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── train / eval loops ────────────────────────────────────────────────────────

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

def run_full_cnn_experiment(
    architecture: str = "resnet50",
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    img_size: int = 224,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pretrained: bool = True,
    freeze_backbone_epochs: int = 3,
    num_workers: int = 8,
) -> dict:
    """Train on full preprocessed VinDr-Mammo (16k train / 4k test, patient-level split)."""
    if not MANIFEST_CSV.exists():
        return {"error": f"Manifest not found at {MANIFEST_CSV}. "
                         "Run scripts/preprocess_full_dataset.py first."}
    if architecture not in ARCHITECTURES:
        return {"error": f"Unknown arch. Choose: {list(ARCHITECTURES.keys())}"}

    manifest = pd.read_csv(MANIFEST_CSV)
    train_ds = VinDrFullDataset(manifest, "training", _train_transform(img_size))
    val_ds   = VinDrFullDataset(manifest, "test",     _val_transform(img_size))

    if len(train_ds) == 0:
        return {"error": "No training images found in manifest"}

    train_counts = np.bincount(train_ds.get_labels())
    print(f"[FullCNN] arch={architecture}  device={DEVICE}  img={img_size}")
    print(f"[FullCNN] Train: {len(train_ds)} (neg={train_counts[0]}, pos={train_counts[1]})")
    print(f"[FullCNN] Val  : {len(val_ds)}")

    sampler = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = ARCHITECTURES[architecture](pretrained).to(DEVICE)
    criterion = FocalLoss(focal_alpha, focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _set_backbone_grad(req):
        for name, p in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                p.requires_grad = req

    history, best_val_f1, best_state = [], -1.0, None
    start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _set_backbone_grad(False)
        elif epoch == freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            print(f"[FullCNN] Epoch {epoch}: backbone unfrozen")

        tr_loss, tr_m = _train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_m = _eval_epoch(model, val_loader, criterion)
        scheduler.step()

        if va_m["f1_score"] > best_val_f1:
            best_val_f1 = va_m["f1_score"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} f1={tr_m['f1_score']:.4f} | "
              f"va_loss={va_loss:.4f} f1={va_m['f1_score']:.4f} "
              f"AUC={va_m['auc_roc']} sens={va_m['sensitivity']:.4f} "
              f"s@90={va_m['sens_at_90pct_spec']}")
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4), "train": tr_m, "val": va_m})

    elapsed = round(time.time() - start, 1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, LARGE_MODELS_DIR / f"full_{architecture}_{ts}.pt")

    result = {
        "method": "full_cnn", "architecture": architecture,
        "dataset": "VinDr-Mammo-Full-20k", "split": "patient-level",
        "train_images": len(train_ds), "val_images": len(val_ds),
        "device": str(DEVICE), "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "img_size": img_size,
        "focal_loss": {"alpha": focal_alpha, "gamma": focal_gamma},
        "elapsed_seconds": elapsed, "best_val_f1": round(best_val_f1, 4),
        "final_metrics": history[-1]["val"], "history": history,
        "timestamp": datetime.now().isoformat(),
    }
    out = LARGE_RESULTS_DIR / f"full_{architecture}_{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out)
    print(f"\n[FullCNN] Done {elapsed:.0f}s  best_F1={best_val_f1:.4f}  → {out}")
    return result
