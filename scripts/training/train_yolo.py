"""
Train YOLOv8 lesion detector on VinDr-Mammo.

Dataset:
  Positives (with boxes): F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/images/train
  Negatives (no finding): G:/breast-cancer-research/preprocessed_png  (training split only)
  Val:                    F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/images/val

Output:
  Model saved to G:/breast-cancer-research/models/yolo_vindr/
  Run RSNA inference with scripts/run_yolo_inference.py
"""
import os
import random
import shutil
import yaml
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ── paths ─────────────────────────────────────────────────────────────────────
POS_TRAIN_IMGS = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/images/train")
POS_TRAIN_LABS = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/labels/train")
VAL_IMGS       = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/images/val")
VAL_LABS       = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/yolov5_dataset/labels/val")
FINDING_CSV    = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/finding_annotations.csv")
MANIFEST_CSV   = Path("G:/breast-cancer-research/split_manifest.csv")
PREPROCESSED   = Path("G:/breast-cancer-research/preprocessed_png")
DATASET_DIR    = Path("G:/breast-cancer-research/yolo_dataset")
MODEL_DIR      = Path("G:/breast-cancer-research/models/yolo_vindr")

NEG_RATIO = 3
SEED      = 42


def build_dataset():
    random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    pos_imgs = list(POS_TRAIN_IMGS.glob("*.png"))
    print(f"  Positive train images: {len(pos_imgs)}")
    for img in pos_imgs:
        dst = DATASET_DIR / "images/train" / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        lab = POS_TRAIN_LABS / img.with_suffix(".txt").name
        if lab.exists():
            dst_lab = DATASET_DIR / "labels/train" / lab.name
            if not dst_lab.exists():
                shutil.copy2(lab, dst_lab)

    val_imgs = list(VAL_IMGS.glob("*.png"))
    print(f"  Val images: {len(val_imgs)}")
    for img in val_imgs:
        dst = DATASET_DIR / "images/val" / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        lab = VAL_LABS / img.with_suffix(".txt").name
        if lab.exists():
            dst_lab = DATASET_DIR / "labels/val" / lab.name
            if not dst_lab.exists():
                shutil.copy2(lab, dst_lab)

    manifest = pd.read_csv(MANIFEST_CSV)
    fa = pd.read_csv(FINDING_CSV)
    pos_ids  = set(fa[fa.finding_categories != "['No Finding']"]["image_id"].unique())
    neg_train = manifest[
        (manifest.split == "training") & (~manifest.image_id.isin(pos_ids))
    ]["image_id"].tolist()

    n_neg      = min(len(neg_train), len(pos_imgs) * NEG_RATIO)
    neg_sample = random.sample(neg_train, n_neg)
    print(f"  Negative train images (No Finding sample): {len(neg_sample)}")

    copied = 0
    for img_id in neg_sample:
        src = PREPROCESSED / f"{img_id}.png"
        dst = DATASET_DIR / "images/train" / f"{img_id}.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} negative images")

    data = {
        "path":  str(DATASET_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["lesion"],
    }
    yaml_path = DATASET_DIR / "vindr_combined.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    total = len(list((DATASET_DIR / "images/train").glob("*.png")))
    print(f"  Total train images: {total}  (pos={len(pos_imgs)}, neg={copied})")
    return yaml_path


def main():
    print("Building combined YOLO dataset...")
    yaml_path = build_dataset()

    print("\nStarting YOLOv8s training...")
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — prevents hang on headless Windows
    model = YOLO("yolov8s.pt")

    model.train(
        data         = str(yaml_path),
        epochs       = 50,
        imgsz        = 640,
        batch        = 8,
        workers      = 0,           # 0 = main-process loading (avoids Windows spawn deadlock)
        project      = str(MODEL_DIR),
        name         = "yolov8s_vindr",
        exist_ok     = True,
        patience     = 15,
        lr0          = 1e-3,
        lrf          = 0.01,
        warmup_epochs = 3,
        hsv_h        = 0.0,
        hsv_s        = 0.0,
        hsv_v        = 0.4,
        fliplr       = 0.5,
        flipud       = 0.0,
        mosaic       = 0.5,
        close_mosaic = 10,
        amp          = True,
        device       = 0,
        save         = True,
        save_period  = 10,
        val          = True,
        plots        = False,   # disabled — matplotlib Agg may still be slow
        verbose      = True,
    )
    print(f"\nTraining complete. Best model: {MODEL_DIR}/yolov8s_vindr/weights/best.pt")


if __name__ == "__main__":
    main()
