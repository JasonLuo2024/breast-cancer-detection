"""
Run trained YOLOv8 lesion detector on RSNA PNG images.

Reads:  G:/breast-cancer-research/rsna_manifest.csv  (CC+MLO pair manifest)
        G:/breast-cancer-research/models/yolo_vindr/yolov8s_vindr/weights/best.pt

Writes: G:/breast-cancer-research/rsna_yolo_detections.csv
  Columns: image_id, png_path, det_count, max_conf, det_boxes (JSON list of
           dicts {x1,y1,x2,y2,conf}), has_lesion (conf>threshold)
"""
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

MANIFEST      = Path("G:/breast-cancer-research/rsna_manifest.csv")
MODEL_PT      = Path("G:/breast-cancer-research/models/yolo_vindr/yolov8s_vindr/weights/best.pt")
DET_CSV_OUT   = Path("G:/breast-cancer-research/rsna_yolo_detections.csv")
CONF_THRESH   = 0.25   # YOLOv8 default; adjust based on val mAP50
IOU_THRESH    = 0.45
BATCH_SIZE    = 32     # images per YOLO call
IMGSZ         = 640


def run_inference(model: YOLO, paths: list[str]) -> list[dict]:
    results = model(paths, imgsz=IMGSZ, conf=CONF_THRESH, iou=IOU_THRESH,
                    verbose=False, stream=False)
    rows = []
    for path, r in zip(paths, results):
        boxes = r.boxes
        if boxes is not None and len(boxes):
            dets = [
                {"x1": float(b[0]), "y1": float(b[1]),
                 "x2": float(b[2]), "y2": float(b[3]),
                 "conf": float(c)}
                for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())
            ]
            max_conf   = float(np.max([d["conf"] for d in dets]))
            det_count  = len(dets)
        else:
            dets      = []
            max_conf  = 0.0
            det_count = 0
        rows.append({
            "png_path":  path,
            "det_count": det_count,
            "max_conf":  round(max_conf, 4),
            "det_boxes": json.dumps(dets),
            "has_lesion": int(max_conf >= CONF_THRESH),
        })
    return rows


def main():
    if not MODEL_PT.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PT}\n"
            "Run scripts/train_yolo.py first and wait for it to complete."
        )

    manifest = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(manifest)} breast pairs  ({manifest.cancer.sum()} cancer)")

    # Collect unique PNG paths (each pair has cc_png and mlo_png)
    cc_paths  = manifest[["cc_image_id",  "cc_png"]].rename(
        columns={"cc_image_id": "image_id", "cc_png": "png_path"})
    mlo_paths = manifest[["mlo_image_id", "mlo_png"]].rename(
        columns={"mlo_image_id": "image_id", "mlo_png": "png_path"})
    all_images = pd.concat([cc_paths, mlo_paths]).drop_duplicates("image_id").reset_index(drop=True)
    print(f"Unique images to process: {len(all_images)}")

    model = YOLO(str(MODEL_PT))
    model.fuse()

    records = []
    for _, row in tqdm(all_images.iterrows(), total=len(all_images), desc="YOLO inference"):
        png = row["png_path"]
        if not Path(png).exists():
            records.append({"png_path": png, "det_count": 0, "max_conf": 0.0,
                            "det_boxes": "[]", "has_lesion": 0, "image_id": row["image_id"]})
            continue
        try:
            res = run_inference(model, [png])
            res[0]["image_id"] = row["image_id"]
            records.extend(res)
        except Exception:
            records.append({"png_path": png, "det_count": 0, "max_conf": 0.0,
                            "det_boxes": "[]", "has_lesion": 0, "image_id": row["image_id"]})

    det_df = pd.DataFrame(records)

    det_df.to_csv(DET_CSV_OUT, index=False)
    print(f"\nDetections saved: {DET_CSV_OUT}")
    print(f"  Total images: {len(det_df)}")
    print(f"  Images with lesion (conf>{CONF_THRESH}): {det_df.has_lesion.sum()}")
    print(f"  Mean max confidence: {det_df.max_conf.mean():.4f}")
    print(f"  Detection rate: {det_df.has_lesion.mean()*100:.1f}%")

    # Merge detections back to manifest
    manifest = manifest.merge(
        det_df[["image_id", "max_conf", "has_lesion"]].rename(
            columns={"image_id": "cc_image_id", "max_conf": "cc_yolo_conf",
                     "has_lesion": "cc_has_lesion"}),
        on="cc_image_id", how="left"
    ).merge(
        det_df[["image_id", "max_conf", "has_lesion"]].rename(
            columns={"image_id": "mlo_image_id", "max_conf": "mlo_yolo_conf",
                     "has_lesion": "mlo_has_lesion"}),
        on="mlo_image_id", how="left"
    )
    manifest["yolo_any_lesion"] = (
        manifest["cc_has_lesion"].fillna(0).astype(int) |
        manifest["mlo_has_lesion"].fillna(0).astype(int)
    )
    manifest.to_csv(MANIFEST, index=False)
    print(f"\nManifest updated with YOLO detections: {MANIFEST}")
    print(f"  YOLO lesion on any view: {manifest.yolo_any_lesion.sum()} / {len(manifest)} pairs")
    # Cross-tab YOLO detection vs ground-truth cancer
    ct = pd.crosstab(manifest.cancer, manifest.yolo_any_lesion,
                     rownames=["cancer"], colnames=["yolo_any_lesion"])
    print("\nYOLO detection vs ground-truth cancer:\n", ct.to_string())


if __name__ == "__main__":
    main()
