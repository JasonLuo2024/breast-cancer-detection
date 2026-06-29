"""
Draw bounding boxes onto mammogram PNGs and save to annotated directories.

VinDr:  uses ground-truth boxes from finding_annotations.csv (no model involved)
RSNA:   uses YOLO-predicted boxes from rsna_yolo_detections.csv

Outputs:
  G:/breast-cancer-research/vindr_png_annotated/   (20,000 PNGs)
  G:/breast-cancer-research/rsna_png_annotated/    (47,652 PNGs)

Also writes updated manifests:
  G:/breast-cancer-research/split_manifest_annotated.csv
  G:/breast-cancer-research/rsna_manifest_annotated.csv

Data leakage note:
  VinDr uses dataset-provided GT boxes for BOTH train and test splits.
  RSNA uses a VinDr-pretrained YOLO detector that was NEVER trained on
  RSNA data — both RSNA train and val splits receive the same quality
  of YOLO prediction, ensuring no information leakage between splits.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
VINDR_PNG_DIR      = Path("G:/breast-cancer-research/preprocessed_png")
RSNA_PNG_DIR       = Path("G:/breast-cancer-research/rsna_png")
VINDR_ANN_DIR      = Path("G:/breast-cancer-research/vindr_png_annotated")
RSNA_ANN_DIR       = Path("G:/breast-cancer-research/rsna_png_annotated")

FINDING_CSV        = Path("F:/BreastCancerDetection/Dataset/Dicom/Vindir/finding_annotations.csv")
VINDR_MANIFEST     = Path("G:/breast-cancer-research/split_manifest.csv")
RSNA_MANIFEST      = Path("G:/breast-cancer-research/rsna_manifest.csv")
RSNA_YOLO_CSV      = Path("G:/breast-cancer-research/rsna_yolo_detections.csv")

VINDR_ANN_MANIFEST = Path("G:/breast-cancer-research/split_manifest_annotated.csv")
RSNA_ANN_MANIFEST  = Path("G:/breast-cancer-research/rsna_manifest_annotated.csv")

# Box drawing settings
BOX_COLOR     = (255, 255, 255)   # white — visible on grayscale mammograms
BOX_THICKNESS = 40                # px at full resolution; ~3px after resize to 224


def _mirror_boxes(boxes: list[dict], img_width: int) -> list[dict]:
    """Mirror x-coordinates for a horizontally flipped image."""
    return [
        {"x1": img_width - b["x2"], "y1": b["y1"],
         "x2": img_width - b["x1"], "y2": b["y2"]}
        for b in boxes
    ]


def _correct_boxes_for_flip(png_path: Path, boxes: list[dict]) -> list[dict]:
    """
    Preprocessing flips images so breast tissue is always on the right.
    GT coordinates are in original DICOM space, so if the image was flipped
    we need to mirror them.

    Detection: if tissue (bright side) is on the right but the GT box centre
    is on the left, the image was flipped — mirror the x-coords.
    """
    arr = np.array(Image.open(png_path).convert("L"))
    W = arr.shape[1]
    tissue_on_right = arr[:, W // 2:].mean() > arr[:, :W // 2].mean()
    box_cx = float(np.mean([(b["x1"] + b["x2"]) / 2 for b in boxes]))
    box_on_right = box_cx >= W / 2
    if tissue_on_right != box_on_right:
        return _mirror_boxes(boxes, W)
    return boxes


def draw_boxes(png_path: Path, boxes: list[dict], out_path: Path):
    """Draw boxes on image and save. boxes = list of {x1,y1,x2,y2}."""
    img = Image.open(png_path).convert("RGB")
    if boxes:
        draw = ImageDraw.Draw(img)
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            # Draw thick rectangle by stacking thin ones
            for t in range(BOX_THICKNESS):
                draw.rectangle(
                    [x1 - t, y1 - t, x2 + t, y2 + t],
                    outline=BOX_COLOR
                )
    img.save(str(out_path), optimize=False)


def annotate_vindr(workers: int):
    print("\n-- VinDr annotation (GT boxes) ----------------------------------")
    VINDR_ANN_DIR.mkdir(parents=True, exist_ok=True)

    fa = pd.read_csv(FINDING_CSV)
    # Only rows with actual findings (has box coordinates)
    fa_with_box = fa.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    fa_with_box = fa_with_box[fa_with_box.finding_categories != "['No Finding']"]

    # Build image_id → list of boxes
    box_map: dict[str, list[dict]] = {}
    for _, row in fa_with_box.iterrows():
        iid = row["image_id"]
        if iid not in box_map:
            box_map[iid] = []
        box_map[iid].append({
            "x1": float(row["xmin"]), "y1": float(row["ymin"]),
            "x2": float(row["xmax"]), "y2": float(row["ymax"]),
        })

    all_pngs = list(VINDR_PNG_DIR.glob("*.png"))
    print(f"  VinDr PNGs: {len(all_pngs)}  |  images with GT boxes: {len(box_map)}")

    copied, annotated = 0, 0
    for png in tqdm(all_pngs, desc="VinDr annotate"):
        out = VINDR_ANN_DIR / png.name
        image_id = png.stem
        boxes = box_map.get(image_id, [])
        if boxes:
            boxes = _correct_boxes_for_flip(png, boxes)
            draw_boxes(png, boxes, out)
            annotated += 1
        else:
            shutil.copy2(png, out)
            copied += 1

    print(f"  Annotated: {annotated}  |  Copied (no finding): {copied}")

    # Update manifest
    manifest = pd.read_csv(VINDR_MANIFEST)
    manifest["png_path"] = manifest["png_path"].str.replace(
        str(VINDR_PNG_DIR), str(VINDR_ANN_DIR), regex=False
    )
    manifest.to_csv(VINDR_ANN_MANIFEST, index=False)
    print(f"  Manifest saved: {VINDR_ANN_MANIFEST}")


def annotate_rsna():
    print("\n-- RSNA annotation (YOLO predictions) ---------------------------")
    RSNA_ANN_DIR.mkdir(parents=True, exist_ok=True)

    if not RSNA_YOLO_CSV.exists():
        print(f"  ERROR: {RSNA_YOLO_CSV} not found. Run run_yolo_inference.py first.")
        return

    det_df = pd.read_csv(RSNA_YOLO_CSV)
    # Build png_path → list of boxes
    box_map: dict[str, list[dict]] = {}
    for _, row in det_df.iterrows():
        boxes = json.loads(row["det_boxes"])
        if boxes:
            box_map[row["png_path"]] = boxes

    manifest = pd.read_csv(RSNA_MANIFEST)
    all_pngs = set(manifest["cc_png"].tolist() + manifest["mlo_png"].tolist())
    all_pngs = {Path(p) for p in all_pngs if Path(p).exists()}
    print(f"  RSNA PNGs in manifest: {len(all_pngs)}  |  with YOLO detections: {len(box_map)}")

    annotated, copied = 0, 0
    for png in tqdm(all_pngs, desc="RSNA annotate"):
        out = RSNA_ANN_DIR / png.name
        boxes = box_map.get(str(png), [])
        if boxes:
            draw_boxes(png, boxes, out)
            annotated += 1
        else:
            shutil.copy2(png, out)
            copied += 1

    print(f"  Annotated: {annotated}  |  Copied (no detection): {copied}")

    # Update manifest paths
    manifest["cc_png"]  = manifest["cc_png"].str.replace(
        str(RSNA_PNG_DIR), str(RSNA_ANN_DIR), regex=False)
    manifest["mlo_png"] = manifest["mlo_png"].str.replace(
        str(RSNA_PNG_DIR), str(RSNA_ANN_DIR), regex=False)
    manifest.to_csv(RSNA_ANN_MANIFEST, index=False)
    print(f"  Manifest saved: {RSNA_ANN_MANIFEST}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr", action="store_true", help="Annotate VinDr PNGs")
    parser.add_argument("--rsna",  action="store_true", help="Annotate RSNA PNGs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.vindr and not args.rsna:
        args.vindr = args.rsna = True  # default: do both

    if args.vindr:
        annotate_vindr(args.workers)
    if args.rsna:
        annotate_rsna()

    print("\nDone. Use --annotated flag in training scripts to use annotated manifests.")


if __name__ == "__main__":
    main()
