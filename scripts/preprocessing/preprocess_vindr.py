"""
Preprocess all 20,000 VinDr-Mammo DICOMs → PNG.

Pipeline per image:
  1. DICOM windowing (VOI-LUT / window-centre)
  2. MONOCHROME1 inversion
  3. CLAHE (clip=2.0, tile=8x8) — enhances microcalcifications/masses
  4. Save grayscale PNG at original resolution to G:\

Outputs:
  G:\breast-cancer-research\preprocessed_png\{image_id}.png  (20,000 files ~15-30 GB)
  G:\breast-cancer-research\split_manifest.csv               (patient-level train/test split)

Patient-level split: uses official 'split' column from breast-level_annotations.csv.
VinDr-Mammo official split is patient-level (4,000 train patients / 1,000 test patients).
No patient appears in both train and test.

Usage:
  python scripts/preprocess_full_dataset.py
  python scripts/preprocess_full_dataset.py --workers 20   # override parallelism
  python scripts/preprocess_full_dataset.py --resume       # skip already-done files
"""
import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DICOM_ROOT   = Path(r"F:\BreastCancerDetection\Dataset\Dicom\Vindir\images")
ANN_CSV      = Path(r"F:\BreastCancerDetection\Dataset\Dicom\Vindir\breast-level_annotations.csv")
OUT_DIR      = Path(r"G:\breast-cancer-research\preprocessed_png")
MANIFEST_CSV = Path(r"G:\breast-cancer-research\split_manifest.csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── per-image worker (runs in subprocess) ─────────────────────────────────────

def process_one(row: dict) -> dict:
    """
    Load one DICOM, apply windowing + CLAHE, save PNG.
    Returns {"image_id": ..., "ok": True/False, "error": ...}.
    Called in a worker process — must be importable at top level.
    """
    image_id = row["image_id"]
    study_id = row["study_id"]
    out_path = OUT_DIR / f"{image_id}.png"

    try:
        import cv2
        import pydicom

        dicom_path = DICOM_ROOT / study_id / f"{image_id}.dicom"
        if not dicom_path.exists():
            return {"image_id": image_id, "ok": False, "error": "file not found"}

        # ── 1. Load DICOM pixel array ──────────────────────────────────────────
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=False)
        img = ds.pixel_array.astype(np.float32)

        # ── 2. VOI-LUT / windowing ─────────────────────────────────────────────
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(wc[0]) if hasattr(wc, "__len__") else float(wc)
            ww = float(ww[0]) if hasattr(ww, "__len__") else float(ww)
            lo, hi = wc - ww / 2, wc + ww / 2
            img = np.clip(img, lo, hi)

        # ── 3. Invert MONOCHROME1 ──────────────────────────────────────────────
        pi = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()
        if pi == "MONOCHROME1":
            img = img.max() - img

        # ── 4. Normalise to 0-255 uint8 ────────────────────────────────────────
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img_u8 = (img * 255).astype(np.uint8)

        # ── 5. CLAHE ───────────────────────────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_u8 = clahe.apply(img_u8)

        # ── 6. Flip breast to standard right-facing orientation ────────────────
        left_mean  = img_u8[:, : img_u8.shape[1] // 2].mean()
        right_mean = img_u8[:, img_u8.shape[1] // 2 :].mean()
        if left_mean > right_mean:
            img_u8 = np.fliplr(img_u8)

        # ── 7. Save PNG (grayscale, original resolution) ───────────────────────
        cv2.imwrite(str(out_path), img_u8)
        return {"image_id": image_id, "ok": True, "error": None}

    except Exception as e:
        return {"image_id": image_id, "ok": False,
                "error": f"{type(e).__name__}: {e}"}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker processes (default 16)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already saved to OUT_DIR")
    args = parser.parse_args()

    print(f"Loading annotations from {ANN_CSV}...")
    df = pd.read_csv(ANN_CSV)
    print(f"Total images in manifest: {len(df)}")
    print(f"  Training: {(df.split=='training').sum()}  |  Test: {(df.split=='test').sum()}")

    # Verify official split is patient-level clean
    train_pts = set(df[df.split == "training"]["study_id"])
    test_pts  = set(df[df.split == "test"]["study_id"])
    leak = train_pts & test_pts
    if leak:
        print(f"WARNING: {len(leak)} patients appear in both splits!")
    else:
        print(f"Split verified patient-level clean: "
              f"{len(train_pts)} train patients / {len(test_pts)} test patients")

    # Print cancer distribution
    for s in ["training", "test"]:
        sub = df[df.split == s]
        pos = sub["breast_birads"].isin({"BI-RADS 4", "BI-RADS 5"}).sum()
        print(f"  {s}: {pos}/{len(sub)} cancer ({100*pos/len(sub):.1f}%)")

    rows = df.to_dict("records")

    if args.resume:
        done = {p.stem for p in OUT_DIR.glob("*.png")}
        rows = [r for r in rows if r["image_id"] not in done]
        print(f"Resume: {len(done)} already done, {len(rows)} remaining")

    print(f"\nProcessing {len(rows)} DICOMs with {args.workers} workers...")
    print(f"Output: {OUT_DIR}\n")

    ok_count = err_count = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, r): r["image_id"] for r in rows}
        total = len(futures)
        done_n = 0
        for fut in as_completed(futures):
            done_n += 1
            res = fut.result()
            if res["ok"]:
                ok_count += 1
            else:
                err_count += 1
                errors.append((res["image_id"], res["error"]))
            if done_n % 500 == 0 or done_n == total:
                pct = 100 * done_n / total
                print(f"  [{done_n}/{total}  {pct:.0f}%]  ok={ok_count}  err={err_count}")

    print(f"\nDone. Saved {ok_count} PNGs to {OUT_DIR}")
    if errors:
        print(f"  {err_count} errors:")
        for img_id, msg in errors[:10]:
            print(f"    {img_id}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    # Write split manifest CSV
    out_rows = []
    for r in df.to_dict("records"):
        png_path = OUT_DIR / f"{r['image_id']}.png"
        if png_path.exists():
            out_rows.append({
                "image_id":      r["image_id"],
                "study_id":      r["study_id"],
                "split":         r["split"],
                "view_position": r["view_position"],
                "laterality":    r["laterality"],
                "breast_birads": r["breast_birads"],
                "breast_density":r["breast_density"],
                "label":         1 if r["breast_birads"] in ("BI-RADS 4", "BI-RADS 5") else 0,
                "png_path":      str(png_path),
            })

    manifest_df = pd.DataFrame(out_rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    print(f"Manifest written: {MANIFEST_CSV}  ({len(manifest_df)} rows)")

    # Summary stats
    for s in ["training", "test"]:
        sub = manifest_df[manifest_df.split == s]
        pos = sub.label.sum()
        print(f"  Manifest {s}: {len(sub)} images, {pos} cancer ({100*pos/len(sub):.1f}%), "
              f"{sub.study_id.nunique()} patients")


if __name__ == "__main__":
    main()
