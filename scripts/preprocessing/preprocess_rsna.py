"""
Preprocess RSNA Breast Cancer Detection dataset.

Converts DICOM → 8-bit PNG (same normalization as VinDr pipeline).
Builds a CC+MLO pair manifest compatible with BreastPairDataset.
Splits patients 80/20 into rsna_train / rsna_val by patient_id.

Output:
  G:/breast-cancer-research/rsna_png/          (PNGs)
  G:/breast-cancer-research/rsna_manifest.csv  (pair manifest)
"""
import argparse, os, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

RSNA_ROOT    = Path("G:/breast-cancer-research/rsna")
PNG_DIR      = Path("G:/breast-cancer-research/rsna_png")
MANIFEST_OUT = Path("G:/breast-cancer-research/rsna_manifest.csv")
TRAIN_CSV    = RSNA_ROOT / "train.csv"
TRAIN_IMGS   = RSNA_ROOT / "train_images"
SEED         = 42
VAL_FRAC     = 0.20
MAX_WORKERS  = 8


def dcm_to_png(dcm_path: Path, out_path: Path) -> bool:
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)
        # Handle PhotometricInterpretation (MONOCHROME1 = inverted)
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr
        # Normalize to 0-255
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        else:
            arr = np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))
        img.save(str(out_path), optimize=False)
        return True
    except Exception as e:
        print(f"  ERROR {dcm_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume",  action="store_true", help="Skip already-converted PNGs")
    args = parser.parse_args()

    PNG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    print(f"RSNA train.csv: {len(df)} rows, {df.patient_id.nunique()} patients")
    print(f"Cancer distribution:\n{df.cancer.value_counts().to_string()}")
    print(f"View distribution:\n{df.view.value_counts().to_string()}\n")

    # ── convert DICOMs to PNG ──────────────────────────────────────────────────
    tasks = []
    for _, row in df.iterrows():
        dcm = TRAIN_IMGS / str(row.patient_id) / f"{row.image_id}.dcm"
        png = PNG_DIR / f"{row.image_id}.png"
        if args.resume and png.exists():
            continue
        tasks.append((dcm, png))

    print(f"Converting {len(tasks)} DICOMs to PNG ({args.workers} workers)...")
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(dcm_to_png, d, p): (d, p) for d, p in tasks}
        for f in tqdm(as_completed(futs), total=len(futs)):
            if f.result():
                ok += 1
            else:
                fail += 1
    print(f"Done: {ok} OK, {fail} failed\n")

    # ── build CC+MLO pair manifest ─────────────────────────────────────────────
    # RSNA view values: "CC", "MLO"  (and rarely "LM", "ML" etc — skip those)
    df_valid = df[df.view.isin(["CC", "MLO"])].copy()
    df_valid["png_path"] = df_valid.image_id.apply(
        lambda x: str(PNG_DIR / f"{x}.png"))

    # Patient-level 80/20 split
    random.seed(SEED)
    patients = df_valid.patient_id.unique().tolist()
    random.shuffle(patients)
    n_val = int(len(patients) * VAL_FRAC)
    val_pats = set(patients[:n_val])
    df_valid["rsna_split"] = df_valid.patient_id.apply(
        lambda p: "rsna_val" if p in val_pats else "rsna_train")

    # Build one row per (patient, laterality) breast pair
    records = []
    for (pat, lat), grp in df_valid.groupby(["patient_id", "laterality"]):
        cc_rows  = grp[grp.view == "CC"]
        mlo_rows = grp[grp.view == "MLO"]
        if cc_rows.empty or mlo_rows.empty:
            continue  # need both views
        cc_row  = cc_rows.iloc[0]
        mlo_row = mlo_rows.iloc[0]
        cancer  = max(cc_row.cancer, mlo_row.cancer)
        records.append({
            "patient_id":  pat,
            "laterality":  lat,
            "cc_image_id": cc_row.image_id,
            "mlo_image_id": mlo_row.image_id,
            "cc_png":      cc_row.png_path,
            "mlo_png":     mlo_row.png_path,
            "cancer":      cancer,
            "age":         cc_row.get("age", None),
            "rsna_split":  cc_row.rsna_split,
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"Manifest saved: {MANIFEST_OUT}")
    print(f"  Total pairs: {len(manifest)}")
    print(f"  rsna_train: {(manifest.rsna_split=='rsna_train').sum()}  pairs  "
          f"(cancer={(manifest[manifest.rsna_split=='rsna_train'].cancer==1).sum()})")
    print(f"  rsna_val:   {(manifest.rsna_split=='rsna_val').sum()}  pairs  "
          f"(cancer={(manifest[manifest.rsna_split=='rsna_val'].cancer==1).sum()})")


if __name__ == "__main__":
    main()
