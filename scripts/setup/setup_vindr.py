"""
VinDr-Mammo dataset setup script.

Run AFTER downloading with:
  wget -r -N -c -np --user <username> --ask-password \
    -P data/vindr-mammo \
    https://physionet.org/files/vindr-mammo/1.0.0/

Usage:
  python3 scripts/setup_vindr.py
"""
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
VINDR_DIR = BASE_DIR / "data" / "vindr-mammo" / "physionet.org" / "files" / "vindr-mammo" / "1.0.0"
ALT_DIR   = BASE_DIR / "data" / "vindr-mammo"


def find_vindr_root() -> Path:
    """Locate the dataset root regardless of wget nesting."""
    # Direct layout (manually extracted)
    for candidate in [VINDR_DIR, ALT_DIR]:
        if (candidate / "breast-level_annotations.csv").exists():
            return candidate
    # Search up to 4 levels deep for the annotations file
    for p in ALT_DIR.rglob("breast-level_annotations.csv"):
        return p.parent
    return None


def check_structure(root: Path) -> dict:
    expected = {
        "breast-level_annotations.csv": "Breast-level BI-RADS + density labels",
        "finding_annotations.csv": "Bounding box annotations for findings",
        "metadata.csv": "DICOM metadata (patient ID, view, laterality)",
    }
    results = {}
    for fname, desc in expected.items():
        path = root / fname
        results[fname] = {"exists": path.exists(), "description": desc}
    return results


def count_images(root: Path) -> dict:
    dicom_dirs = list(root.glob("images/*/*"))
    total_dicoms = len(list(root.rglob("*.dicom"))) + len(list(root.rglob("*.dcm")))
    return {"study_dirs": len(dicom_dirs), "dicom_files": total_dicoms}


def summarise_annotations(root: Path) -> dict:
    summary = {}
    ann_file = root / "breast-level_annotations.csv"
    if not ann_file.exists():
        return summary

    birads_counts = {}
    density_counts = {}
    total = 0
    with open(ann_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            b = row.get("breast_birads", "unknown")
            d = row.get("breast_density", "unknown")
            birads_counts[b] = birads_counts.get(b, 0) + 1
            density_counts[d] = density_counts.get(d, 0) + 1

    cancer_cases = sum(v for k, v in birads_counts.items() if k in ("BI-RADS 4", "BI-RADS 5"))
    summary = {
        "total_breasts": total,
        "birads_distribution": birads_counts,
        "density_distribution": density_counts,
        "cancer_cases_birads_4_5": cancer_cases,
        "cancer_rate_pct": round(100 * cancer_cases / total, 2) if total else 0,
    }
    return summary


def main():
    print("VinDr-Mammo Dataset Setup Check")
    print("=" * 50)

    root = find_vindr_root()
    if root is None:
        print("\n[ERROR] VinDr-Mammo not found.")
        print("\nDownload it first:\n")
        print("  cd /Users/jasonl/breast_cancer_research/data")
        print("  mkdir -p vindr-mammo")
        print("  wget -r -N -c -np \\")
        print("    --user YOUR_PHYSIONET_USERNAME \\")
        print("    --ask-password \\")
        print("    -P vindr-mammo \\")
        print("    https://physionet.org/files/vindr-mammo/1.0.0/")
        sys.exit(1)

    print(f"\nDataset root: {root}\n")

    # Check required files
    print("Required files:")
    structure = check_structure(root)
    all_ok = True
    for fname, info in structure.items():
        status = "✓" if info["exists"] else "✗ MISSING"
        print(f"  {status}  {fname}  — {info['description']}")
        if not info["exists"]:
            all_ok = False

    # Count images
    print("\nImage files:")
    counts = count_images(root)
    print(f"  Study directories : {counts['study_dirs']}")
    print(f"  DICOM files found : {counts['dicom_files']}")
    if counts["dicom_files"] == 0:
        print("  [WARNING] No DICOM files found — download may be incomplete.")

    # Annotation summary
    print("\nAnnotation summary:")
    ann = summarise_annotations(root)
    if ann:
        print(f"  Total breast entries : {ann['total_breasts']}")
        print(f"  Cancer cases (4+5)   : {ann['cancer_cases_birads_4_5']} ({ann['cancer_rate_pct']}%)")
        print(f"  BI-RADS distribution : {ann['birads_distribution']}")
        print(f"  Density distribution : {ann['density_distribution']}")
    else:
        print("  Could not read annotations.")

    print("\n" + ("=" * 50))
    if all_ok and counts["dicom_files"] > 0:
        print("Dataset looks complete. Ready to use.")
        print(f"\nNext step: run the CNN experiment agent with dataset path:\n  {root}")
    else:
        print("Dataset incomplete — check the download and re-run this script.")


if __name__ == "__main__":
    main()
