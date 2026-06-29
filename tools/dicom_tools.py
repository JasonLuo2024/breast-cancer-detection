"""DICOM feature extraction and wavelet-based ML experiments on VinDr-Mammo."""
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import pywt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from config import RESULTS_DIR, VINDR_DIR

warnings.filterwarnings("ignore")

WAVELET = "haar"
N_LEVELS = 3
CANCER_BIRADS = {"BI-RADS 4", "BI-RADS 5"}

CLASSIFIERS = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000, "class_weight": "balanced"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "class_weight": "balanced"}),
    "svm": (SVC, {"kernel": "rbf", "probability": True, "class_weight": "balanced"}),
    "decision_tree": (DecisionTreeClassifier, {"class_weight": "balanced"}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 100}),
}


# ── annotation helpers ────────────────────────────────────────────────────────

def _load_annotations(data_dir: Path) -> dict:
    """Return {image_id: label} where label=1 means cancer (BI-RADS 4/5)."""
    ann_file = data_dir / "breast-level_annotations.csv"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_file}")
    labels = {}
    with open(ann_file, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["image_id"]] = 1 if row["breast_birads"] in CANCER_BIRADS else 0
    return labels


def get_vindr_dataset_info() -> dict:
    """Return summary statistics about the VinDr-Mammo dataset."""
    try:
        ann_file = VINDR_DIR / "breast-level_annotations.csv"
        if not ann_file.exists():
            return {"error": f"Dataset not found at {VINDR_DIR}"}

        total = cancer = train = test = 0
        birads_counts: dict = {}
        with open(ann_file, newline="") as f:
            for row in csv.DictReader(f):
                total += 1
                b = row.get("breast_birads", "unknown")
                birads_counts[b] = birads_counts.get(b, 0) + 1
                if b in CANCER_BIRADS:
                    cancer += 1
                if row.get("split") == "training":
                    train += 1
                else:
                    test += 1

        image_dirs = list((VINDR_DIR / "images").glob("*"))
        return {
            "dataset": "VinDr-Mammo",
            "path": str(VINDR_DIR),
            "total_images": total,
            "training": train,
            "test": test,
            "cancer_cases_birads_4_5": cancer,
            "cancer_rate_pct": round(100 * cancer / total, 2) if total else 0,
            "birads_distribution": birads_counts,
            "study_dirs": len(image_dirs),
            "dicom_available": len(image_dirs) > 0,
        }
    except Exception as e:
        return {"error": str(e)}


# ── feature extraction ────────────────────────────────────────────────────────

def _wavelet_features(image: np.ndarray) -> list[float]:
    """Extract per-level wavelet statistics: min, max, mean, std, skewness, kurtosis."""
    feats: list[float] = []
    coeffs = pywt.wavedec2(image.astype(np.float32), WAVELET, level=N_LEVELS)
    for level_coeffs in coeffs:
        if isinstance(level_coeffs, tuple):
            arr = np.concatenate([c.flatten() for c in level_coeffs])
        else:
            arr = level_coeffs.flatten()
        feats += [
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.mean(arr)),
            float(np.std(arr)),
            float(skew(arr)),
            float(kurtosis(arr)),
        ]
    return feats


def _global_stats(image: np.ndarray) -> list[float]:
    """Global pixel statistics."""
    flat = image.flatten().astype(np.float32)
    return [
        float(np.min(flat)), float(np.max(flat)),
        float(np.mean(flat)), float(np.std(flat)),
        float(skew(flat)), float(kurtosis(flat)),
        float(np.percentile(flat, 25)), float(np.percentile(flat, 75)),
    ]


def extract_features_from_dicom(dicom_path: str) -> list[float]:
    """Read one DICOM file and return a combined feature vector."""
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    # Normalise to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return _global_stats(img) + _wavelet_features(img)


# ── dataset builder ───────────────────────────────────────────────────────────

def build_vindr_wavelet_dataset(
    n_samples_per_class: int = 300,
    random_seed: int = 42,
) -> dict:
    """
    Load DICOM images, extract wavelet features, return X/y arrays.
    Balances by sampling n_samples_per_class from each class.
    """
    try:
        labels_map = _load_annotations(VINDR_DIR)
        images_root = VINDR_DIR / "images"
        if not images_root.exists():
            return {"error": f"Images directory not found: {images_root}"}

        rng = np.random.default_rng(random_seed)

        pos_paths, neg_paths = [], []
        for study_dir in images_root.iterdir():
            if not study_dir.is_dir():
                continue
            for dicom_file in study_dir.glob("*"):
                if dicom_file.suffix.lower() not in ("", ".dcm", ".dicom"):
                    if dicom_file.suffix:
                        continue
                img_id = dicom_file.stem
                label = labels_map.get(img_id)
                if label is None:
                    continue
                if label == 1:
                    pos_paths.append((dicom_file, 1))
                else:
                    neg_paths.append((dicom_file, 0))

        n_pos = min(n_samples_per_class, len(pos_paths))
        n_neg = min(n_samples_per_class, len(neg_paths))

        selected_pos = [pos_paths[i] for i in rng.choice(len(pos_paths), n_pos, replace=False)]
        selected_neg = [neg_paths[i] for i in rng.choice(len(neg_paths), n_neg, replace=False)]
        selected = selected_pos + selected_neg
        rng.shuffle(selected)

        X, y, failed = [], [], 0
        for path, label in selected:
            try:
                feats = extract_features_from_dicom(str(path))
                X.append(feats)
                y.append(label)
            except Exception:
                failed += 1

        return {
            "X": np.array(X, dtype=np.float32),
            "y": np.array(y, dtype=np.int32),
            "n_positive": sum(1 for lbl in y if lbl == 1),
            "n_negative": sum(1 for lbl in y if lbl == 0),
            "n_failed": failed,
            "n_features": len(X[0]) if X else 0,
            "feature_description": (
                "Global stats (8) + wavelet Haar L3 stats (4 levels × 6 stats = 24) = 32 features"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ── ML experiments ────────────────────────────────────────────────────────────

def run_vindr_wavelet_experiment(
    algorithm: str = "random_forest",
    n_samples_per_class: int = 300,
    cv_folds: int = 5,
) -> dict:
    """
    Run a wavelet-feature ML experiment on VinDr DICOM data.
    Uses stratified k-fold CV to handle the small balanced sample.
    """
    if algorithm not in CLASSIFIERS:
        return {"error": f"Unknown algorithm. Choose from: {list(CLASSIFIERS.keys())}"}

    dataset = build_vindr_wavelet_dataset(n_samples_per_class=n_samples_per_class)
    if "error" in dataset:
        return dataset

    X: np.ndarray = dataset["X"]
    y: np.ndarray = dataset["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf_class, params = CLASSIFIERS[algorithm]
    try:
        clf_class(random_state=42)
        params = {**params, "random_state": 42}
    except TypeError:
        pass

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, val_idx in skf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = clf_class(**params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)

        try:
            y_proba = clf.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, y_proba))
        except Exception:
            auc = float(roc_auc_score(y_val, y_pred))

        fold_metrics.append({
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "auc_roc": auc,
        })

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    result = {
        "method": "wavelet_ml",
        "algorithm": algorithm,
        "dataset": "VinDr-Mammo",
        "n_samples": len(y),
        "n_positive": int(dataset["n_positive"]),
        "n_negative": int(dataset["n_negative"]),
        "cv_folds": cv_folds,
        "features": dataset["feature_description"],
        "mean_metrics": avg,
        "std_metrics": std,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"vindr_wavelet_{algorithm}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out_path)
    return result


def run_vindr_wavelet_benchmark(n_samples_per_class: int = 300) -> dict:
    """Benchmark all wavelet-feature classifiers on VinDr data."""
    results = {}
    for algo in CLASSIFIERS:
        print(f"  Running {algo}...")
        res = run_vindr_wavelet_experiment(algo, n_samples_per_class=n_samples_per_class)
        if "error" not in res:
            results[algo] = res["mean_metrics"]

    if not results:
        return {"error": "All algorithms failed"}

    ranking = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
    return {
        "method": "wavelet_benchmark",
        "dataset": "VinDr-Mammo",
        "n_samples_per_class": n_samples_per_class,
        "ranking": [{"algorithm": k, **v} for k, v in ranking],
        "best": ranking[0][0],
        "timestamp": datetime.now().isoformat(),
    }
