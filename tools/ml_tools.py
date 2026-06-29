"""ML experiment tools for breast cancer detection."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RESULTS_DIR

CLASSIFIERS = {
    "random_forest": (RandomForestClassifier, {"n_estimators": 100}),
    "svm": (SVC, {"kernel": "rbf", "probability": True}),
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "neural_network": (MLPClassifier, {"hidden_layer_sizes": (100, 50), "max_iter": 500}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 100}),
    "knn": (KNeighborsClassifier, {"n_neighbors": 5}),
}


def get_dataset_info() -> dict:
    """Return info about the Wisconsin Breast Cancer dataset."""
    data = load_breast_cancer()
    malignant = int((data.target == 0).sum())
    benign = int((data.target == 1).sum())
    return {
        "name": "Wisconsin Breast Cancer Dataset",
        "n_samples": int(len(data.data)),
        "n_features": int(len(data.feature_names)),
        "classes": data.target_names.tolist(),
        "class_distribution": {"malignant": malignant, "benign": benign},
        "features": data.feature_names.tolist(),
        "description": data.DESCR[:500],
    }


def run_experiment(
    algorithm: str,
    hyperparams: dict | None = None,
    test_size: float = 0.2,
    cross_val_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run a breast cancer classification experiment.

    Args:
        algorithm: One of random_forest, svm, logistic_regression,
                   neural_network, gradient_boosting, knn
        hyperparams: Override default hyperparameters
        test_size: Fraction of data for testing
        cross_val_folds: Number of cross-validation folds (0 to skip)
        random_state: Random seed for reproducibility
    """
    if algorithm not in CLASSIFIERS:
        return {
            "error": f"Unknown algorithm '{algorithm}'. Available: {list(CLASSIFIERS.keys())}"
        }

    clf_class, default_params = CLASSIFIERS[algorithm]
    params = {**default_params, **(hyperparams or {})}

    # Inject random_state where supported
    try:
        clf_class(random_state=0)
        params["random_state"] = random_state
    except TypeError:
        pass

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = clf_class(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))
    except AttributeError:
        auc = float(roc_auc_score(y_test, y_pred))

    result = {
        "algorithm": algorithm,
        "hyperparams": params,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc_roc": auc,
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=data.target_names
        ),
        "dataset": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X.shape[1],
            "test_size": test_size,
        },
    }

    if cross_val_folds > 1:
        cv_clf = clf_class(**params)
        cv_scores = cross_val_score(cv_clf, X_scaled, y, cv=cross_val_folds, scoring="f1")
        result["cross_validation"] = {
            "folds": cross_val_folds,
            "scores": cv_scores.tolist(),
            "mean_f1": float(cv_scores.mean()),
            "std_f1": float(cv_scores.std()),
        }

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        pairs = sorted(
            zip(data.feature_names, importances), key=lambda x: x[1], reverse=True
        )[:10]
        result["top_features"] = [
            {"feature": str(f), "importance": float(i)} for f, i in pairs
        ]

    return result


def run_benchmark(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Benchmark all available algorithms and rank them by F1 score."""
    rankings = []
    all_results = {}

    for algo in CLASSIFIERS:
        res = run_experiment(
            algo,
            test_size=test_size,
            cross_val_folds=0,
            random_state=random_state,
        )
        if "error" not in res:
            all_results[algo] = res["metrics"]
            rankings.append(
                {
                    "algorithm": algo,
                    "f1_score": res["metrics"]["f1_score"],
                    "accuracy": res["metrics"]["accuracy"],
                    "auc_roc": res["metrics"]["auc_roc"],
                }
            )

    rankings.sort(key=lambda x: x["f1_score"], reverse=True)
    return {
        "results": all_results,
        "ranking": rankings,
        "best_algorithm": rankings[0]["algorithm"] if rankings else None,
        "timestamp": datetime.now().isoformat(),
    }


def save_results(results: dict, experiment_name: str) -> str:
    """Save experiment results to a JSON file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = experiment_name.replace(" ", "_")
    path = RESULTS_DIR / f"{safe_name}_{ts}.json"
    path.write_text(json.dumps(results, indent=2))
    return str(path)


def load_results(filename: str) -> dict:
    """Load previously saved experiment results."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    return json.loads(path.read_text())


def list_saved_experiments() -> list[dict]:
    """List all saved experiment result files."""
    experiments = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
            experiments.append(
                {
                    "filename": p.name,
                    "algorithm": data.get("algorithm", "benchmark"),
                    "timestamp": data.get("timestamp", ""),
                    "f1_score": data.get("metrics", {}).get("f1_score"),
                    "accuracy": data.get("metrics", {}).get("accuracy"),
                }
            )
        except Exception:
            pass
    return experiments


def compare_experiments(filenames: list[str]) -> dict:
    """Compare multiple saved experiments side by side."""
    rows = []
    for fname in filenames:
        res = load_results(fname)
        if "error" not in res:
            rows.append(
                {
                    "filename": fname,
                    "algorithm": res.get("algorithm"),
                    "accuracy": res.get("metrics", {}).get("accuracy"),
                    "f1_score": res.get("metrics", {}).get("f1_score"),
                    "auc_roc": res.get("metrics", {}).get("auc_roc"),
                    "recall": res.get("metrics", {}).get("recall"),
                    "precision": res.get("metrics", {}).get("precision"),
                }
            )

    if not rows:
        return {"error": "No valid experiments found"}

    rows.sort(key=lambda x: (x.get("f1_score") or 0), reverse=True)
    return {"count": len(rows), "comparison": rows, "best": rows[0]}
