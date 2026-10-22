"""Strict patient-cluster paired AUROC comparisons from saved predictions.

Each input must contain exactly one breast-level score for the same cohort.
This program fails on missing, extra, duplicate, non-finite, or
label-mismatched cases. It does not train models, select checkpoints, tune
thresholds, or calculate p-values.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEYS = ["study_id", "laterality"]
REQUIRED = {*KEYS, "label"}


def parse_spec(spec: str) -> tuple[str, Path, str]:
    try:
        name, remainder = spec.split("=", 1)
        path, score = remainder.rsplit(":", 1)
    except ValueError as error:
        raise ValueError(f"Invalid --pred specification: {spec!r}") from error
    if not name or not path or not score:
        raise ValueError(f"Invalid --pred specification: {spec!r}")
    return name, Path(path), score


def read_prediction(spec: str, expected_breasts: int, expected_patients: int,
                    expected_positives: int) -> tuple[str, pd.DataFrame]:
    name, path, score_column = parse_spec(spec)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = REQUIRED | {score_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[[*KEYS, "label", score_column]].rename(columns={score_column: name}).copy()
    frame["study_id"] = frame["study_id"].astype(str)
    frame["laterality"] = frame["laterality"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate breast keys")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: labels must be binary 0/1")
    if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite scores")
    if len(frame) != expected_breasts:
        raise ValueError(f"{path}: expected {expected_breasts} breasts, found {len(frame)}")
    if frame["study_id"].nunique() != expected_patients:
        raise ValueError(f"{path}: expected {expected_patients} patients, found {frame['study_id'].nunique()}")
    if int(frame["label"].sum()) != expected_positives:
        raise ValueError(f"{path}: expected {expected_positives} positives, found {int(frame['label'].sum())}")
    return name, frame.sort_values(KEYS).reset_index(drop=True)


def load_aligned(specifications: list[str], expected_breasts: int, expected_patients: int,
                 expected_positives: int) -> tuple[pd.DataFrame, list[str]]:
    names: list[str] = []
    reference: pd.DataFrame | None = None
    reference_keys: set[tuple[str, str]] | None = None
    for spec in specifications:
        name, frame = read_prediction(spec, expected_breasts, expected_patients, expected_positives)
        if name in names:
            raise ValueError(f"Duplicate model name in --pred: {name}")
        keys = set(map(tuple, frame[KEYS].to_numpy()))
        if reference is None:
            reference, reference_keys = frame[[*KEYS, "label"]].copy(), keys
        else:
            assert reference is not None and reference_keys is not None
            missing, extra = reference_keys - keys, keys - reference_keys
            if missing or extra:
                raise ValueError(f"{name}: case-set mismatch; missing={len(missing)}, extra={len(extra)}. Refusing to intersect case sets.")
            labels = frame.set_index(KEYS)["label"]
            expected = reference.set_index(KEYS)["label"]
            if not labels.equals(expected):
                raise ValueError(f"{name}: labels differ from the reference prediction file")
        score = frame.set_index(KEYS)[name]
        reference[name] = pd.MultiIndex.from_frame(reference[KEYS]).map(score)
        if reference[name].isna().any():
            raise RuntimeError(f"{name}: alignment unexpectedly created missing scores")
        names.append(name)
    assert reference is not None
    return reference, names


def paired_bootstrap(frame: pd.DataFrame, model_a: str, model_b: str, n_boot: int, seed: int) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    scores_a, scores_b = frame[model_a].to_numpy(float), frame[model_b].to_numpy(float)
    patient_indices = [group.to_numpy() for _, group in frame.groupby("study_id", sort=True).groups.items()]
    auc_a, auc_b = float(roc_auc_score(y, scores_a)), float(roc_auc_score(y, scores_b))
    rng, differences, replacements = np.random.default_rng(seed), [], 0
    while len(differences) < n_boot:
        selected = rng.integers(0, len(patient_indices), size=len(patient_indices))
        index = np.concatenate([patient_indices[i] for i in selected])
        if np.unique(y[index]).size != 2:
            replacements += 1
            continue
        differences.append(float(roc_auc_score(y[index], scores_a[index]) - roc_auc_score(y[index], scores_b[index])))
    values = np.asarray(differences)
    lower, upper = (float(x) for x in np.percentile(values, [2.5, 97.5]))
    return {
        "model_A": model_a, "model_B": model_b, "AUROC_A": auc_a, "AUROC_B": auc_b,
        "delta_AUROC": auc_a - auc_b, "delta_CI95_lower": lower, "delta_CI95_upper": upper,
        "number_of_patients": int(frame.study_id.nunique()), "number_of_breasts": int(len(frame)),
        "number_of_positives": int(y.sum()), "valid_resamples": int(len(values)),
        "single_class_replacements": replacements, "seed": seed,
        "ci_includes_zero": bool(lower <= 0 <= upper), "bootstrap_differences": values.tolist(),
    }


def manuscript_table(results: list[dict]) -> str:
    rows = ["| Comparison (A - B) | AUROC A | AUROC B | Delta AUROC | 95% paired bootstrap CI | Includes 0? |",
            "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in results:
        rows.append(f"| {row['model_A']} - {row['model_B']} | {row['AUROC_A']:.4f} | {row['AUROC_B']:.4f} | {row['delta_AUROC']:+.4f} | [{row['delta_CI95_lower']:+.4f}, {row['delta_CI95_upper']:+.4f}] | {'yes' if row['ci_includes_zero'] else 'no'} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True, help="name=path.csv:score_column")
    parser.add_argument("--pairs", nargs="+", required=True, help="A:B, delta = AUROC(A) - AUROC(B)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-breasts", type=int, default=2000)
    parser.add_argument("--expected-patients", type=int, default=1000)
    parser.add_argument("--expected-positives", type=int, default=99)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    frame, names = load_aligned(args.pred, args.expected_breasts, args.expected_patients, args.expected_positives)
    results = []
    for pair in args.pairs:
        model_a, model_b = pair.split(":", 1)
        if model_a not in names or model_b not in names:
            raise ValueError(f"Unknown model in {pair!r}; available: {names}")
        results.append(paired_bootstrap(frame, model_a, model_b, args.n_boot, args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "patient-cluster paired percentile bootstrap",
        "cohort_requirements": {"breasts": args.expected_breasts, "patients": args.expected_patients, "positives": args.expected_positives},
        "n_boot_requested": args.n_boot, "seed": args.seed,
        "p_values": "Not calculated; paired AUROC-difference percentile CIs are the primary output.",
        "comparisons": results,
    }, indent=2) + "\n")
    args.table.write_text(manuscript_table(results))
    print(manuscript_table(results), end="")
    print(f"Saved full-precision results to {args.out}")


if __name__ == "__main__":
    main()
