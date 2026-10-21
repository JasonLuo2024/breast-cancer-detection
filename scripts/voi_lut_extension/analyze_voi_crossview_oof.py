"""CPU-only analysis of completed VOI-LUT CrossView OOF predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


COMPONENTS = {
    "crossview_focal075": ("b3voi_cv_fa75_f", 25),
    "crossview_sampler05": ("b3voi_cv_sp05_f", 25),
}


def auc(frame, score):
    return float(roc_auc_score(frame.label, frame[score]))


def load_epoch_predictions(root, prefix, epochs):
    curves, predictions = {}, {}
    for fold in range(5):
        run_key = f"{prefix}{fold}"
        curves[fold], predictions[fold] = {}, {}
        for path in root.glob(f"*{run_key}*_epoch*_oof.csv"):
            frame = pd.read_csv(path)
            if frame.empty or str(frame.run_key.iloc[0]) != run_key:
                continue
            epoch = int(frame.epoch.iloc[0])
            curves[fold][epoch] = auc(frame, "probability")
            predictions[fold][epoch] = frame
        if set(curves[fold]) != set(range(1, epochs + 1)):
            raise RuntimeError(f"Incomplete epochs for {run_key}: {sorted(curves[fold])}")
    return curves, predictions


def crossfit_epochs(curves, predictions, epochs):
    selected = {}
    for heldout in range(5):
        score = {
            epoch: np.mean([
                curves[fold][window]
                for fold in range(5) if fold != heldout
                for window in (epoch - 1, epoch, epoch + 1)
            ]) for epoch in range(2, epochs)
        }
        selected[heldout] = max(score, key=lambda epoch: (score[epoch], -epoch))
    pooled = pd.concat([predictions[f][selected[f]] for f in range(5)], ignore_index=True)
    if len(pooled) != 8000 or pooled.duplicated(["study_id", "laterality"]).any():
        raise RuntimeError("Cross-fitted OOF coverage failure")
    all_score = {
        epoch: np.mean([
            curves[fold][window]
            for fold in range(5)
            for window in (epoch - 1, epoch, epoch + 1)
        ]) for epoch in range(2, epochs)
    }
    return pooled, selected, max(all_score, key=lambda epoch: (all_score[epoch], -epoch))


def patient_bootstrap(frame, score_a, score_b=None, n_boot=2000, seed=42):
    groups = [np.asarray(index) for index in frame.groupby("study_id").indices.values()]
    rng, values = np.random.default_rng(seed), []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[i] for i in chosen])
        sample = frame.iloc[index]
        if sample.label.nunique() < 2:
            continue
        value = auc(sample, score_a)
        if score_b is not None:
            value -= auc(sample, score_b)
        values.append(value)
    estimate = auc(frame, score_a) - (auc(frame, score_b) if score_b else 0)
    return {"estimate": estimate, "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "valid_bootstraps": len(values)}


def calibration_metrics(labels, probabilities):
    clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=5000).fit(logits, labels)
    observed, predicted = calibration_curve(labels, clipped, n_bins=10, strategy="quantile")
    return {
        "brier": float(brier_score_loss(labels, clipped)),
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "ece_10bin": float(np.mean(np.abs(observed - predicted))),
    }


def crossfit_calibration(frame, score, method):
    output = np.zeros(len(frame))
    for fold in range(5):
        train, held = frame.fold != fold, frame.fold == fold
        x_train = frame.loc[train, score].to_numpy()
        x_held = frame.loc[held, score].to_numpy()
        y_train = frame.loc[train, "label"].to_numpy()
        if method == "platt":
            model = LogisticRegression(max_iter=2000).fit(x_train.reshape(-1, 1), y_train)
            output[held] = model.predict_proba(x_held.reshape(-1, 1))[:, 1]
        else:
            model = IsotonicRegression(out_of_bounds="clip").fit(x_train, y_train)
            output[held] = model.predict(x_held)
    return output


def crossfit_stack(frame, columns):
    convex, logistic, weights = np.zeros(len(frame)), np.zeros(len(frame)), {}
    grid = np.linspace(0, 1, 101)
    for fold in range(5):
        train, held = frame.fold != fold, frame.fold == fold
        train_values = frame.loc[train, columns].to_numpy()
        held_values = frame.loc[held, columns].to_numpy()
        labels = frame.loc[train, "label"].to_numpy()
        grid_auc = [roc_auc_score(labels, w * train_values[:, 0] + (1 - w) * train_values[:, 1])
                    for w in grid]
        weight = float(grid[int(np.argmax(grid_auc))])
        weights[str(fold)] = weight
        convex[held] = weight * held_values[:, 0] + (1 - weight) * held_values[:, 1]
        model = LogisticRegression(max_iter=2000).fit(train_values, labels)
        logistic[held] = model.predict_proba(held_values)[:, 1]
    return convex, logistic, weights


def threshold_for_sensitivity(labels, scores, target):
    positive = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    allowed_misses = int(np.floor((1 - target) * len(positive)))
    return float(positive[min(allowed_misses, len(positive) - 1)])


def crossfit_routing(frame, score, targets):
    output = {}
    for target in targets:
        decisions = np.zeros(len(frame), dtype=bool)
        thresholds = {}
        for fold in range(5):
            train, held = frame.fold != fold, frame.fold == fold
            threshold = threshold_for_sensitivity(frame.loc[train, "label"],
                                                  frame.loc[train, score], target)
            thresholds[str(fold)] = threshold
            decisions[held] = frame.loc[held, score] >= threshold
        y = frame.label.to_numpy().astype(bool)
        output[str(target)] = {
            "threshold_by_fold": thresholds,
            "achieved_sensitivity": float(decisions[y].mean()),
            "specificity": float((~decisions[~y]).mean()),
            "referred_fraction": float(decisions.mean()),
            "workload_reduction": float((~decisions).mean()),
            "false_negatives": int((~decisions & y).sum()),
            "positives": int(y.sum()),
        }
    return output


def add_metadata(frame, manifest_path, age_path=None):
    manifest = pd.read_csv(manifest_path)
    if age_path:
        age = pd.read_csv(age_path, usecols=["SOP Instance UID", "Patient's Age"])
        age["age"] = pd.to_numeric(age["Patient's Age"].astype(str).str.extract(r"(\d+)")[0],
                                   errors="coerce")
        age_map = age.drop_duplicates("SOP Instance UID").set_index("SOP Instance UID").age
        manifest["age"] = manifest.image_id.map(age_map)
    else:
        manifest["age"] = np.nan
    meta = manifest[manifest.split == "training"].groupby(["study_id", "laterality"], as_index=False).agg(
        age=("age", "median"), density=("breast_density", "first"),
        birads=("breast_birads", "first"))
    return frame.merge(meta, on=["study_id", "laterality"], validate="one_to_one")


def subgroup_summary(frame, score):
    frame = frame.copy()
    frame["age_group"] = pd.cut(frame.age, [0, 39, 49, 59, 69, np.inf],
                                labels=["<40", "40-49", "50-59", "60-69", "70+"])
    summaries = {}
    for variable in ["age_group", "density", "laterality", "birads"]:
        rows = []
        for value, group in frame.groupby(variable, observed=True, dropna=False):
            rows.append({"group": str(value), "n": len(group), "positive": int(group.label.sum()),
                         "auc": None if group.label.nunique() < 2 else auc(group, score),
                         "mean_score": float(group[score].mean())})
        summaries[variable] = rows
    return summaries


def subgroup_auc_delta_bootstrap(frame, score, variable, first, second, n_boot, seed):
    usable = frame[frame[variable].isin([first, second])].copy()
    if any(g.label.nunique() < 2 for _, g in usable.groupby(variable)):
        return None
    rng = np.random.default_rng(seed)
    patient_groups = [np.asarray(index) for index in usable.groupby("study_id").indices.values()]
    values = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(patient_groups), len(patient_groups))
        sample = usable.iloc[np.concatenate([patient_groups[index] for index in sampled])]
        strata = {key: group for key, group in sample.groupby(variable)}
        if first not in strata or second not in strata or any(g.label.nunique() < 2 for g in strata.values()):
            continue
        values.append(auc(strata[first], score) - auc(strata[second], score))
    original = {key: group for key, group in usable.groupby(variable)}
    return {"contrast": f"{first} minus {second}",
            "estimate": auc(original[first], score) - auc(original[second], score),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(), "valid_bootstraps": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--age-metadata")
    parser.add_argument("--historical-aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    root = Path(args.results_dir)
    component_frames, component_results = {}, {}
    for name, (prefix, epochs) in COMPONENTS.items():
        curves, predictions = load_epoch_predictions(root, prefix, epochs)
        frame, selected, final_epoch = crossfit_epochs(curves, predictions, epochs)
        frame = frame[["study_id", "laterality", "label", "probability", "fold"]].copy()
        frame = frame.rename(columns={"probability": name})
        component_frames[name] = frame
        component_results[name] = {"selected_epoch_by_fold": selected,
                                   "final_training_epoch": final_epoch,
                                   "auc": auc(frame, name),
                                   "patient_bootstrap": patient_bootstrap(frame, name, n_boot=args.n_boot, seed=args.seed)}
    columns = list(COMPONENTS)
    merged = component_frames[columns[0]].merge(
        component_frames[columns[1]], on=["study_id", "laterality", "label", "fold"],
        validate="one_to_one")
    merged["convex_stack"], merged["logistic_stack"], weights = crossfit_stack(merged, columns)
    candidate_scores = columns + ["convex_stack", "logistic_stack"]
    score_results = {}
    for score in candidate_scores:
        score_results[score] = {"auc": auc(merged, score),
                                "patient_bootstrap": patient_bootstrap(merged, score, n_boot=args.n_boot, seed=args.seed),
                                "calibration": calibration_metrics(merged.label, merged[score])}
    best_score = max(candidate_scores, key=lambda score: score_results[score]["auc"])
    for method in ["platt", "isotonic"]:
        column = f"{best_score}_{method}"
        merged[column] = crossfit_calibration(merged, best_score, method)
        score_results[column] = {"auc": auc(merged, column),
                                 "patient_bootstrap": patient_bootstrap(merged, column, n_boot=args.n_boot, seed=args.seed),
                                 "calibration": calibration_metrics(merged.label, merged[column])}
    merged = add_metadata(merged, args.manifest, args.age_metadata)
    historical = json.loads(Path(args.historical_aggregate).read_text())
    matched = {
        "scope": "matched CrossView ResNet-50, 224px, 5 folds x 3 seeds; distinct from the 1152x896 ConvNeXt analysis",
        "voi_lut_auc": historical["arms"]["voi_lut"]["seed_averaged_oof_auc"],
        "baseline_auc": historical["arms"]["baseline"]["seed_averaged_oof_auc"],
        "paired_patient_bootstrap": historical["arms"]["voi_lut"]["paired_patient_bootstrap_vs_baseline"],
        "nonnegative_mean_fold_deltas": historical["arms"]["voi_lut"]["nonnegative_mean_fold_deltas"],
    }
    interactions = {
        "laterality": subgroup_auc_delta_bootstrap(merged, best_score, "laterality", "L", "R", args.n_boot, args.seed),
        "density": subgroup_auc_delta_bootstrap(merged, best_score, "density", "DENSITY C", "DENSITY D", args.n_boot, args.seed),
    }
    if merged.age.notna().any():
        merged["age_binary"] = np.where(merged.age >= 50, ">=50", "<50")
        interactions["age"] = subgroup_auc_delta_bootstrap(
            merged, best_score, "age_binary", ">=50", "<50", args.n_boot, args.seed)
    output = {
        "endpoint": "BI-RADS 4/5 versus 1/2/3", "unit": "breast",
        "selection_partition": "VinDr official training only", "official_test_used": False,
        "crossfit_protocol": "held-out fold epoch/calibrator/threshold/stacker selected or fitted on other four folds",
        "components": component_results, "scores": score_results,
        "paired_score_differences": {
            "convex_stack_minus_focal075": patient_bootstrap(
                merged, "convex_stack", "crossview_focal075", args.n_boot, args.seed),
            "logistic_stack_minus_focal075": patient_bootstrap(
                merged, "logistic_stack", "crossview_focal075", args.n_boot, args.seed),
            "focal075_minus_sampler05": patient_bootstrap(
                merged, "crossview_focal075", "crossview_sampler05", args.n_boot, args.seed),
        },
        "stacker": {"convex_weight_on_focal_by_heldout_fold": weights},
        "selected_development_score": best_score,
        "routing": crossfit_routing(merged, best_score, [0.90, 0.95, 0.975, 0.99]),
        "subgroups": subgroup_summary(merged, best_score),
        "subgroup_auc_interactions": interactions,
        "historical_preprocessing_comparison": matched,
        "limitations": [
            "The selected development score is selected on pooled OOF evidence and requires external locked confirmation.",
            "BI-RADS-category AUC is undefined because BI-RADS 4/5 defines the positive endpoint; category score summaries are descriptive.",
            "The historical preprocessing comparison uses ResNet-50 at 224px, not the new ConvNeXt 1152x896 configuration.",
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    merged.to_csv(destination.with_suffix(".predictions.csv"), index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
