#!/usr/bin/env python3
"""Descriptive full-test evaluation of an immutable 30-checkpoint VOI snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.run_ensemble import _infer_bp_dict, _load_checkpoint  # noqa: E402
from tools.cnn_tools import ARCHITECTURES, SingleViewDataset, _val_transform  # noqa: E402


EXPECTED = {"stage1": 5, "stage2_default": 5, "stage2_focal": 5,
            "stage2_sampler": 5, "crossview_focal": 5, "crossview_sampler": 5}


def component(name: str) -> str:
    if name.startswith("cnn_"):
        return "stage1"
    if name.startswith("cv_"):
        return "crossview_sampler" if "sp05" in name else "crossview_focal"
    if name.startswith("mv_"):
        if "samp0.5" in name:
            return "stage2_sampler"
        if "fa0.75" in name:
            return "stage2_focal"
        return "stage2_default"
    raise ValueError(f"Unrecognized checkpoint: {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(source: Path) -> list[Path]:
    paths = list(source.glob("cnn_densenet121_b3voi_s1_f*.pt"))
    paths += list(source.glob("cv_convnext_base_img1152x896_b3voi_cv_*.pt"))
    paths += list(source.glob("mv_efficientnet_b3_attention_base_elastic*fold*_img1152x896_202610*.pt"))
    # Exclude the earlier high-resolution OOF study, whose filenames contain
    # an explicit `_seed...` token but otherwise overlap this experiment.
    return sorted({p for p in paths if "_seed" not in p.name})


def freeze_checkpoints(source: Path, snapshot: Path) -> list[Path]:
    snapshot.mkdir(parents=True, exist_ok=True)
    paths = discover(source)
    counts = pd.Series([component(p.name) for p in paths]).value_counts().to_dict()
    if counts != EXPECTED:
        raise RuntimeError(f"Expected component counts {EXPECTED}, found {counts}")
    frozen = []
    for path in paths:
        target = snapshot / path.name
        if not target.exists():
            shutil.copy2(path, target)
        frozen.append(target)
    manifest = [{"checkpoint": p.name, "component": component(p.name), "sha256": sha256(p)}
                for p in frozen]
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2))
    return frozen


@torch.no_grad()
def infer_stage1(path: Path, manifest: pd.DataFrame, batch: int, workers: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    architecture = ckpt.get("architecture", "densenet121")
    model = ARCHITECTURES[architecture](pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda().eval()
    dataset = SingleViewDataset(manifest, "test", _val_transform(ckpt.get("img_size", (1152, 896))))
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=True)
    probabilities = []
    for images, _ in loader:
        probabilities.extend(torch.sigmoid(model(images.cuda(non_blocking=True)).squeeze(1))
                             .cpu().numpy().tolist())
    rows = pd.DataFrame(dataset.get_meta(), columns=["study_id", "laterality", "image_id"])
    rows["label"] = dataset.get_labels()
    rows["probability"] = probabilities
    breast = rows.groupby(["study_id", "laterality"], as_index=False).agg(
        label=("label", "max"), probability=("probability", "mean"))
    del model
    torch.cuda.empty_cache()
    return {(r.study_id, r.laterality): r.probability for r in breast.itertuples()}, \
        {(r.study_id, r.laterality): int(r.label) for r in breast.itertuples()}


def patient_bootstrap(labels, scores, patient_ids, n_boot=2000, seed=20261014):
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    indices = {p: np.flatnonzero(patient_ids == p) for p in patients}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, len(patients), replace=True)
        idx = np.concatenate([indices[p] for p in sampled])
        if np.unique(labels[idx]).size == 2:
            values.append(roc_auc_score(labels[idx], scores[idx]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def score_summary(labels, scores, patient_ids):
    return {"auc": float(roc_auc_score(labels, scores)),
            "auc_ci": patient_bootstrap(labels, scores, patient_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoints = freeze_checkpoints(args.model_dir, args.snapshot_dir)
    manifest = pd.read_csv(args.manifest)
    predictions, labels_by_key, members = {}, None, []
    for index, path in enumerate(checkpoints, 1):
        group = component(path.name)
        print(f"[{index:02d}/{len(checkpoints)}] {group}: {path.name}", flush=True)
        if group == "stage1":
            probs, labels = infer_stage1(path, manifest, args.batch, args.workers)
        else:
            model, model_type, _, _, img_size, _, _ = _load_checkpoint(path)
            probs, labels = _infer_bp_dict(model, manifest, img_size, args.batch,
                                           args.workers, 1)
            del model
            torch.cuda.empty_cache()
        if labels_by_key is None:
            labels_by_key = labels
        if labels != labels_by_key:
            raise RuntimeError(f"Label/key mismatch for {path.name}")
        predictions[path.name] = probs
        members.append({"checkpoint": path.name, "component": group,
                        "sha256": sha256(path)})

    keys = sorted(set.intersection(*(set(v) for v in predictions.values())))
    y = np.array([labels_by_key[k] for k in keys], dtype=int)
    patients = np.array([k[0] for k in keys])
    matrix = np.array([[predictions[name][k] for k in keys] for name in predictions])
    names = list(predictions)

    individual = {}
    for i, name in enumerate(names):
        individual[name] = score_summary(y, matrix[i], patients)
        members[i].update(individual[name])

    component_scores = {}
    component_arrays = {}
    for group in EXPECTED:
        idx = [i for i, name in enumerate(names) if component(name) == group]
        component_arrays[group] = matrix[idx].mean(axis=0)
        component_scores[group] = score_summary(y, component_arrays[group], patients)
    all_components = np.array(list(component_arrays.values()))
    equal_six = all_components.mean(axis=0)
    ensemble = score_summary(y, equal_six, patients)

    prediction_frame = pd.DataFrame({"study_id": patients,
                                     "laterality": [k[1] for k in keys], "label": y})
    for group, values in component_arrays.items():
        prediction_frame[group] = values
    prediction_frame["equal_six_component_ensemble"] = equal_six
    prediction_path = args.output.with_suffix(".predictions.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)
    result = {
        "status": "descriptive_test_informed",
        "manifest": str(args.manifest),
        "snapshot_manifest": str(args.snapshot_dir / "snapshot_manifest.json"),
        "n_checkpoints": len(checkpoints),
        "n_breasts": len(y),
        "n_patients": len(np.unique(patients)),
        "n_positive_breasts": int(y.sum()),
        "tta": 1,
        "aggregation": "mean folds within each of six components; equal mean across components",
        "members": members,
        "components": component_scores,
        "equal_six_component_ensemble": ensemble,
        "predictions": str(prediction_path),
        "warning": "Official test was inspected. Do not use these results for checkpoint, epoch, weight, preprocessing, or architecture selection."
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"components": component_scores, "ensemble": ensemble}, indent=2), flush=True)


if __name__ == "__main__":
    main()
