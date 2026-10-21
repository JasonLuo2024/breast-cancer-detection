# VOI-LUT Extension Results

This directory contains aggregate, reproducible results for the DICOM
Modality/VOI-LUT extension added after the original six-stage paper.

## Endpoint and unit

- Dataset: VinDr-Mammo.
- Endpoint: BI-RADS 4/5 versus BI-RADS 1/2/3.
- Unit: breast.

## Files

| File | Evaluation | Main result |
| --- | --- | --- |
| `crossview_oof_analysis.json` | Patient-stratified five-fold cross-fitted OOF evaluation on the official training partition | Focal CrossView AUROC 0.8212, 95% patient-bootstrap CI [0.7933, 0.8487]. |
| `final_test_20261015.json` | Frozen, equal-weight six-component snapshot on the official test partition | AUROC 0.8500, 95% patient-bootstrap CI [0.8016, 0.8949]. |

The OOF evaluation selects each held-out fold's epoch using only the other four
folds, with a three-epoch mean-AUROC window. The official-test snapshot uses
five fold checkpoints per component and equal averaging across the six
components; test results did not alter checkpoint, preprocessing, model, or
weight selection.

The 0.8500 value is a valid measurement for this frozen recipe. Its nominal
0.0042 difference from the historical 0.8458 ensemble is not an established
improvement because a paired, predeclared comparison of the two ensembles was
not available.

Row-level prediction files are deliberately not published here: although study
identifiers were hashed, they also contain age, breast density, laterality, and
BI-RADS metadata. The JSON files provide aggregate metrics, confidence
intervals, component configuration, checkpoint hashes, and run provenance.

SHA-256:

- `final_test_20261015.json`: `5f0c341418ad4c66875cdcfe120db79b76a5882ac73b1c142e2e22b17a854abd`
- `crossview_oof_analysis.json`: `6910600f8976ea5991e69df0b076353f6c2f4d2bd318ddcf0fd954ddf9d317c9`
