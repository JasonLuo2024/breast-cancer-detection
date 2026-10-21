# VOI-LUT Extension Scripts

These are the preserved CAIR execution sources used to generate the aggregate
artifacts in [`results/voi_lut_extension/`](../../results/voi_lut_extension/).
They are intentionally separate from the historical six-stage scripts because
the extension uses a different DICOM conversion protocol and five-fold
cross-fitted evaluation.

| Script | Purpose |
| --- | --- |
| `evaluate_voi_checkpoint_snapshot.py` | Freezes 30 checkpoints, evaluates the full official VinDr test partition, aggregates five folds within each of six components, then computes a patient-bootstrap AUROC CI. |
| `analyze_voi_crossview_oof.py` | Selects epochs cross-fitted across five patient folds, pools OOF predictions, computes bootstrap CIs, calibration, routing, subgroup, and stacker analyses. |
| `generate_voi_reliability_figure.py` | Generates the cross-fitted reliability curves and bin-level calibration data. |

## Execution environment

The sources are preserved as executed on CAIR and therefore require the CAIR
research directory layout, the VinDr manifests and preprocessed VOI-LUT PNGs,
checkpoint snapshots, and the original `tools`/ensemble helpers. They are not
intended to execute from a fresh clone without first configuring those data and
model paths. Their command-line interfaces document the necessary inputs.

No patient data, images, checkpoints, or row-level prediction files are
included in this public repository.

SHA-256:

- `evaluate_voi_checkpoint_snapshot.py`: `c606a76ae19f7ad7321b839e9e50283aab6340a448852822001052f1c6a38633`
- `analyze_voi_crossview_oof.py`: `c97fcfae3cdd180e82fc3f6b383af1ff2fc9de5c00db9d7a75dff690b7becff0`
- `generate_voi_reliability_figure.py`: `d5f3307d16d18736f7b3bf16514cc895d0d0c22f7170538628d8e6135ef07d62`
