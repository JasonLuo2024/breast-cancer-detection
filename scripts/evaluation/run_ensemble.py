"""
Ensemble inference over saved BreastPairNet + AsymmetryNet checkpoints.

Supports:
  - All mv_*.pt  (ipsilateral BreastPairNet, legacy or new head)
  - All asymmetry_*.pt  (AsymmetryNet bilateral diff, skip medical_pretrained)
  - Test-Time Augmentation (TTA): horizontal flip, average N passes
  - Mean and AUC-weighted ensemble combination
  - Alignment by (study_id, laterality) so mixed model types can be combined

Usage:
    python scripts/run_ensemble.py                         # all models, mean, no TTA
    python scripts/run_ensemble.py --method weighted       # AUC-weighted
    python scripts/run_ensemble.py --tta 5                 # 5-pass TTA
    python scripts/run_ensemble.py --top 3 --method weighted --tta 5
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from config import LARGE_MODELS_DIR, LARGE_RESULTS_DIR
from tools.multiview_tools import (
    BreastPairNet, AsymmetryNet,
    BreastPairDataset, PatientQuadDataset,
    _val_transform, _compute_metrics, find_optimal_threshold,
    _build_backbone,
)
from tools.crossview_tools import CrossViewTransformerNet

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_CSV  = Path(r"G:\breast-cancer-research\split_manifest.csv")
PROGRESS_FILE = BASE_DIR / "data" / "progress.json"


# ── legacy BreastPairNet (pre Phase-A 2-layer head) ──────────────────────────

class _LegacyBreastPairNet(nn.Module):
    def __init__(self, arch, pretrained=False):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained)
        self.fusion_mode = "concat"
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(feat_dim * 2, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1),
        )
    def forward(self, cc, mlo):
        return self.head(torch.cat([self.backbone(cc), self.backbone(mlo)], dim=1)).squeeze(1)


# ── checkpoint loading ────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: Path):
    """Returns (model, model_type, arch, fusion_or_none, img_sz, thresh, is_medical)."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    arch   = ckpt.get("arch",      "densenet121")
    img_sz = ckpt.get("img_size",  224)
    thresh = ckpt.get("best_thresh", 0.5)
    sd     = ckpt["state_dict"]
    mtype  = ckpt.get("model_type", "breastpair")
    is_med = ckpt.get("medical_pretrained", False)

    if mtype == "asymmetry":
        model = AsymmetryNet(arch, pretrained=False, medical_pretrained=False).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()
        return model, "asymmetry", arch, None, img_sz, thresh, is_med

    # CrossView Transformer
    name = ckpt_path.stem
    mtype = ckpt.get("model_type")
    if not mtype:
        mtype = "crossview" if name.startswith("cv_") else "breastpair"

    if mtype == "crossview":
        d_model = ckpt.get("d_model", 512)
        n_heads = ckpt.get("n_heads", 8)
        medical = ckpt.get("medical", False)
        model = CrossViewTransformerNet(arch, pretrained=False,
                                        d_model=d_model, n_heads=n_heads,
                                        medical=medical).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()
        return model, "crossview", arch, "cross-attn", img_sz, thresh, medical

    # BreastPairNet — detect head version
    fusion = ckpt.get("fusion", "concat")
    is_legacy = (fusion == "concat" and
                 "head.4.weight" in sd and
                 sd["head.4.weight"].shape[0] == 1)
    if is_legacy:
        model = _LegacyBreastPairNet(arch).to(DEVICE)
    else:
        model = BreastPairNet(arch, fusion, pretrained=False).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, "breastpair", arch, fusion, img_sz, thresh, False


# ── inference with TTA ────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_bp_dict(model, manifest, img_sz, batch_size, num_workers, tta_passes):
    """BreastPairNet → {(study_id, lat): prob}, labels dict."""
    val_ds = BreastPairDataset(manifest, "test", _val_transform(img_sz))
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_probs, all_meta = [], []
    for cc, mlo, _ in loader:
        cc, mlo = cc.to(DEVICE), mlo.to(DEVICE)
        passes = [torch.sigmoid(model(cc, mlo))]
        for _ in range(tta_passes - 1):
            cc_f  = TF.hflip(cc)
            mlo_f = TF.hflip(mlo)
            passes.append(torch.sigmoid(model(cc_f, mlo_f)))
        prob = torch.stack(passes).mean(0).cpu().numpy().tolist()
        all_probs.extend(prob)
    all_meta = val_ds.get_meta()   # [(study_id, lat, label), ...]
    prob_dict  = {(m[0], m[1]): p for m, p in zip(all_meta, all_probs)}
    label_dict = {(m[0], m[1]): m[2] for m in all_meta}
    return prob_dict, label_dict


@torch.no_grad()
def _infer_asym_dict(model, manifest, img_sz, batch_size, num_workers, tta_passes):
    """AsymmetryNet → {(study_id, lat): prob}, labels dict."""
    val_ds = PatientQuadDataset(manifest, "test", _val_transform(img_sz))
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    l_probs, r_probs = [], []
    for l_cc, l_mlo, r_cc, r_mlo, _, _ in loader:
        l_cc, l_mlo = l_cc.to(DEVICE), l_mlo.to(DEVICE)
        r_cc, r_mlo = r_cc.to(DEVICE), r_mlo.to(DEVICE)
        lL, lR = model(l_cc, l_mlo, r_cc, r_mlo)
        pL = [torch.sigmoid(lL)]
        pR = [torch.sigmoid(lR)]
        for _ in range(tta_passes - 1):
            lL2, lR2 = model(TF.hflip(l_cc), TF.hflip(l_mlo),
                             TF.hflip(r_cc), TF.hflip(r_mlo))
            pL.append(torch.sigmoid(lL2))
            pR.append(torch.sigmoid(lR2))
        l_probs.extend(torch.stack(pL).mean(0).cpu().numpy().tolist())
        r_probs.extend(torch.stack(pR).mean(0).cpu().numpy().tolist())

    meta = val_ds.get_breast_meta()   # [(study_id,'L',l_lbl),(study_id,'R',r_lbl),...]
    probs_flat = []
    for i, (sid, lat, lbl) in enumerate(meta):
        if lat == "L":
            probs_flat.append(l_probs[i // 2])
        else:
            probs_flat.append(r_probs[i // 2])
    prob_dict  = {(m[0], m[1]): p for m, p in zip(meta, probs_flat)}
    label_dict = {(m[0], m[1]): m[2] for m in meta}
    return prob_dict, label_dict


# ── progress tracking ─────────────────────────────────────────────────────────

def _mark_done(key, result):
    try:
        data = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
        data.setdefault("completed_experiments", [])
        data.setdefault("best_results", {})
        if key not in data["completed_experiments"]:
            data["completed_experiments"].append(key)
        data["best_results"][key] = {
            "auc_roc":            result.get("ensemble_auc"),
            "f1_score":           (result.get("final_metrics") or {}).get("f1_score"),
            "sensitivity":        (result.get("final_metrics") or {}).get("sensitivity"),
            "sens_at_90pct_spec": (result.get("final_metrics") or {}).get("sens_at_90pct_spec"),
        }
        PROGRESS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[warn] Could not update progress.json: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",     type=int, default=0,
                        help="Use top-N checkpoints by val AUC (0 = all)")
    parser.add_argument("--method",  choices=["mean", "weighted"], default="mean")
    parser.add_argument("--tta",     type=int, default=1,
                        help="TTA passes (1=no TTA, 5=flip+original+3 more)")
    parser.add_argument("--batch",   type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_asym", action="store_true",
                        help="Skip AsymmetryNet checkpoints")
    parser.add_argument("--include", type=str, default="",
                        help="Comma-separated checkpoint stems to include (e.g. "
                             "mv_effb3_attn_reg_e25,cv_convnext_base_e30). "
                             "Matched by substring against filename. "
                             "If omitted, all mv_*/cv_*/asymmetry_* are used.")
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"ERROR: manifest not found at {MANIFEST_CSV}")
        sys.exit(1)

    # Collect checkpoints: BreastPairNet (mv_*) + CrossView (cv_*) + AsymmetryNet
    ckpt_paths = sorted(LARGE_MODELS_DIR.glob("mv_*.pt")) + \
                 sorted(LARGE_MODELS_DIR.glob("cv_*.pt"))
    if not args.no_asym:
        asym_paths = [p for p in sorted(LARGE_MODELS_DIR.glob("asymmetry_*.pt"))
                      if "medical" not in p.name]
        ckpt_paths = ckpt_paths + asym_paths

    if args.include:
        stems = [s.strip() for s in args.include.split(",") if s.strip()]
        ckpt_paths = [p for p in ckpt_paths
                      if any(s in p.name for s in stems)]
        print(f"[--include] Filtered to {len(ckpt_paths)} checkpoints matching: {stems}")

    if not ckpt_paths:
        print(f"No checkpoints found in {LARGE_MODELS_DIR}")
        sys.exit(1)

    print(f"\nFound {len(ckpt_paths)} checkpoints (TTA passes={args.tta}):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    manifest = pd.read_csv(MANIFEST_CSV)

    # Run inference for each model, collect as {(sid,lat): prob} dicts
    all_prob_dicts = []
    model_aucs     = []
    model_info     = []

    for ckpt_path in ckpt_paths:
        print(f"\n[Ensemble] Loading {ckpt_path.name} ...")
        model, mtype, arch, fusion, img_sz, thresh, is_med = _load_checkpoint(ckpt_path)

        if mtype == "asymmetry":
            prob_d, label_d = _infer_asym_dict(
                model, manifest, img_sz, args.batch, args.workers, args.tta)
        else:
            # Both BreastPairNet and CrossViewTransformerNet share the same
            # (cc, mlo) -> logit interface and use BreastPairDataset
            prob_d, label_d = _infer_bp_dict(
                model, manifest, img_sz, args.batch, args.workers, args.tta)

        # Compute per-model AUC on common keys
        common_keys = sorted(prob_d.keys())
        p_arr = np.array([prob_d[k]  for k in common_keys])
        l_arr = np.array([label_d[k] for k in common_keys])
        m = _compute_metrics(l_arr.tolist(), p_arr.tolist(), 0.5)
        auc = m.get("auc_roc") or 0.0
        print(f"  type={mtype} arch={arch} fusion={fusion or 'asym-diff'} "
              f"img={img_sz}  AUC={auc:.4f}  s@90={m['sens_at_90pct_spec']}")

        all_prob_dicts.append(prob_d)
        model_aucs.append(auc)
        model_info.append({"checkpoint": ckpt_path.name, "type": mtype,
                            "arch": arch, "fusion": fusion,
                            "img_size": img_sz, "val_auc": auc})
        del model

    # Align all models on common (study_id, lat) keys
    key_sets  = [set(d.keys()) for d in all_prob_dicts]
    common    = sorted(set.intersection(*key_sets))
    labels    = np.array([label_d[k] for k in common])
    all_probs = np.array([[d[k] for k in common] for d in all_prob_dicts])
    model_aucs = np.array(model_aucs)

    print(f"\n[Ensemble] Aligned on {len(common)} breast predictions "
          f"from {len(all_prob_dicts)} models")

    # Optionally filter to top-N
    if args.top > 0 and args.top < len(model_aucs):
        top_idx    = np.argsort(model_aucs)[-args.top:]
        all_probs  = all_probs[top_idx]
        model_aucs = model_aucs[top_idx]
        model_info = [model_info[i] for i in top_idx]
        print(f"[Ensemble] Using top-{args.top}: {[m['checkpoint'] for m in model_info]}")

    # Combine
    if args.method == "weighted":
        w = model_aucs / model_aucs.sum()
        ens_probs = (all_probs * w[:, None]).sum(axis=0)
        print(f"[Ensemble] Weights: { {m['checkpoint']: round(float(wi),3) for m,wi in zip(model_info,w)} }")
    else:
        ens_probs = all_probs.mean(axis=0)

    thresh_opt  = find_optimal_threshold(labels.tolist(), ens_probs.tolist())
    final_m     = _compute_metrics(labels.tolist(), ens_probs.tolist(), 0.5)
    final_m_opt = _compute_metrics(labels.tolist(), ens_probs.tolist(), thresh_opt)
    ens_auc     = final_m.get("auc_roc") or 0.0

    tta_tag  = f"_tta{args.tta}" if args.tta > 1 else ""
    asym_tag = "" if args.no_asym else "_+asym"
    n_cv     = sum(1 for m in model_info if m["type"] == "crossview")
    cv_tag   = f"_+cv{n_cv}" if n_cv > 0 else ""
    print(f"\n{'='*65}")
    print(f"  ENSEMBLE ({args.method}, {len(model_info)} models{asym_tag}{cv_tag}{tta_tag})")
    print(f"  AUC={ens_auc:.4f}  F1={final_m['f1_score']:.4f}  "
          f"sens={final_m['sensitivity']:.4f}  s@90={final_m['sens_at_90pct_spec']}")
    print(f"  F1@opt={final_m_opt['f1_score']:.4f}(t={thresh_opt:.3f})  "
          f"sens@opt={final_m_opt['sensitivity']:.4f}")
    print(f"{'='*65}\n")

    from datetime import datetime
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = f"ensemble_{args.method}_top{len(model_info)}{asym_tag}{cv_tag}{tta_tag}_{ts}"
    result = {
        "method": f"ensemble_{args.method}",
        "n_models": len(model_info), "tta_passes": args.tta,
        "include_asymmetry": not args.no_asym,
        "models": model_info,
        "ensemble_auc":   round(ens_auc, 4),
        "final_metrics":  final_m,
        "final_metrics_opt": final_m_opt,
        "opt_threshold":  round(thresh_opt, 4),
    }
    out = LARGE_RESULTS_DIR / f"{key}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[Ensemble] Saved -> {out}")
    _mark_done(key, result)


if __name__ == "__main__":
    main()
