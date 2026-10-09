"""
Phase 0 validation: regenerate five representative cases with the corrected
(offset-aware) preprocessing and compare against the original (offset=0) export.

Writes ONLY under validation/phase0/ (never outputs/ or results/):
  model_input/case_{id}_slice.png   corrected image exactly as the VLM would receive it (no overlays)
  qc_overlay/case_{id}_qc.png       corrected image with kidney/tumor outlines -- QC ONLY, never model input
  contact_sheet.png                 old | corrected | corrected + QC overlay, one row per case
  metrics.csv                       quality metrics for old and corrected exports

Reads data/batch_cases/ (already extracted locally) and results/batch_measurements.csv.
No model inference, no downloads.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

from ct_preprocessing import (
    DEFAULT_INTENSITY_OFFSET,
    DEFAULT_WINDOW_CENTER,
    DEFAULT_WINDOW_WIDTH,
    assess_export,
    window_ct_slice,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = REPO_ROOT / "data" / "batch_cases"
BATCH_CSV = REPO_ROOT / "results" / "batch_measurements.csv"
COMMITTED_OLD_DIR = REPO_ROOT / "outputs" / "vlm_batch5"
OUT_DIR = REPO_ROOT / "validation" / "phase0"

CASE_IDS = ["0", "2", "4", "6", "8"]  # smallest tumor, narrow range, floor-clipped, largest spacing, largest tumor
KIDNEY_LABEL, TUMOR_LABEL = 2, 4
METRIC_KEYS = ["window_coverage", "kidney_median_hu", "tumor_mean_grey", "tumor_clipped_fraction",
               "n_unique_levels", "grey_std", "entropy_bits", "frac_ge_254", "frac_le_1"]


def slice_indices() -> dict:
    with open(BATCH_CSV, newline="") as f:
        return {r["case_id"]: int(r["selected_slice_idx"]) for r in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--intensity-offset", type=float, default=DEFAULT_INTENSITY_OFFSET)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.output_dir
    for sub in ("model_input", "qc_overlay"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    slices = slice_indices()
    rows, sheet = [], []
    for cid in CASE_IDS:
        idx = slices[cid]
        raw = np.asanyarray(nib.load(str(EXTRACT_DIR / "image" / f"{cid}.nii.gz")).dataobj[:, :, idx])
        lab = np.asanyarray(nib.load(str(EXTRACT_DIR / "label" / f"{cid}.nii.gz")).dataobj[:, :, idx])
        kidney, tumor = lab == KIDNEY_LABEL, lab == TUMOR_LABEL

        variants = {
            "old_offset0": args.intensity_offset * 0.0,
            "corrected": args.intensity_offset,
        }
        grey = {}
        for name, off in variants.items():
            grey[name] = window_ct_slice(raw, off, DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH)
            rep = assess_export(grey[name], raw, intensity_offset=off, center=DEFAULT_WINDOW_CENTER,
                                width=DEFAULT_WINDOW_WIDTH, kidney_mask=kidney, tumor_mask=tumor)
            rows.append({"case_id": cid, "slice_idx": idx, "variant": name, "intensity_offset": off,
                         "passed": rep.ok, "failures": " | ".join(rep.failures),
                         **{k: rep.metrics.get(k) for k in METRIC_KEYS}})

        # sanity: the offset-0 reproduction must equal the original committed export, where one exists
        committed = COMMITTED_OLD_DIR / f"case_{cid}_slice.png"
        if committed.exists():
            same = np.array_equal(np.array(Image.open(committed)), grey["old_offset0"].T)
            print(f"case {cid}: offset-0 reproduction matches committed original PNG: {same}")
            rows[-2]["failures"] += "" if same else " | WARNING: reproduction != committed PNG"
        else:
            print(f"case {cid}: no committed original PNG (reproduced with offset 0)")

        # model input: exactly what a VLM would see
        Image.fromarray(grey["corrected"].T, mode="L").save(out / "model_input" / f"case_{cid}_slice.png")

        # QC overlay: separate file, outlines drawn here only
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grey["corrected"].T, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.contour(kidney.T, levels=[0.5], colors="yellow", linewidths=0.8)
        ax.contour(tumor.T, levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_title(f"case {cid} slice {idx} - QC overlay (NOT model input)", fontsize=8)
        ax.axis("off")
        fig.savefig(out / "qc_overlay" / f"case_{cid}_qc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        sheet.append((cid, idx, grey["old_offset0"], grey["corrected"], kidney, tumor))

    # contact sheet
    fig, axes = plt.subplots(len(sheet), 3, figsize=(10, 3.4 * len(sheet)))
    for r, (cid, idx, old, new, kidney, tumor) in enumerate(sheet):
        for c, (img, title) in enumerate([(old, "original export (offset 0)"),
                                          (new, f"corrected (offset {args.intensity_offset:g}) = model input"),
                                          (new, "corrected + QC overlay (not model input)")]):
            ax = axes[r, c]
            ax.imshow(img.T, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if c == 2:
                ax.contour(kidney.T, levels=[0.5], colors="yellow", linewidths=0.8)
                ax.contour(tumor.T, levels=[0.5], colors="cyan", linewidths=1.0)
            ax.set_title(f"case {cid} (slice {idx}): {title}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Phase 0 preprocessing validation. Yellow=kidney, cyan=tumor (QC only).", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "contact_sheet.png", dpi=130)
    plt.close(fig)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'case':>4} {'variant':>12} {'pass':>5} {'cover':>6} {'kidHU':>6} {'tumGrey':>7} {'uniq':>5} {'std':>5} {'H':>5} {'>=254':>6}")
    for r in rows:
        f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
        print(f"{r['case_id']:>4} {r['variant']:>12} {str(r['passed']):>5} {f(r['window_coverage']):>6} "
              f"{f(r['kidney_median_hu'], 0):>6} {f(r['tumor_mean_grey'], 0):>7} {r['n_unique_levels']:>5} "
              f"{f(r['grey_std'], 1):>5} {f(r['entropy_bits']):>5} {f(r['frac_ge_254']):>6}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
