"""
Milestone 6: prepare windowed CT slice images for the 5-case scored VLM batch
plus 1 exploratory flagged case, reusing Milestone 3's already-extracted case
files (data/batch_cases/) and Milestone 3's reference measurements
(results/batch_measurements.csv) -- no re-download, no new reference values.

Same HU windowing as Milestone 5 (center=40, width=400, abdominal soft-tissue
window) applied via ct_preprocessing (stored intensities are converted to HU
first using a configurable, unverified offset -- see ct_preprocessing.py; every
export is validated before saving), no tumor overlay/outline/bbox/scale bar -- identical image style,
just applied per-case instead of to a single case.

Also writes a small metadata CSV (case_id, scored, slice_idx, pixel_spacing_mm,
major_mm_ref, minor_mm_ref) that the Colab notebook reads to build its prompts,
so the reference values used for scoring always come straight from
batch_measurements.csv, never retyped by hand.
"""

import argparse
import ast
import csv
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from ct_preprocessing import (
    DEFAULT_INTENSITY_OFFSET,
    DEFAULT_WINDOW_CENTER,
    DEFAULT_WINDOW_WIDTH,
    validate_export,
    window_ct_slice,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_CSV = REPO_ROOT / "results" / "batch_measurements.csv"
EXTRACT_DIR = REPO_ROOT / "data" / "batch_cases"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vlm_batch5"

WINDOW_CENTER = DEFAULT_WINDOW_CENTER
WINDOW_WIDTH = DEFAULT_WINDOW_WIDTH
TUMOR_LABEL = 4
KIDNEY_LABEL = 2

SCORED_CASE_IDS = ["0", "3", "5", "6", "8"]
EXPLORATORY_CASE_IDS = ["2"]
ALL_CASE_IDS = SCORED_CASE_IDS + EXPLORATORY_CASE_IDS


def load_batch_reference() -> dict:
    """Read case_id -> {slice_idx, pixel_spacing_mm, major_mm_ref, minor_mm_ref, qc_flag} from Milestone 3's CSV."""
    ref = {}
    with open(BATCH_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["case_id"] not in ALL_CASE_IDS:
                continue
            voxel_spacing = ast.literal_eval(row["voxel_spacing_mm"])
            ref[row["case_id"]] = {
                "slice_idx": int(row["selected_slice_idx"]),
                "pixel_spacing_mm": round(float(voxel_spacing[0]), 6),  # in-plane spacing (isotropic per case)
                "major_mm_ref": float(row["major_axis_mm"]),
                "minor_mm_ref": float(row["minor_axis_mm"]),
                "qc_flag": row["medvision_qc_flag"],
            }
    missing = set(ALL_CASE_IDS) - set(ref.keys())
    if missing:
        raise RuntimeError(f"Case IDs not found in {BATCH_CSV}: {missing}")
    return ref


def prepare_case_image(case_id: str, slice_idx: int, output_dir: Path, intensity_offset: float) -> Path:
    image_path = EXTRACT_DIR / "image" / f"{case_id}.nii.gz"
    label_path = EXTRACT_DIR / "label" / f"{case_id}.nii.gz"
    slice_raw = np.asanyarray(nib.load(str(image_path)).dataobj[:, :, slice_idx])  # stored values, unscaled
    label_slice = np.asanyarray(nib.load(str(label_path)).dataobj[:, :, slice_idx])

    windowed = window_ct_slice(slice_raw, intensity_offset, WINDOW_CENTER, WINDOW_WIDTH)
    validate_export(
        windowed, slice_raw,
        intensity_offset=intensity_offset, center=WINDOW_CENTER, width=WINDOW_WIDTH,
        kidney_mask=label_slice == KIDNEY_LABEL, tumor_mask=label_slice == TUMOR_LABEL,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"case_{case_id}_slice.png"
    Image.fromarray(windowed.T, mode="L").save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--intensity-offset", type=float, default=DEFAULT_INTENSITY_OFFSET,
                        help="stored = HU + offset (default %(default)s; unverified hypothesis)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir

    reference = load_batch_reference()

    metadata_rows = []
    image_paths = []
    for case_id in ALL_CASE_IDS:
        info = reference[case_id]
        out_path = prepare_case_image(case_id, info["slice_idx"], output_dir, args.intensity_offset)
        image_paths.append(out_path)
        metadata_rows.append({
            "case_id": case_id,
            "scored": case_id in SCORED_CASE_IDS,
            "slice_idx": info["slice_idx"],
            "pixel_spacing_mm": info["pixel_spacing_mm"],
            "major_mm_ref": info["major_mm_ref"],
            "minor_mm_ref": info["minor_mm_ref"],
            "qc_flag": info["qc_flag"],
            "image_filename": out_path.name,
        })

    metadata_path = output_dir / "batch_metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    zip_path = output_dir / "images.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in image_paths:
            zf.write(p, arcname=p.name)
        zf.write(metadata_path, arcname=metadata_path.name)

    print("=== Milestone 6 batch preparation ===")
    for row in metadata_rows:
        tag = "SCORED" if row["scored"] else "EXPLORATORY"
        print(f"  case {row['case_id']:>2} [{tag:>11}]  slice={row['slice_idx']:>3}  "
              f"spacing={row['pixel_spacing_mm']}mm  ref=({row['major_mm_ref']}, {row['minor_mm_ref']})mm  "
              f"qc_flag={row['qc_flag']}  -> {row['image_filename']}")
    print(f"\nMetadata CSV: {metadata_path}")
    print(f"Zip for Colab upload: {zip_path}")


if __name__ == "__main__":
    main()
