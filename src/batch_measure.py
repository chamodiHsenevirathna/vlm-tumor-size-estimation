"""
Milestone 3: small-batch validation of the tumor-size measurement pipeline
(Milestone 2's method) across 10 KiPA22 cases (IDs 0-9), extracted straight
from the already-downloaded data/train.zip -- no re-download, no full unzip.

For each case, reuses Milestone 2's functions unchanged (fit_ellipse_real_space,
select_largest_tumor_slice, largest_contour, visualize) from measure_tumor.py,
and adds:

  - a connected-component count on the selected slice
  - a `medvision_qc_flag` column that flags cases which would likely be
    excluded (or need a closer look) under MedVision's real quality-control
    rules, WITHOUT excluding them here -- we still measure and record them.

QC flags (checked in this priority order; first match wins):
  - multiple_components : slice's raw tumor mask has >1 disconnected blob
                           (MedVision: n_total_clusters > 1 -> slice rejected)
  - too_small            : largest component < 20 px
                           (MedVision v1.1.0 minimum cluster-size threshold)
  - ellipse_outside_bbox : any of the 4 ellipse axis endpoints falls outside
                           the 1.1x-enlarged cluster bounding box, or all 4
                           fall inside the 0.9x-shrunk bounding box
                           (MedVision: "all_within" bounding-box sanity check)
  - other_issue          : the QC check itself raised an unexpected error
  - pass_basic           : none of the above triggered

NOT implemented here (see Milestone 3 plan): annotation-version handling,
multi-orientation (coronal/sagittal) slicing, and MedVision's own precomputed
biometric_profile annotations -- we compute fits ourselves throughout.
"""

import csv
import zipfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

from measure_tumor import (
    fit_ellipse_real_space,
    largest_contour,
    select_largest_tumor_slice,
    visualize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data" / "train.zip"
EXTRACT_DIR = REPO_ROOT / "data" / "batch_cases"
RESULTS_DIR = REPO_ROOT / "results"
BATCH_EXAMPLES_DIR = REPO_ROOT / "outputs" / "batch_examples"

TUMOR_LABEL = 4
CASE_IDS = [str(i) for i in range(10)]  # "0".."9"
MIN_CLUSTER_SIZE_PX = 20  # MedVision v1.1.0 threshold
BBOX_SHRINK = 0.9
BBOX_ENLARGE = 1.1


def extract_case(case_id: str) -> tuple[Path, Path]:
    """Extract one case's image+label from the local train.zip, if not already extracted."""
    image_out = EXTRACT_DIR / "image" / f"{case_id}.nii.gz"
    label_out = EXTRACT_DIR / "label" / f"{case_id}.nii.gz"
    if image_out.exists() and label_out.exists():
        return image_out, label_out

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(f"train/image/{case_id}.nii.gz") as src, open(image_out, "wb") as dst:
            dst.write(src.read())
        with zf.open(f"train/label/{case_id}.nii.gz") as src, open(label_out, "wb") as dst:
            dst.write(src.read())
    return image_out, label_out


def count_components(mask_2d: np.ndarray) -> int:
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return len(contours)


def bbox_qc_pass(contour_xy: np.ndarray, fit: dict) -> bool:
    """MedVision's 0.9x/1.1x bounding-box sanity check: all 4 ellipse axis
    endpoints must lie in the buffer zone between the shrunk and enlarged
    bounding box of the tumor contour (in array/pixel coordinates)."""
    xs, ys = contour_xy[:, 0], contour_xy[:, 1]  # x=dim1, y=dim0
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    hw, hh = (x_max - x_min) / 2.0, (y_max - y_min) / 2.0

    def in_box(px, py, scale):
        return (abs(px - cx) <= hw * scale) and (abs(py - cy) <= hh * scale)

    points_d1d0 = [fit["major"][0], fit["major"][1], fit["minor"][0], fit["minor"][1]]
    for (d0, d1) in points_d1d0:
        px, py = d1, d0  # back to (x, y) = (dim1, dim0)
        if not in_box(px, py, BBOX_ENLARGE):
            return False  # outside enlarged box -> fit too big / off
    if all(in_box(d1, d0, BBOX_SHRINK) for (d0, d1) in points_d1d0):
        return False  # all points inside shrunk box -> fit too small
    return True


def process_case(case_id: str) -> dict:
    row = {
        "case_id": case_id,
        "ct_shape": "",
        "voxel_spacing_mm": "",
        "selected_slice_idx": "",
        "tumor_voxel_count_3d": "",
        "tumor_area_slice_px": "",
        "n_components": "",
        "major_axis_mm": "",
        "minor_axis_mm": "",
        "status": "failed",
        "failure_reason": "",
        "medvision_qc_flag": "",
    }

    try:
        image_path, label_path = extract_case(case_id)
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image = image_nii.get_fdata()
        label = label_nii.get_fdata()
        voxel_spacing = image_nii.header.get_zooms()

        row["ct_shape"] = image.shape
        row["voxel_spacing_mm"] = tuple(round(float(v), 4) for v in voxel_spacing)

        tumor_mask = label == TUMOR_LABEL
        tumor_voxel_count = int(tumor_mask.sum())
        row["tumor_voxel_count_3d"] = tumor_voxel_count
        if tumor_voxel_count == 0:
            row["failure_reason"] = "no tumor voxels in volume"
            return row

        slice_idx = select_largest_tumor_slice(tumor_mask)
        row["selected_slice_idx"] = slice_idx
        image_2d = image[:, :, slice_idx]
        mask_2d = tumor_mask[:, :, slice_idx]
        row["tumor_area_slice_px"] = int(mask_2d.sum())

        n_components = count_components(mask_2d)
        row["n_components"] = n_components

        try:
            contour_xy = largest_contour(mask_2d)
        except RuntimeError as exc:
            row["failure_reason"] = str(exc)
            return row

        pixel_sizes_2d = (float(voxel_spacing[0]), float(voxel_spacing[1]))

        try:
            fit = fit_ellipse_real_space(contour_xy, pixel_sizes_2d)
        except cv2.error as exc:
            row["failure_reason"] = f"cv2.fitEllipse failed (likely <5 contour points): {exc}"
            return row

        row["major_axis_mm"] = round(fit["major_mm"], 2)
        row["minor_axis_mm"] = round(fit["minor_mm"], 2)
        row["status"] = "success"
        row["failure_reason"] = ""

        # --- MedVision-style QC flag (informational only, nothing excluded) ---
        largest_area = cv2.contourArea(contour_xy.reshape(-1, 1, 2).astype(np.int32))
        try:
            if n_components > 1:
                row["medvision_qc_flag"] = "multiple_components"
            elif largest_area < MIN_CLUSTER_SIZE_PX:
                row["medvision_qc_flag"] = "too_small"
            elif not bbox_qc_pass(contour_xy, fit):
                row["medvision_qc_flag"] = "ellipse_outside_bbox"
            else:
                row["medvision_qc_flag"] = "pass_basic"
        except Exception:
            row["medvision_qc_flag"] = "other_issue"

        # Stash extra data needed by the caller for visualization, not written to CSV.
        row["_viz"] = (image_2d, mask_2d, contour_xy, fit)
        return row

    except Exception as exc:  # noqa: BLE001 - batch must keep going on any per-case error
        row["failure_reason"] = f"unexpected error: {exc}"
        return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = [process_case(cid) for cid in CASE_IDS]

    # --- write CSV (drop the internal _viz field) ---
    fieldnames = [k for k in rows[0].keys() if k != "_viz"]
    csv_path = RESULTS_DIR / "batch_measurements.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "_viz"})

    # --- save 3 example visualizations: first success, a multi-component case
    # if one exists in this batch, and one more success for variety ---
    successes = [r for r in rows if r["status"] == "success"]
    multi = [r for r in successes if r["medvision_qc_flag"] == "multiple_components"]

    examples = []
    if successes:
        examples.append(successes[0])
    if multi and multi[0] not in examples:
        examples.append(multi[0])
    for r in successes:
        if len(examples) >= 3:
            break
        if r not in examples:
            examples.append(r)

    example_paths = []
    for r in examples:
        image_2d, mask_2d, contour_xy, fit = r["_viz"]
        out_path = BATCH_EXAMPLES_DIR / f"case_{r['case_id']}_measurement.png"
        visualize(image_2d, mask_2d, contour_xy, fit, r["selected_slice_idx"], out_path)
        example_paths.append(out_path)

    # --- summary ---
    n_success = len(successes)
    n_failed = len(rows) - n_success
    qc_counts = {}
    for r in rows:
        flag = r["medvision_qc_flag"] or "(n/a - failed case)"
        qc_counts[flag] = qc_counts.get(flag, 0) + 1

    print("=== Batch summary (10 cases) ===")
    print(f"Success: {n_success}/10")
    print(f"Failed:  {n_failed}/10")
    print("\nQC flag counts:")
    for flag, count in sorted(qc_counts.items()):
        print(f"  {flag}: {count}")
    print(f"\nCSV written to: {csv_path}")
    print("Example visualizations:")
    for p in example_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
