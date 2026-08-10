"""
Milestone 4: diagnose why 4/10 KiPA22 batch cases (2, 4, 7, 9) were flagged
`ellipse_outside_bbox` by our MedVision-style QC check, vs. 2 clean
`pass_basic` cases (0, 8), from Milestone 3's batch.

Reuses Milestone 2/3 functions unchanged (select_largest_tumor_slice,
largest_contour, fit_ellipse_real_space) on cases already extracted to
data/batch_cases/ by batch_measure.py -- no re-download, no new cases.

For each case, this script makes visible the intermediate values that
batch_measure.py's bbox_qc_pass() computed internally but never exposed:
the raw tumor bounding box, the 0.9x-shrunk and 1.1x-enlarged buffer boxes,
and which of the 4 ellipse axis endpoints falls outside the acceptable
buffer zone (and how).

It also computes simple shape diagnostics per case (bbox fill ratio, axis
ratio, solidity via convex hull) to see whether flagged cases are
systematically more irregular/elongated than clean ones.
"""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np

from measure_tumor import fit_ellipse_real_space, largest_contour, select_largest_tumor_slice

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = REPO_ROOT / "data" / "batch_cases"
RESULTS_DIR = REPO_ROOT / "results"
QC_VIZ_DIR = REPO_ROOT / "outputs" / "qc_analysis"

TUMOR_LABEL = 4
CLEAN_CASE_IDS = ["0", "8"]
FLAGGED_CASE_IDS = ["2", "4", "7", "9"]
ANALYSIS_CASE_IDS = CLEAN_CASE_IDS + FLAGGED_CASE_IDS

BBOX_SHRINK = 0.9
BBOX_ENLARGE = 1.1


def load_case(case_id: str):
    image_path = EXTRACT_DIR / "image" / f"{case_id}.nii.gz"
    label_path = EXTRACT_DIR / "label" / f"{case_id}.nii.gz"
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    return image_nii.get_fdata(), label_nii.get_fdata(), image_nii.header.get_zooms()


def tumor_bbox(contour_xy: np.ndarray) -> dict:
    xs, ys = contour_xy[:, 0], contour_xy[:, 1]  # x=dim1, y=dim0
    x_min, x_max, y_min, y_max = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    hw, hh = (x_max - x_min) / 2.0, (y_max - y_min) / 2.0
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "cx": cx, "cy": cy, "hw": hw, "hh": hh}


def bbox_qc_detail(contour_xy: np.ndarray, fit: dict) -> dict:
    """Same rule as batch_measure.py's bbox_qc_pass, but returns full detail:
    per-endpoint pass/fail against the enlarged/shrunk boxes, and why."""
    bbox = tumor_bbox(contour_xy)
    cx, cy, hw, hh = bbox["cx"], bbox["cy"], bbox["hw"], bbox["hh"]

    def in_box(px, py, scale):
        return (abs(px - cx) <= hw * scale) and (abs(py - cy) <= hh * scale)

    endpoints = {
        "major_p1": fit["major"][0],
        "major_p2": fit["major"][1],
        "minor_p1": fit["minor"][0],
        "minor_p2": fit["minor"][1],
    }

    endpoint_detail = {}
    violating = []
    for name, (d0, d1) in endpoints.items():
        px, py = d1, d0  # (x, y) = (dim1, dim0)
        in_enlarged = in_box(px, py, BBOX_ENLARGE)
        in_shrunk = in_box(px, py, BBOX_SHRINK)
        endpoint_detail[name] = {"x": px, "y": py, "in_enlarged": in_enlarged, "in_shrunk": in_shrunk}
        if not in_enlarged:
            violating.append((name, "outside_enlarged_box"))

    all_inside_shrunk = all(d["in_shrunk"] for d in endpoint_detail.values())
    if all_inside_shrunk:
        for name in endpoints:
            violating.append((name, "inside_shrunk_box (fit too small)"))

    passed = len(violating) == 0
    return {
        "bbox": bbox,
        "endpoint_detail": endpoint_detail,
        "violating": violating,
        "passed": passed,
        "all_inside_shrunk": all_inside_shrunk,
    }


def shape_diagnostics(contour_xy: np.ndarray, fit: dict, bbox: dict) -> dict:
    contour_i32 = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    area_px = cv2.contourArea(contour_i32)
    hull = cv2.convexHull(contour_i32)
    hull_area_px = cv2.contourArea(hull)
    solidity = area_px / hull_area_px if hull_area_px > 0 else float("nan")

    bbox_w = bbox["x_max"] - bbox["x_min"]
    bbox_h = bbox["y_max"] - bbox["y_min"]
    bbox_area = bbox_w * bbox_h
    fill_ratio = area_px / bbox_area if bbox_area > 0 else float("nan")

    axis_ratio = fit["major_mm"] / fit["minor_mm"] if fit["minor_mm"] > 0 else float("nan")

    return {
        "tumor_area_px": area_px,
        "bbox_w_px": round(bbox_w, 1),
        "bbox_h_px": round(bbox_h, 1),
        "bbox_fill_ratio": round(fill_ratio, 3),
        "solidity": round(solidity, 3),
        "axis_ratio": round(axis_ratio, 3),
    }


def visualize_case(image_2d, mask_2d, contour_xy, fit, detail, slice_idx, case_id, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_2d.T, cmap="gray", origin="lower")

    # Tumor outline
    contour_plot = np.vstack([contour_xy, contour_xy[0]])
    ax.plot(contour_plot[:, 0], contour_plot[:, 1], color="cyan", linewidth=1.5, label="Tumor outline")

    bbox = detail["bbox"]
    cx, cy, hw, hh = bbox["cx"], bbox["cy"], bbox["hw"], bbox["hh"]

    # Raw tumor bbox
    ax.add_patch(patches.Rectangle((bbox["x_min"], bbox["y_min"]), bbox["x_max"] - bbox["x_min"],
                                    bbox["y_max"] - bbox["y_min"], fill=False, edgecolor="white",
                                    linewidth=1, linestyle="-", label="Tumor bbox"))
    # Shrunk (0.9x) box
    ax.add_patch(patches.Rectangle((cx - hw * BBOX_SHRINK, cy - hh * BBOX_SHRINK), 2 * hw * BBOX_SHRINK,
                                    2 * hh * BBOX_SHRINK, fill=False, edgecolor="orange",
                                    linewidth=1.5, linestyle="--", label="0.9x shrunk box"))
    # Enlarged (1.1x) box
    ax.add_patch(patches.Rectangle((cx - hw * BBOX_ENLARGE, cy - hh * BBOX_ENLARGE), 2 * hw * BBOX_ENLARGE,
                                    2 * hh * BBOX_ENLARGE, fill=False, edgecolor="orange",
                                    linewidth=1.5, linestyle=":", label="1.1x enlarged box"))

    # Fitted ellipse
    theta = np.linspace(0, 2 * np.pi, 200)
    maj, minr = fit["major"], fit["minor"]
    maj_vec = np.array([maj[1][0] - maj[0][0], maj[1][1] - maj[0][1]]) / 2.0
    min_vec = np.array([minr[1][0] - minr[0][0], minr[1][1] - minr[0][1]]) / 2.0
    center_arr = ((maj[0][0] + maj[1][0]) / 2.0, (maj[0][1] + maj[1][1]) / 2.0)
    ellipse_d0 = center_arr[0] + np.outer(np.cos(theta), maj_vec[0]) + np.outer(np.sin(theta), min_vec[0])
    ellipse_d1 = center_arr[1] + np.outer(np.cos(theta), maj_vec[1]) + np.outer(np.sin(theta), min_vec[1])
    ax.plot(ellipse_d1.ravel(), ellipse_d0.ravel(), color="lime", linewidth=1.5, label="Fitted ellipse")

    (m0a, m1a), (m0b, m1b) = fit["major"]
    ax.plot([m1a, m1b], [m0a, m0b], color="red", linewidth=2, label=f"Major: {fit['major_mm']:.1f} mm")
    (n0a, n1a), (n0b, n1b) = fit["minor"]
    ax.plot([n1a, n1b], [n0a, n0b], color="yellow", linewidth=2, label=f"Minor: {fit['minor_mm']:.1f} mm")

    # Mark violating endpoints with a red X
    violating_names = {name for name, _ in detail["violating"]}
    for name, d in detail["endpoint_detail"].items():
        if name in violating_names:
            ax.plot(d["x"], d["y"], marker="x", color="magenta", markersize=14, markeredgewidth=3,
                    label="Violating endpoint" if name == next(iter(violating_names)) else None)

    status = "PASS" if detail["passed"] else "FLAGGED: ellipse_outside_bbox"
    ax.set_title(f"Case {case_id} — slice {slice_idx} — {status}")
    ax.legend(loc="upper right", fontsize=7, facecolor="black", labelcolor="white")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_case(case_id: str) -> dict:
    image, label, voxel_spacing = load_case(case_id)
    tumor_mask = label == TUMOR_LABEL
    slice_idx = select_largest_tumor_slice(tumor_mask)
    image_2d = image[:, :, slice_idx]
    mask_2d = tumor_mask[:, :, slice_idx]
    pixel_sizes_2d = (float(voxel_spacing[0]), float(voxel_spacing[1]))

    contour_xy = largest_contour(mask_2d)
    fit = fit_ellipse_real_space(contour_xy, pixel_sizes_2d)
    detail = bbox_qc_detail(contour_xy, fit)
    diagnostics = shape_diagnostics(contour_xy, fit, detail["bbox"])

    out_path = QC_VIZ_DIR / f"case_{case_id}_qc.png"
    visualize_case(image_2d, mask_2d, contour_xy, fit, detail, slice_idx, case_id, out_path)

    violating_str = "; ".join(f"{name} ({reason})" for name, reason in detail["violating"]) or "none"

    return {
        "case_id": case_id,
        "group": "clean" if case_id in CLEAN_CASE_IDS else "flagged",
        "qc_flag": "pass_basic" if detail["passed"] else "ellipse_outside_bbox",
        "slice_idx": slice_idx,
        "major_axis_mm": round(fit["major_mm"], 2),
        "minor_axis_mm": round(fit["minor_mm"], 2),
        **diagnostics,
        "violating_endpoints": violating_str,
        "violation_type": "inside_shrunk_box (fit too small)" if detail["all_inside_shrunk"]
        else ("outside_enlarged_box (fit too big/off)" if detail["violating"] else "none"),
        "viz_path": str(out_path),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QC_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    rows = [analyze_case(cid) for cid in ANALYSIS_CASE_IDS]

    csv_path = RESULTS_DIR / "qc_analysis.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    clean_rows = [r for r in rows if r["group"] == "clean"]
    flagged_rows = [r for r in rows if r["group"] == "flagged"]

    def avg(rows_, key):
        return sum(r[key] for r in rows_) / len(rows_)

    metrics = ["bbox_fill_ratio", "solidity", "axis_ratio", "tumor_area_px"]

    print("=== QC analysis results ===\n")
    print(f"{'case':>5} {'group':>8} {'qc_flag':>20} {'fill':>6} {'solidity':>9} {'axis_ratio':>11} {'violation_type'}")
    for r in rows:
        print(f"{r['case_id']:>5} {r['group']:>8} {r['qc_flag']:>20} {r['bbox_fill_ratio']:>6} "
              f"{r['solidity']:>9} {r['axis_ratio']:>11} {r['violation_type']}")

    print("\n=== Clean vs flagged: average shape metrics ===")
    for m in metrics:
        print(f"  {m:>16}: clean={avg(clean_rows, m):.3f}   flagged={avg(flagged_rows, m):.3f}")

    print("\n=== Per-flagged-case explanation ===")
    for r in flagged_rows:
        print(f"Case {r['case_id']}: {r['violation_type']}; fill_ratio={r['bbox_fill_ratio']}, "
              f"solidity={r['solidity']}, axis_ratio={r['axis_ratio']}, "
              f"violating endpoints: {r['violating_endpoints']}")

    print(f"\nCSV written to: {csv_path}")
    print("Visualizations:")
    for r in rows:
        print(f"  {r['viz_path']}")


if __name__ == "__main__":
    main()
