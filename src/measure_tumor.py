"""
Milestone 2: MedVision-style tumor size measurement for the single KiPA22 case
already extracted by load_sample.py (data/one_case/train/{image,label}/0.nii.gz).

Method (mirrors MedVision's `medvision_ds.__fit_ellipses` / their
`viz_ellipse_fit_comparison.py`):
    1. Take the kidney-tumor mask (label 4) only.
    2. Pick the axial slice (fixed index along dim 2) with the largest tumor area.
    3. Find the tumor's contour on that slice; if there are multiple disconnected
       blobs, keep the largest one (MedVision's production pipeline would instead
       reject such a slice — we simplify since this is a single hand-inspected case).
    4. Fit an ellipse IN REAL-WORLD (mm) SPACE: scale contour points by the
       per-axis pixel spacing before calling cv2.fitEllipse, then map the fitted
       axis endpoints back to array coordinates for drawing. Fitting directly in
       mm-space (rather than fitting in pixels and scaling the result) is what
       MedVision does, because anisotropic per-axis scaling is not a similarity
       transform -- it can rotate the principal axes and change their lengths.
    5. Report major/minor axis length in mm.

Deviations from the full MedVision pipeline (intentional, for this single-case
milestone): no n_total_clusters > 1 rejection, no 0.9x/1.1x bounding-box quality
check, no minimum cluster-size threshold.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "one_case" / "train"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
IMAGE_PATH = DATA_DIR / "image" / "0.nii.gz"
LABEL_PATH = DATA_DIR / "label" / "0.nii.gz"

TUMOR_LABEL = 4


def _phys_len(a, b, ps):
    """Physical length (mm) of segment a-b; a, b are (dim0, dim1) array coords."""
    return float(np.hypot((a[0] - b[0]) * ps[0], (a[1] - b[1]) * ps[1]))


def fit_ellipse_real_space(contour_xy: np.ndarray, pixel_sizes: tuple[float, float]) -> dict:
    """Fit an ellipse to a contour in real-world (mm) space.

    contour_xy : (N, 2) cv2 contour points, ordered (x=col=dim1, y=row=dim0).
    pixel_sizes: (ps_dim0, ps_dim1) mm -- spacing along array axes 0 and 1.

    Returns endpoints of the major/minor axes in ARRAY space (dim0, dim1), plus
    their physical lengths in mm.
    """
    ps0, ps1 = pixel_sizes

    # Scale to mm: cv2 x -> dim1 -> * ps1 ; cv2 y -> dim0 -> * ps0
    pts_mm = contour_xy.astype(np.float32).copy()
    pts_mm[:, 0] *= ps1
    pts_mm[:, 1] *= ps0

    (cx, cy), (ax0, ax1), angle_deg = cv2.fitEllipse(pts_mm)
    angle_rad = np.deg2rad(angle_deg)
    a, b = ax0 / 2.0, ax1 / 2.0
    mvx, mvy = a * np.cos(angle_rad), a * np.sin(angle_rad)
    nvx, nvy = -b * np.sin(angle_rad), b * np.cos(angle_rad)

    def back_to_array(px, py):
        """mm-space (x, y) -> array space (dim0, dim1)."""
        return (py / ps0, px / ps1)

    p1, p2 = back_to_array(cx + mvx, cy + mvy), back_to_array(cx - mvx, cy - mvy)
    p3, p4 = back_to_array(cx + nvx, cy + nvy), back_to_array(cx - nvx, cy - nvy)

    len_a = _phys_len(p1, p2, pixel_sizes)
    len_b = _phys_len(p3, p4, pixel_sizes)
    major, minor = ((p1, p2), (p3, p4)) if len_a >= len_b else ((p3, p4), (p1, p2))
    maj_len, min_len = max(len_a, len_b), min(len_a, len_b)

    return {
        "center_mm": (cx, cy),
        "angle_deg": angle_deg,
        "major": major,
        "minor": minor,
        "major_mm": maj_len,
        "minor_mm": min_len,
    }


def select_largest_tumor_slice(tumor_mask: np.ndarray) -> int:
    """Return the index along axis 2 (axial) with the most tumor voxels."""
    areas_per_slice = tumor_mask.sum(axis=(0, 1))
    return int(np.argmax(areas_per_slice))


def largest_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Return the largest connected-component contour, shape (N, 2) as (x, y)."""
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No tumor contour found on the selected slice.")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)  # (x, y)


def visualize(image_2d, mask_2d, contour_xy, fit, slice_idx, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_2d.T, cmap="gray", origin="lower")

    # Tumor outline
    contour_plot = np.vstack([contour_xy, contour_xy[0]])  # close the loop
    ax.plot(contour_plot[:, 0], contour_plot[:, 1], color="cyan", linewidth=1.5, label="Tumor outline")

    # Fitted ellipse: sample points on the ellipse in mm-space, map back to array/display space
    center_mm = fit["center_mm"]
    angle_rad = np.deg2rad(fit["angle_deg"])
    a_mm, b_mm = fit["major_mm"] / 2.0, fit["minor_mm"] / 2.0
    # NOTE: a_mm/b_mm here correspond to whichever fitted axis is major/minor;
    # cv2.fitEllipse's own (ax0, ax1) ordering is recovered indirectly via major/minor endpoints instead.
    theta = np.linspace(0, 2 * np.pi, 200)
    maj = fit["major"]
    minr = fit["minor"]
    maj_vec = np.array([maj[1][0] - maj[0][0], maj[1][1] - maj[0][1]]) / 2.0  # (d0, d1) half-vector
    min_vec = np.array([minr[1][0] - minr[0][0], minr[1][1] - minr[0][1]]) / 2.0
    center_arr = ((maj[0][0] + maj[1][0]) / 2.0, (maj[0][1] + maj[1][1]) / 2.0)
    ellipse_d0 = center_arr[0] + np.outer(np.cos(theta), maj_vec[0]) + np.outer(np.sin(theta), min_vec[0])
    ellipse_d1 = center_arr[1] + np.outer(np.cos(theta), maj_vec[1]) + np.outer(np.sin(theta), min_vec[1])
    ax.plot(ellipse_d1.ravel(), ellipse_d0.ravel(), color="lime", linewidth=1.5, label="Fitted ellipse")

    # Major / minor axis lines
    (m0a, m1a), (m0b, m1b) = fit["major"]
    ax.plot([m1a, m1b], [m0a, m0b], color="red", linewidth=2, label=f"Major axis: {fit['major_mm']:.1f} mm")
    (n0a, n1a), (n0b, n1b) = fit["minor"]
    ax.plot([n1a, n1b], [n0a, n0b], color="yellow", linewidth=2, label=f"Minor axis: {fit['minor_mm']:.1f} mm")

    ax.set_title(f"Axial slice {slice_idx} — kidney tumor size measurement")
    ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    image_nii = nib.load(str(IMAGE_PATH))
    label_nii = nib.load(str(LABEL_PATH))

    image = image_nii.get_fdata()
    label = label_nii.get_fdata()
    voxel_spacing = image_nii.header.get_zooms()  # (mm, mm, mm)

    tumor_mask = label == TUMOR_LABEL
    slice_idx = select_largest_tumor_slice(tumor_mask)

    image_2d = image[:, :, slice_idx]
    mask_2d = tumor_mask[:, :, slice_idx]
    pixel_sizes_2d = (float(voxel_spacing[0]), float(voxel_spacing[1]))  # in-plane spacing (dim0, dim1)

    contour_xy = largest_contour(mask_2d)
    fit = fit_ellipse_real_space(contour_xy, pixel_sizes_2d)

    out_path = OUTPUT_DIR / "tumor_size_measurement.png"
    visualize(image_2d, mask_2d, contour_xy, fit, slice_idx, out_path)

    print("=== Tumor size measurement ===")
    print(f"Selected slice index (axial, axis=2): {slice_idx}")
    print(f"In-plane pixel spacing (mm):           {pixel_sizes_2d}")
    print(f"Major axis length (mm):                {fit['major_mm']:.2f}")
    print(f"Minor axis length (mm):                {fit['minor_mm']:.2f}")
    print(f"Overlay saved to:                      {out_path}")


if __name__ == "__main__":
    main()
