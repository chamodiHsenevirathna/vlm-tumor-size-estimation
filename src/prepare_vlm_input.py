"""
Milestone 5: prepare the single CT image we'll show to the VLM.

Reuses case 0's already-extracted data (data/one_case/train/{image,label}/0.nii.gz,
same case used in Milestones 2-4) and the same slice index (133) already selected
as "largest tumor area" in Milestone 2/3.

Applies a standard abdominal soft-tissue HU window (center=40, width=400 -- the
usual radiology default for viewing kidneys/soft tissue) to convert the raw
Hounsfield-Unit float slice into a normal 8-bit grayscale PNG. No tumor mask,
outline, or bounding box is drawn -- the model only ever sees the same kind of
image a radiologist would look at.

Runs locally (no GPU needed); the output PNG is what you'll upload to Colab.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "one_case" / "train"
IMAGE_PATH = DATA_DIR / "image" / "0.nii.gz"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vlm_case0"

SLICE_IDX = 133  # same slice selected in Milestone 2/3 for case 0
WINDOW_CENTER = 40   # HU, standard abdominal soft-tissue window
WINDOW_WIDTH = 400   # HU


def apply_hu_window(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip a Hounsfield-Unit slice to [center-width/2, center+width/2] and
    rescale to 8-bit grayscale (0-255)."""
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(slice_hu, low, high)
    scaled = (clipped - low) / (high - low) * 255.0
    return scaled.astype(np.uint8)


def main() -> None:
    image_nii = nib.load(str(IMAGE_PATH))
    image = image_nii.get_fdata()

    slice_hu = image[:, :, SLICE_IDX]
    windowed = apply_hu_window(slice_hu, WINDOW_CENTER, WINDOW_WIDTH)

    # Match display orientation used in earlier milestones' visualizations (transpose).
    windowed_display = windowed.T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "slice_windowed.png"
    Image.fromarray(windowed_display, mode="L").save(out_path)

    print(f"Slice index used:        {SLICE_IDX}")
    print(f"HU window (center/width): {WINDOW_CENTER} / {WINDOW_WIDTH}")
    print(f"Saved windowed image to: {out_path}")


if __name__ == "__main__":
    main()
