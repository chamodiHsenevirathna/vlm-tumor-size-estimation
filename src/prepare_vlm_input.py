"""
Milestone 5: prepare the single CT image we'll show to the VLM.

Reuses case 0's already-extracted data (data/one_case/train/{image,label}/0.nii.gz,
same case used in Milestones 2-4) and the same slice index (133) already selected
as "largest tumor area" in Milestone 2/3.

Converts the stored CT intensities to HU (stored values are NOT HU; see
ct_preprocessing.py -- the offset is a dataset-specific hypothesis and is
configurable with --intensity-offset), then applies a standard abdominal
soft-tissue HU window (center=40, width=400 -- the usual radiology default for
viewing kidneys/soft tissue) to produce a normal 8-bit grayscale PNG. The export
is validated (ct_preprocessing.validate_export) before it is saved. No tumor mask,
outline, or bounding box is drawn -- the model only ever sees the same kind of
image a radiologist would look at.

Runs locally (no GPU needed); the output PNG is what you'll upload to Colab.
"""

import argparse
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
DATA_DIR = REPO_ROOT / "data" / "one_case" / "train"
IMAGE_PATH = DATA_DIR / "image" / "0.nii.gz"
LABEL_PATH = DATA_DIR / "label" / "0.nii.gz"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vlm_case0"

SLICE_IDX = 133  # same slice selected in Milestone 2/3 for case 0
WINDOW_CENTER = DEFAULT_WINDOW_CENTER  # HU, standard abdominal soft-tissue window
WINDOW_WIDTH = DEFAULT_WINDOW_WIDTH    # HU
TUMOR_LABEL = 4
KIDNEY_LABEL = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--intensity-offset", type=float, default=DEFAULT_INTENSITY_OFFSET,
                        help="stored = HU + offset (default %(default)s; unverified hypothesis)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    image = nib.load(str(IMAGE_PATH)).dataobj  # raw stored values; no float conversion/scaling
    label = np.asanyarray(nib.load(str(LABEL_PATH)).dataobj)

    slice_raw = np.asanyarray(image[:, :, SLICE_IDX])
    windowed = window_ct_slice(slice_raw, args.intensity_offset, WINDOW_CENTER, WINDOW_WIDTH)
    validate_export(
        windowed, slice_raw,
        intensity_offset=args.intensity_offset, center=WINDOW_CENTER, width=WINDOW_WIDTH,
        kidney_mask=label[:, :, SLICE_IDX] == KIDNEY_LABEL,
        tumor_mask=label[:, :, SLICE_IDX] == TUMOR_LABEL,
    )

    # Match display orientation used in earlier milestones' visualizations (transpose).
    windowed_display = windowed.T

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "slice_windowed.png"
    Image.fromarray(windowed_display, mode="L").save(out_path)

    print(f"Slice index used:        {SLICE_IDX}")
    print(f"Intensity offset:        {args.intensity_offset:g} (hypothesis, unverified)")
    print(f"HU window (center/width): {WINDOW_CENTER:g} / {WINDOW_WIDTH:g}")
    print(f"Saved windowed image to: {out_path}")


if __name__ == "__main__":
    main()
