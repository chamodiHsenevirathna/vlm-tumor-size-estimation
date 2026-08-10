"""
Milestone 1: download and inspect ONE case from the KiPA22 kidney-tumor CT dataset.

KiPA22 is redistributed by MedVision as a single ~400 MB `train.zip` on Hugging Face
(https://huggingface.co/datasets/YongchengYAO/KiPA22). We download that one zip, but only
extract and load ONE case's image + label volumes from it.

Labels (per the KiPA22 dataset card):
    1 = renal vein, 2 = kidney, 3 = renal artery, 4 = kidney tumor
"""

import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "YongchengYAO/KiPA22"
ZIP_FILENAME = "train.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

LABELS_MAP = {1: "renal vein", 2: "kidney", 3: "renal artery", 4: "kidney tumor"}
TUMOR_LABEL = 4


def download_zip() -> Path:
    """Download train.zip (~400 MB) from the KiPA22 Hugging Face repo, cached locally."""
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ZIP_FILENAME,
            local_dir=DATA_DIR,
        )
    )


def extract_one_case(zip_path: Path) -> tuple[Path, Path]:
    """Extract exactly one case's image + label file from the zip, skipping the rest."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        image_names = sorted(n for n in names if "image" in n.lower() and n.lower().endswith((".nii", ".nii.gz")))
        label_names = sorted(n for n in names if "label" in n.lower() and n.lower().endswith((".nii", ".nii.gz")))

        if not image_names or not label_names:
            raise RuntimeError(
                f"Could not find image/label NIfTI files by name pattern. "
                f"First 20 entries in zip: {names[:20]}"
            )

        # Pick the first case, matched by shared case id prefix.
        image_name = image_names[0]
        case_id = Path(image_name).name.split("_image")[0]
        matching_labels = [n for n in label_names if case_id in n]
        label_name = matching_labels[0] if matching_labels else label_names[0]

        extract_dir = DATA_DIR / "one_case"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(image_name, extract_dir)
        zf.extract(label_name, extract_dir)

        return extract_dir / image_name, extract_dir / label_name


def load_case(image_path: Path, label_path: Path):
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    return image_nii, label_nii


def summarize_case(image_nii, label_nii) -> None:
    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    voxel_spacing_mm = image_nii.header.get_zooms()

    print("=== Sample summary ===")
    print(f"Image shape (voxels):   {image.shape}")
    print(f"Voxel spacing (mm):     {voxel_spacing_mm}")
    print(f"Label volume shape:     {label.shape}")
    print(f"Labels present:         {sorted(int(v) for v in np.unique(label) if v != 0)}")
    for label_id, name in LABELS_MAP.items():
        count = int(np.sum(label == label_id))
        print(f"  - {label_id} ({name}): {count} voxels")

    tumor_voxels = int(np.sum(label == TUMOR_LABEL))
    if tumor_voxels == 0:
        print("\nNo tumor voxels found in this case; pick a different one for the overlay.")
    else:
        print(f"\nTumor label name:      '{LABELS_MAP[TUMOR_LABEL]}'")
        print(f"Tumor voxel count:      {tumor_voxels}")


def save_overlay(image_nii, label_nii, out_path: Path) -> None:
    image = image_nii.get_fdata()
    label = label_nii.get_fdata()

    tumor_mask = label == TUMOR_LABEL
    slice_tumor_counts = tumor_mask.sum(axis=(0, 1))
    if slice_tumor_counts.max() == 0:
        print("Skipping overlay: no tumor voxels in this case.")
        return

    slice_idx = int(np.argmax(slice_tumor_counts))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[:, :, slice_idx].T, cmap="gray", origin="lower")
    ax.imshow(
        np.ma.masked_where(~tumor_mask[:, :, slice_idx].T, tumor_mask[:, :, slice_idx].T),
        cmap="autumn",
        alpha=0.5,
        origin="lower",
    )
    ax.set_title(f"Axial slice {slice_idx} — kidney tumor overlay")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay to {out_path}")


def main() -> None:
    zip_path = download_zip()
    image_path, label_path = extract_one_case(zip_path)
    image_nii, label_nii = load_case(image_path, label_path)
    summarize_case(image_nii, label_nii)
    save_overlay(image_nii, label_nii, OUTPUT_DIR / "sample_overlay.png")


if __name__ == "__main__":
    main()
