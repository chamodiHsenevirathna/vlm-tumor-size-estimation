# VLM Tumor Size Estimation

A small, MedVision-inspired pilot study testing whether an open vision-language model (VLM) can estimate kidney tumor size from a CT slice, and whether telling it the image's pixel spacing helps.

## Overview

This project builds a minimal, reproducible pipeline that:
1. Loads a real tumor CT case (KiPA22) with its ground-truth segmentation mask.
2. Derives a MedVision-style reference tumor measurement (major/minor axis, in mm) directly from that mask.
3. Prompts an open medical VLM (`google/medgemma-1.5-4b-it`) to estimate the same measurement from the CT image alone — with and without being told the image's real-world pixel spacing.
4. Compares the model's predictions against the reference measurements.

It is not a reproduction of the MedVision paper's full benchmark, and not a fine-tuning or training project — it's a small, self-contained experiment built end-to-end from public data to a scored (if inconclusive) result.

## Research question

**Does giving a vision-language model the physical pixel spacing of a CT slice improve its zero-shot estimate of tumor size, compared to giving it no spacing information at all?**

## Why this matters

Tumor size (e.g. longest diameter) is a standard clinical measurement used to track disease and treatment response. A VLM reading a scan only sees pixels — it has no inherent sense of scale unless that information is supplied. Pixel spacing (how many millimeters one pixel represents) is exactly the missing conversion factor between "pixels" and "millimeters." If explicitly providing it doesn't help — or actively hurts — that's a meaningful, checkable finding about how (or whether) these models actually ground numeric answers in the image, rather than assuming spacing is trivially useful context.

## Dataset: KiPA22

[KiPA22](https://kipa22.grand-challenge.org) is a kidney CT dataset with segmentation masks for four structures: kidney, kidney tumor, renal artery, renal vein. Redistributed by the MedVision project on Hugging Face ([`YongchengYAO/KiPA22`](https://huggingface.co/datasets/YongchengYAO/KiPA22), CC BY-NC-4.0) as a single ~400MB zip of 49 training cases — by far the smallest of MedVision's tumor-size-annotated datasets, which made it the practical choice for a single-machine pilot (alternatives range from ~3GB to ~96GB).

## Model: `google/medgemma-1.5-4b-it`

[MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) is Google's open, medically-tuned multimodal model (4B parameters, Gemma 3-based), trained on medical images including radiology. Chosen over larger general-purpose VLMs for being open-weight, small enough to run on a free-tier GPU, and domain-relevant — a reasonable candidate for "can an off-the-shelf medical VLM do this out of the box," with no fine-tuning involved.

## Methodology

### Reference measurements

For each case, the kidney-tumor mask (label 4) is isolated, and the axial slice with the largest tumor area is selected. An ellipse is fit to that slice's tumor contour **in real-world (mm) space** — contour points are scaled by the voxel spacing *before* fitting, not after, since scaling by different factors per axis is not a similarity transform and can otherwise change the fitted axis lengths and orientation. This mirrors the approach used in MedVision's own `medvision_ds.__fit_ellipses` pipeline. Major/minor axis lengths in millimeters are read directly off the fitted ellipse. A lightweight, MedVision-inspired QC check (`pass_basic` vs `ellipse_outside_bbox`, based on a 0.9×/1.1× bounding-box sanity check) flags fits that may be unreliable; only `pass_basic` cases are used for scoring.

### VLM input image

CT slices are windowed with a standard abdominal soft-tissue HU window (center=40, width=400) before being shown to the model — the same normalization radiologists use to view kidneys/soft tissue on a raw Hounsfield-Unit scan. No tumor mask, outline, or bounding box is drawn on the image; the model sees only what a radiologist would see on the raw slice, so it has to both locate and size the tumor unaided.

### Experiment A — no pixel spacing

The model is shown the windowed slice and asked to estimate the tumor's major and minor axis in millimeters, with no scale information given.

### Experiment B — explicit pixel spacing

Identical prompt and image, except the case's true pixel spacing (e.g. "0.5859 mm × 0.5859 mm per pixel") is stated in the prompt.

Both experiments require a structured response: `<answer>{"major_mm": X, "minor_mm": Y}</answer>`, parsed and validated (finite, > 0) before scoring. Generation is deterministic (`do_sample=False`, `num_beams=1`, fixed `max_new_tokens`), and the model is loaded once per Colab session and reused across all cases and both experiments.

### Scored vs. exploratory cases

5 cases (IDs 0, 3, 5, 6, 8) — all `pass_basic` QC, spanning a ~3.5× range of tumor sizes (16–57mm) — form the **scored** set used for every averaged statistic. One additional case (ID 2, flagged `ellipse_outside_bbox`) is run through the identical pipeline as an **exploratory** case, reported separately and excluded from all averages, since its own reference measurement is less trustworthy.

### Evaluation metrics

- **MAE** (mean absolute error, mm): average of `|predicted − reference|` across the major and minor axes.
- **MRE** (mean relative error, %): average of `|predicted − reference| / reference` across both axes — normalizes error by tumor size, so a 5mm error means more on a 16mm tumor than a 57mm one.

## Main results (n=5 scored cases)

| | Experiment A (no spacing) | Experiment B (with spacing) |
|---|---|---|
| Average MAE | 17.96 mm | 25.81 mm |
| Average MRE | 51.3% | 87.5% |
| Cases improved by spacing | — | 0 / 5 |
| Cases worsened by spacing | — | 5 / 5 |

**In this pilot, explicitly providing pixel spacing increased error on all 5 scored cases** — both MAE and MRE roughly doubled going from Experiment A to Experiment B.

## Important observation: repetitive predictions

Looking at the raw predicted values (not just the error), a striking pattern emerges:

- **Experiment A returned the exact same prediction, `15.0mm × 10.0mm`, for all 6 cases** (all 5 scored + the exploratory case) — regardless of the case's true tumor size, which ranged from 16mm to 57mm.
- **Experiment B returned `1.5mm × 1.0mm` — exactly 1/10th of Experiment A's answer — for 4 of the 6 cases.** The remaining two cases (IDs 6 and 8) deviated from that exact ratio, giving `10.5×5.5` and `11.5×11.5` respectively.

### Interpretation (stated carefully)

This pattern is **descriptive, not a proven causal explanation**. What it does suggest: Experiment A's identical output across cases with visibly different images and different true tumor sizes is consistent with the model producing a generic, templated-looking guess rather than a measurement actually grounded in that image's content — at least under this exact zero-shot prompt and windowing setup. The suspiciously clean 10× relationship in most Experiment B cases raises the possibility that the model responded to the stated spacing value by scaling its answer down by a fixed factor rather than performing a genuine unit conversion — but the two cases that don't follow this exact ratio argue against a single deterministic rule, and no mechanism has been verified (e.g. by inspecting attention, trying reworded prompts, or testing on more cases).

**We do not claim that pixel spacing is generally harmful to VLM tumor-size estimation.** The honest summary is narrower: in this small pilot, explicit spacing increased error across all five scored cases, and the largely repetitive predictions across both experiments suggest weak image-grounded quantitative measurement under the specific zero-shot prompting setup tested here — not a general result about MedGemma, spacing information, or VLMs.

## Limitations

- **n=5 scored cases** — a pilot sample, not a statistically powered study. No claim here generalizes beyond this specific setup.
- Single model (`medgemma-1.5-4b-it`), single prompt phrasing, single windowing/normalization choice, single dataset (KiPA22, kidney tumors only) — all of these are unexplored variables that could change the outcome.
- Reference measurements come from an ellipse fit to a single 2D slice, not a clinician's read — a reasonable proxy, but not clinical ground truth.
- The repetitive-prediction pattern is an observation from 6 cases; it has not been mechanistically explained or verified with larger-scale or ablation testing.
- Zero-shot only — no fine-tuning, no chain-of-thought reasoning was requested or relied upon (only the structured final answer was scored, though full raw responses were saved for transparency).

## Reproducibility

Requires Python 3.11+ (project developed on 3.14.6) for all local steps; VLM inference requires a Colab GPU session (free T4 tier is sufficient) since it isn't run locally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pipeline, in order:
1. `src/load_sample.py` — downloads KiPA22's `train.zip` (~400MB, cached), extracts one case, prints its fields, saves a sanity-check overlay.
2. `src/measure_tumor.py` — fits the reference ellipse for that one case, reports major/minor axis in mm.
3. `src/batch_measure.py` — repeats the measurement across a 10-case batch, applies the QC flag, writes `results/batch_measurements.csv`.
4. `src/analyze_qc.py` — diagnoses the QC-flagged cases (`results/qc_analysis.csv`, `outputs/qc_analysis/`).
5. `src/prepare_vlm_input.py` / `src/prepare_vlm_batch.py` — window and export the CT slice(s) shown to the VLM, plus a metadata/zip bundle for Colab upload.
6. `notebooks/milestone5_vlm_case0.ipynb` / `notebooks/milestone6_vlm_batch5.ipynb` — run in Google Colab (free T4 GPU) to perform the actual VLM inference; requires a Hugging Face token supplied via Colab secrets (never hardcoded) and acceptance of MedGemma's usage terms.
7. `src/analyze_vlm_batch5.py` — analyzes the completed VLM batch results, produces the comparison table and figures below.

## Repository structure

```
src/                    Analysis/preprocessing scripts (run locally, no GPU needed)
notebooks/              Colab notebooks (VLM inference; needs a GPU)
data/                   Downloaded/extracted case data (gitignored — not committed)
results/                CSV outputs: reference measurements, QC analysis, VLM predictions
outputs/                Figures, windowed images, raw model responses
  ├── batch_examples/       Example reference-measurement visualizations
  ├── qc_analysis/          QC diagnostic visualizations
  ├── vlm_case0/             Milestone 5 single-case VLM smoke test
  └── vlm_batch5/            Milestone 6 batch VLM results + Milestone 7 analysis
      └── analysis/              Final comparison figures and findings (below)
```

## Key figures

- `outputs/vlm_batch5/analysis/overall_avg_comparison.png` — average MAE/MRE, Experiment A vs B
- `outputs/vlm_batch5/analysis/mae_per_case.png` — per-case MAE, A vs B
- `outputs/vlm_batch5/analysis/mre_per_case.png` — per-case MRE, A vs B
- `outputs/vlm_batch5/analysis/ref_vs_pred.png` — reference vs. predicted tumor size, all cases

![Overall MAE/MRE comparison](outputs/vlm_batch5/analysis/overall_avg_comparison.png)
![Reference vs predicted major axis](outputs/vlm_batch5/analysis/ref_vs_pred.png)

## Conclusion

In this small pilot, an off-the-shelf medical VLM's zero-shot tumor-size estimates did not benefit from being told the CT slice's pixel spacing — error increased on every scored case — and the largely repetitive predictions across both experiments suggest the model's answers were only weakly grounded in each image's actual content under this prompting setup. The pipeline built here (reference measurement → windowed VLM input → structured prediction → scored comparison) is reusable for a larger, better-powered follow-up before drawing any general conclusion.
