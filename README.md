# VLM Tumor Size Estimation

A research prototype that asks whether an open medical vision-language model (`google/medgemma-1.5-4b-it`) can
estimate kidney-tumor size from a CT slice, and whether telling it the image's pixel spacing helps.

> **Status: engineering foundation complete; corrected VLM inference is PENDING.**
> The first VLM experiment (Milestones 5-7) used **blank, invalid input images** because of a CT intensity-preprocessing
> bug. **Its results cannot support its original conclusions** and are retained only for transparency. The bug is fixed
> and tested, and a corrected five-case smoke test is implemented, but **no corrected model run has been performed**.
> This is not a clinical tool. See [What went wrong](#what-went-wrong-and-what-is-invalid) and
> [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md).

## Contents

1. [Problem statement](#problem-statement) · 2. [Research questions and objectives](#research-questions-and-objectives)
3. [What went wrong and what is invalid](#what-went-wrong-and-what-is-invalid) · 4. [Status at a glance](#status-at-a-glance)
5. [Architecture and methodology](#architecture-and-methodology) · 6. [Implemented features and verified tests](#implemented-features-and-verified-tests)
7. [Reference measurements (valid)](#reference-measurements-valid) · 8. [Development phases](#development-phases)
9. [Limitations, assumptions and future work](#limitations-assumptions-and-future-work)
10. [Setup and reproduction](#setup-and-reproduction) · 11. [Repository structure](#repository-structure)
12. [Licensing, data and model terms](#licensing-data-and-model-terms) · 13. [Intended use](#intended-use-and-boundaries)
14. [AI assistance and acknowledgements](#ai-assistance-and-acknowledgements)

## Problem statement

Tumor size (for example longest diameter) is a standard clinical measurement. A vision-language model reading a scan
sees only pixels; it has no inherent sense of physical scale unless that is supplied. Pixel spacing (millimetres per
pixel) is the missing conversion factor between "pixels" and "millimetres". Once a tumor is segmented, size is a solved
deterministic problem (fit an ellipse, apply the voxel spacing), and that is how this project produces its reference
values. The open question is the harder one: **can an off-the-shelf medical VLM infer a quantitative measurement from the
raw image alone, without a mask, and does a stated scale help?** The project tests that deliberately and carefully; it
does not assume a VLM is the right tool for measurement.

## Research questions and objectives

- **RQ1.** Does stating the physical pixel spacing improve a VLM's zero-shot tumor-size estimate compared with no
  spacing information? *(Open. The first attempt was invalid; see below.)*
- **RQ2.** Are the model's numeric answers grounded in the image, or generic/templated? *(Open.)*

Objectives: build a reproducible pipeline from public data to a scored comparison; derive trustworthy reference
measurements from ground-truth masks; make the image-preparation step verifiable so a silent failure like the one below
cannot recur; run any model experiment with complete raw logging and honest, descriptive reporting.

## What went wrong and what is invalid

The VLM input images were produced by windowing the NIfTI intensities as if they were Hounsfield units. They are not:
the KiPA22 volumes are `uint16` with no scaling metadata (values about 0-2500), so the window clipped almost every pixel
to white. In the six slices that were sent to the model, cases 0, 2 and 5 were a single flat white value, case 3 had one
non-white pixel out of 29,241, and cases 6 and 8 were 99.6% white (90 and 128 non-white pixels).

**Consequence.** The earlier results (average error figures, "spacing increased error on 5 of 5 cases", the identical
`15 x 10 mm` outputs, and the "10x" pattern) describe how the model answered a blank image. They say nothing about
MedGemma's measurement ability or about pixel-spacing context. Those conclusions are withdrawn.

**What is preserved, unmodified:** the original notebooks, raw model responses, result CSVs, figures and
`milestone7_findings.md`, each directory carrying a notice (`results/` and `outputs/NOTICE_INVALID_VLM_RESULTS.md`).
The reference measurements and QC results do not depend on image intensities and are **not** affected.

**The fix** is a shared, offset-aware conversion with export validation (`src/ct_preprocessing.py`). The offset
**1024 (stored = HU + 1024) is an unverified, dataset-specific hypothesis**: it fits the data well, but the data only
constrain it to roughly 1000-1040 and no documentation confirms it. It is therefore a configurable parameter.

## Status at a glance

| Area | Status |
|---|---|
| Reference tumor measurement from masks, QC flagging (10 cases) | Done, valid |
| Original VLM experiment (Milestones 5-7) | **Invalid** (blank images); preserved for transparency |
| Root-cause analysis of the invalid images | Done ([`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md)) |
| Corrected CT preprocessing + validation + tests | Done; offset 1024 remains an unverified hypothesis |
| Five-case pilot pipeline (bundle, verification, notebook, logging) | Implemented and unit-tested; **never run against a model** |
| Corrected VLM inference and any conclusion from it | **Pending** |
| Packaging, CI, larger evaluation | Not started |

## Architecture and methodology

```
KiPA22 NIfTI (CT + labels)
   |
   +--> label mask --> largest-tumor axial slice --> ellipse fit in mm --> reference major/minor (mm) + QC flag
   |                    (src/measure_tumor.py, batch_measure.py, analyze_qc.py)             [valid]
   |
   +--> CT slice --> stored values -> HU (offset) --> HU window (40/400) --> 8-bit PNG
                      (src/ct_preprocessing.py)  --> export validation (fails loudly if blank/invalid)
                                   |
                                   +--> pilot bundle (hash-pinned) --> Colab notebook --> MedGemma, prompts A/B
                                        (validation/phase0_pilot5/)                       [implemented, not yet run]
```

**Reference measurements.** For each case the kidney-tumor mask (label 4) is isolated and the axial slice with the
largest tumor area is chosen. An ellipse is fit to that slice's tumor contour **in millimetre space** (contour points are
scaled by the voxel spacing before fitting; scaling by different per-axis factors is not a similarity transform and can
otherwise change axis lengths). The approach follows the one described for MedVision's ellipse-fitting pipeline; this
repository is an independent implementation of it (MedVision source files are not included). A MedVision-style QC check (`pass_basic` vs `ellipse_outside_bbox`, a
0.9x/1.1x bounding-box sanity test, plus multiple-component and minimum-size rules) flags fits that may be unreliable.

**VLM input image.** Stored intensities are converted to HU with a configurable offset, windowed with an abdominal
soft-tissue window (center 40, width 400), and exported as 8-bit greyscale. No mask, outline or bounding box is ever
drawn on a model input. Each export is validated (below) before it is saved.

**Experiments (original design, kept for the pilot).** A: the model sees the slice and is asked for the tumor's major and
minor axis in mm, with no scale. B: the same, with the native pixel spacing stated in the prompt. Answers are requested as
`<answer>{"major_mm": X, "minor_mm": Y}</answer>`, parsed and validated (finite, positive). Generation is deterministic
(`do_sample=False`, `num_beams=1`, `max_new_tokens=300`). Prompts are identical to Milestone 6 (`m6-prompts-v1`).
Planned metrics (not computed on valid data yet): MAE and relative error per axis, reported separately for scored and
exploratory cases.

**Pilot design.** Five cases: scored 0, 6, 8 (`pass_basic`) and exploratory 2, 4 (`ellipse_outside_bbox`). Scored and
exploratory cases are logged, saved and summarised separately and are never pooled. It is a smoke test of the pipeline
and of model behaviour, not an accuracy study.

## Implemented features and verified tests

| Component | What it does |
|---|---|
| [`src/ct_preprocessing.py`](src/ct_preprocessing.py) | Offset-aware HU conversion (float64, no `uint16` wraparound), windowing, and `validate_export` with content-aware checks (HU-window coverage, kidney/tumor ROI HU and grey ranges, integrity against the source) plus a generic grey-level backstop |
| [`src/prepare_vlm_input.py`](src/prepare_vlm_input.py), [`src/prepare_vlm_batch.py`](src/prepare_vlm_batch.py) | Use the shared module, validate every export, `--intensity-offset` / `--output-dir` options |
| [`src/validate_phase0_preprocessing.py`](src/validate_phase0_preprocessing.py) | Regenerates five representative cases with old-vs-corrected comparison, QC overlays (never model input), metrics |
| [`validation/phase0_pilot5/pilot_lib.py`](validation/phase0_pilot5/pilot_lib.py) | SHA-256 bundle verification with a stale-image blocklist; run-metadata schema; single-use inference gate; write-ahead per-call logging; interruption and early-stop accounting; failure-preserving parsing; group-separated summaries |
| [`validation/phase0_pilot5/build_pilot_bundle.py`](validation/phase0_pilot5/build_pilot_bundle.py) | Builds the hash-pinned bundle/ZIP and the Colab notebook (needs the dataset locally) |
| [`validation/phase0_pilot5/phase0_pilot5.ipynb`](validation/phase0_pilot5/phase0_pilot5.ipynb) | Pilot notebook: fail-closed verification, metadata before inference, full raw logging. Contains no outputs and no credentials (token is read from a Colab secret) |

**Tests (run in a clean clone of the published commit, without the dataset):**

| Suite | Command | Result |
|---|---|---|
| Preprocessing and validation | `python -m unittest discover -s tests` | 25 tests, all pass |
| Pilot tooling | `python -m unittest discover -s validation/phase0_pilot5/tests` | 87 tests: 85 pass, 2 skipped (they need the locally built, CT-derived bundle) |

Covered: window formula and boundaries, input types and dtypes, `uint16` wraparound, rejection of wrong offsets and
blank or near-constant images, stale-image rejection (even when a manifest is regenerated to match), gate failures
(missing, invalid or mismatched metadata; existing logs; planted lock files), write-ahead logging, `KeyboardInterrupt` and
early-stop accounting, rerun refusal, parse-failure preservation, and scored/exploratory separation. The pilot's GPU
cells were exercised only with stub objects, **never with a real model**.

**Not verified:** the intensity offset itself; any behaviour of the real MedGemma processor on these images; anything
about model accuracy.

## Reference measurements (valid)

Computed from ground-truth masks only (`results/batch_measurements.csv`); axes in mm on the largest-tumor axial slice.

| Case | Spacing (mm) | Slice | Major | Minor | QC flag |
|---|---|---|---|---|---|
| 0 | 0.5859 | 133 | 16.33 | 13.98 | pass_basic |
| 1 | 0.6523 | 144 | 26.72 | 25.13 | pass_basic |
| 2 | 0.7793 | 47 | 27.44 | 19.36 | ellipse_outside_bbox |
| 3 | 0.6816 | 33 | 22.57 | 21.47 | pass_basic |
| 4 | 0.5898 | 136 | 47.94 | 30.67 | ellipse_outside_bbox |
| 5 | 0.5859 | 127 | 28.77 | 23.39 | pass_basic |
| 6 | 0.7148 | 39 | 37.23 | 36.09 | pass_basic |
| 7 | 0.6172 | 148 | 32.42 | 30.88 | ellipse_outside_bbox |
| 8 | 0.5859 | 133 | 56.64 | 48.09 | pass_basic |
| 9 | 0.6836 | 60 | 47.91 | 45.27 | ellipse_outside_bbox |

These are 2D ellipse-fit proxies, not clinician measurements. Four of ten cases fail the bounding-box QC rule; whether
that rule is too strict for this data has not been investigated.

## Development phases

| Phase | Scope | Status |
|---|---|---|
| Milestones 1-4 | Data loading, reference measurement, batch measurement, QC diagnosis | Done |
| Milestones 5-7 | Original VLM runs and analysis | **Invalid input images; withdrawn** |
| Phase 0 | Audit, root cause, corrected preprocessing, validation tooling, tests, pilot pipeline | Done (pilot not run) |
| Phase 0b | Run the five-case corrected smoke test; record offset provenance if it can be found | **Pending** |
| Phase 1 | Reproducible packaging (pinned dependencies, CLI, CI) | Not started |
| Phase 2 | Larger evaluation: more cases, prompt variants, a second model | Not started |
| Phase 3 | Baselines (for example with the mask provided) and a written report | Not started |

## Limitations, assumptions and future work

- **The offset 1024 is unverified** (data constrain it only to about 1000-1040). If dataset documentation is found, update
  the default and rerun.
- **No valid VLM result exists yet.** Nothing here supports any claim about MedGemma.
- **Effective-spacing mismatch.** Experiment B states the native pixel spacing (about 0.59-0.78 mm/px), but the model
  receives an image resized to about 896x896 (public model card), where a pixel spans roughly 0.09-0.12 mm. The stated scale
  does not describe the pixels the model sees. Prompts were kept unchanged for comparability, not because this is correct.
- The checkpoint's preprocessor configuration is gated and unverified; the pilot records it at run time.
- Crops are small (116-176 px) and tightly cut around the kidney region; the field of view is about 80-105 mm.
- The local dataset zip holds 70 image/label pairs; the earlier README said 49. Only the first 10 cases were extracted.
- Single organ, dataset and model; one prompt wording; one window; zero-shot only; no fine-tuning.
- Reference values are single-slice 2D proxies; four of ten fail the QC rule.
- The pilot has 3 scored cases and 2 exploratory ones: it can show whether outputs vary and parse, not how accurate they are.
- GPU decoding is not guaranteed bit-reproducible; `transformers` is installed unpinned in Colab, so versions are recorded in
  the run metadata instead.

**Future work:** run the pilot; find or establish intensity provenance; rerun the original questions on valid images with
more cases; test prompts that state the spacing of the resized image; add mask-provided and pixel-measurement baselines;
package with pinned dependencies and CI; consider the product framing below only after valid evidence exists.

**Design position (not a result).** Where a mask is available, deterministic geometry is more auditable than a VLM
estimate. A VLM is more plausibly a supporting layer (explanation, flagging disagreement for human review) than the source
of truth for measurements. Production use would need validated ground truth, clinician review and regulatory consideration,
tracked with accuracy, parse-failure and repeated-answer rates, disagreement with deterministic measurements, drift, and
full audit logs of prompts, model versions and raw outputs.

## Setup and reproduction

Developed and tested on macOS with Python 3.14.7; other versions are untested.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tests (no dataset needed)
python -m unittest discover -s tests
python -m unittest discover -s validation/phase0_pilot5/tests
```

**Data.** The scripts need KiPA22 (about 400 MB, `YongchengYAO/KiPA22` on Hugging Face, CC BY-NC-4.0). `src/load_sample.py`
downloads it into `data/` (git-ignored). Check the terms before use or redistribution; see the licensing section.

**Reference measurements** (run from `src/`; paths are relative to the repo):

```bash
cd src
python load_sample.py        # download + inspect one case
python measure_tumor.py      # reference ellipse for that case
python batch_measure.py      # 10 cases -> results/batch_measurements.csv
python analyze_qc.py         # QC diagnosis -> results/qc_analysis.csv
```

**Corrected preprocessing and validation** (writes to git-ignored `validation/phase0/`):

```bash
python src/validate_phase0_preprocessing.py                      # default offset 1024 (unverified); needs the cases extracted by batch_measure.py
python src/prepare_vlm_input.py --intensity-offset 1024 --output-dir /tmp/check
```

**Pilot (not yet run).** `python validation/phase0_pilot5/build_pilot_bundle.py` builds the bundle, ZIP and a notebook with the
hashes of *your* build pinned inside it (the committed notebook pins the author's local build, which is not published).
Follow [`validation/phase0_pilot5/README_pilot.md`](validation/phase0_pilot5/README_pilot.md). Running it uploads CT-derived
images to Google Colab and needs a Hugging Face token with accepted MedGemma terms; read the data-handling notes first.

## Repository structure

```
src/                        Preprocessing, reference measurement, QC, validation (run locally, no GPU)
tests/                      Unit tests for preprocessing and validation
validation/phase0_pilot5/   Pilot library, bundle builder, notebook, tests, pilot README
notebooks/                  ORIGINAL Milestone 5-6 Colab notebooks (their results are invalid; see above)
results/ , outputs/         Original outputs, preserved unmodified; VLM-derived files are INVALID (see notices)
docs/PHASE0_AUDIT.md        Evidence for the invalid images, the fix, and what is unverified
data/                       Local data (git-ignored)
```

## Licensing, data and model terms

- **Code:** [MIT](LICENSE) (c) 2026 Chamodi Senevirathna. This license covers this repository's code and documentation only.
  **It grants no rights to the dataset or to the model.** Dependencies are permissively licensed (for example nibabel MIT,
  numpy BSD-3-Clause, OpenCV Apache-2.0, huggingface_hub Apache-2.0, Pillow MIT-CMU, matplotlib PSF-style).
- **Dataset:** KiPA22, redistributed via MedVision on Hugging Face under **CC BY-NC-4.0** (non-commercial, attribution). You must
  download it yourself and comply with its terms. This repository does not include the dataset.
- **CT-derived images are intentionally not distributed with new work.** The CT-derived validation images, pilot bundle/ZIP
  and run folders are git-ignored. Earlier commits already contain some CT-derived figures under `outputs/` (overlays and QC
  figures); whether redistributing those is permitted under the dataset's terms has **not been confirmed**. Do not assume reuse
  rights for them.
- **Model:** `google/medgemma-1.5-4b-it` is gated on Hugging Face; you must accept its terms yourself. They are separate from
  this repository's license.
- **MedVision** (CC BY 4.0) is credited for the ellipse-fitting and QC methodology this work follows; MedVision source files
  are not included. Its license also requires downstream use to comply with each source dataset's terms.

## Intended use and boundaries

A research and portfolio prototype. It is **not** clinically validated, not for diagnosis, staging or treatment decisions, not a
replacement for a radiologist's or a deterministic algorithm's measurement, and not suitable for autonomous reporting.

## AI assistance and acknowledgements

This repository was developed with AI coding assistance (Claude Code). The Phase 0 audit, correction, tests and pilot tooling
were produced in AI-assisted sessions; outputs and claims here should be checked against the code and tests, which is why
unverified items are listed explicitly. Thanks to the KiPA22 challenge organisers, the MedVision project (Yongcheng Yao et al.)
for the redistributed data and methodology, and Google for the open MedGemma models.
