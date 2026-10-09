# VLM Tumor Size Estimation

`Python` · `NumPy` · `nibabel` · `OpenCV` · `Pillow` · `unittest` · `Hugging Face / MedGemma (Colab)` · `Research prototype`

Can an open medical vision-language model estimate kidney-tumor size from a CT slice, and does telling it the pixel
spacing help? This repository holds a reference-measurement pipeline built from ground-truth masks, a fail-closed
preprocessing and experiment harness, and an honest account of a first experiment that turned out to be invalid.

> **Status: corrected VLM inference is still PENDING.** The first VLM experiment (Milestones 5-7) used **blank, invalid
> input images** because CT intensities were handled incorrectly, so **its findings are invalid and withdrawn**. The bug is
> fixed and tested, and a five-case smoke test is implemented, but **no model has been run on the corrected images**.
> This is a research prototype, not a clinical tool. Evidence: [`docs/PHASE0_AUDIT.md`](docs/PHASE0_AUDIT.md).

**Contents:** [Overview](#overview) · [Status](#current-status) · [Capabilities](#key-capabilities) ·
[What went wrong](#what-went-wrong-and-how-it-was-identified) · [Architecture](#architecture-and-workflow) ·
[Engineering decisions](#engineering-decisions-and-risk-management) · [Reference measurements](#reference-measurements-and-their-limitations) ·
[Testing](#testing-and-validation) · [Roadmap](#roadmap) · [Setup](#setup-and-reproduction) ·
[Data, licensing and limitations](#data-model-licensing-limitations-and-intended-use)

## Overview

Tumor size (for example longest diameter) is a standard clinical measurement. A vision-language model sees only pixels, so
physical scale must be supplied; pixel spacing (mm per pixel) is the missing conversion factor. Once a tumor is segmented,
size is a solved geometry problem, which is how this project produces its reference values. The open question is harder:

- **RQ1.** Does stating the pixel spacing improve a VLM's zero-shot tumor-size estimate versus giving no scale?
- **RQ2.** Are the model's numeric answers grounded in the image, or generic and templated?

Both are **open**. The project's objectives are to derive trustworthy reference measurements from masks, make the
image-preparation step verifiable so a silent failure cannot recur, and run any model experiment with complete raw logging
and descriptive, non-overstated reporting.

**What this project demonstrates** (each point is backed by files in this repository):

- *Failure analysis and data-quality assessment:* diagnosing blank model inputs from the data's own statistics ([audit](docs/PHASE0_AUDIT.md)).
- *Validation thinking:* content-aware export checks, not only generic thresholds ([`src/ct_preprocessing.py`](src/ct_preprocessing.py)).
- *Reproducible, traceable experiment design:* hash-pinned inputs, recorded run metadata, preserved raw responses ([`validation/phase0_pilot5/`](validation/phase0_pilot5/)).
- *Honest reporting:* invalid results are withdrawn and labelled, not deleted or quietly re-spun.

This project applies business analysis principles to AI research through explicit problem definition, data-quality
requirements, validation criteria, risk management, traceability, and evidence-based decision-making. Concretely, the research
questions above are the problem definition; "model inputs must be non-blank and plausible HU" became enforced validation rules;
risks (stale inputs, incomplete metadata, lost partial runs, pooled scored and exploratory results) each map to a control in the
[decisions table](#engineering-decisions-and-risk-management); and the status table tags each area as verified, implemented, hypothesis or planned. It is a research prototype:
it has no production deployment, no user requirements gathered from clinicians, and no clinical validation.

## Current status

Tags used below: **[Verified]** covered by automated tests or direct inspection · **[Implemented]** built, not exercised on a
real model · **[Hypothesis]** plausible but unconfirmed · **[Planned]** not started.

| Area | Status |
|---|---|
| Reference measurement from masks, QC flagging, 10 cases | **[Verified]** valid; 2D proxies, see limitations |
| Original VLM experiment (Milestones 5-7) | **Invalid** (blank input images); preserved, with notices |
| Root cause of the invalid images | **[Verified]** by inspection of data and outputs |
| Corrected CT preprocessing and export validation | **[Verified]** by 25 unit tests |
| CT intensity offset `stored = HU + 1024` | **[Hypothesis]** fits the data; not confirmed by any documentation |
| Five-case pilot pipeline (bundle, gate, logging, notebook) | **[Implemented]** unit-tested and dry-run with stubs; **never run against a model** |
| Corrected VLM inference, and any conclusion from it | **[Planned]** pending |

## Key capabilities

- **Reference measurement** ([`src/measure_tumor.py`](src/measure_tumor.py), [`src/batch_measure.py`](src/batch_measure.py)): largest-tumor axial slice, ellipse fit in millimetre space, MedVision-style QC flag.
- **Offset-aware preprocessing** ([`src/ct_preprocessing.py`](src/ct_preprocessing.py)): stored values to HU (float64, no `uint16` wraparound), HU windowing, and `validate_export`, which raises `InvalidImageError` on blank or implausible exports.
- **Updated export scripts** ([`prepare_vlm_input.py`](src/prepare_vlm_input.py), [`prepare_vlm_batch.py`](src/prepare_vlm_batch.py)) validate every image and take `--intensity-offset` / `--output-dir`.
- **Phase 0 validation tool** ([`src/validate_phase0_preprocessing.py`](src/validate_phase0_preprocessing.py)): old-versus-corrected comparison and QC overlays (overlays are never model input).
- **Pilot harness** ([`validation/phase0_pilot5/`](validation/phase0_pilot5/)): SHA-256 bundle verification with a stale-image blocklist, run-metadata schema, single-use inference gate, write-ahead call logging, failure-preserving parsing, and group-separated summaries. Details in its [README](validation/phase0_pilot5/README_pilot.md).

## What went wrong and how it was identified

The VLM input images were made by windowing the NIfTI intensities as if they were Hounsfield units. They are not: the
KiPA22 volumes are `uint16` with no scaling metadata (values about 0-2500), so the window clipped almost everything to white.

**How it was found.** An audit of the repository noticed that the exported PNGs were only 222-553 bytes. Inspecting them
showed that cases 0, 2 and 5 were a single flat white value, case 3 had one non-white pixel out of 29,241, and cases 6 and 8
were 99.6% white. Re-running the original formula reproduced the originally committed images pixel for pixel (those images have since been removed from the published history; see [`docs/HISTORY.md`](docs/HISTORY.md)). The volume's own
statistics then pointed to the cause: two tissue peaks (fat near 921, soft tissue near 1074) about 150 raw units apart,
consistent with a unit-slope intensity offset of roughly 1021-1024.

**Consequence.** The earlier error figures, the "spacing increased error on 5 of 5 cases" observation, the identical
`15 x 10 mm` outputs and the "10x" pattern describe how a model answered a blank image. They say nothing about MedGemma or
about pixel-spacing context. **Those conclusions are withdrawn.** The offset **1024 is an unverified hypothesis**: the data
constrain it to roughly 1000-1040, and the dataset documentation consulted does not state it. It is therefore configurable.

## Architecture and workflow

```
KiPA22 NIfTI (CT + labels)
  |
  +-> label mask -> largest-tumor slice -> ellipse fit in mm -> reference major/minor + QC flag   [valid]
  |
  +-> CT slice -> stored values -> HU (offset) -> HU window 40/400 -> 8-bit image
                     |                                  \-> export validation (fails closed)
                     +-> hash-pinned bundle -> Colab notebook (verify -> gate -> write-ahead log) -> MedGemma, prompts A/B
                                                  [Implemented, not yet run]
```

- **Reference values:** the tumor mask contour is scaled to millimetres *before* fitting, because anisotropic scaling is not a similarity transform. The method follows the approach described for MedVision; this is an independent implementation.
- **Model input:** no mask, outline or box is ever drawn on an image sent to the model.
- **Experiments:** A gives no scale; B states the native pixel spacing. Output format `<answer>{"major_mm": X, "minor_mm": Y}</answer>`, greedy decoding, `max_new_tokens=300`. Prompts are unchanged from Milestone 6 (`m6-prompts-v1`).
- **Pilot:** scored cases 0, 6, 8 (`pass_basic`) and exploratory cases 2, 4 (`ellipse_outside_bbox`). It is a **smoke test, not an accuracy study**: it can show whether outputs parse and vary, not how accurate they are.

## Engineering decisions and risk management

| Decision | Rationale | Trade-off |
|---|---|---|
| Reject invalid exports (`InvalidImageError`) with content-aware checks (HU-window coverage, kidney/tumor ROI HU, integrity against the source) plus a generic backstop | The original failure was silent; grey-level statistics alone could be gamed or misleading. A wrong offset is caught by window coverage even with all generic thresholds disabled (tested) | Thresholds were tuned on 10 local cases and could reject unusual but valid data |
| Offset is a parameter, recorded as `UNVERIFIED` in every manifest and run record | No provenance exists; hiding the assumption would repeat the original mistake | Results will depend on a hypothesis (about +/-24 HU) |
| Preserve the audit trail: original outputs untouched, plus notices and an audit document | Transparency about what failed; nothing is quietly rewritten | Invalid text and CSV artifacts remain in the repo and history, so readers must follow the notices; the blank input images and CT-derived figures were removed from the published history (see `docs/HISTORY.md`) |
| Separate scored and exploratory cases (separate files and summaries; `require_scored_only` guard; no pooled statistic) | Exploratory cases have a QC-flagged reference and must not leak into scored numbers | Only 3 scored cases |
| Hash-pinned bundle with a stale-image blocklist; verification repeated before inference | Makes it hard to feed old blank images to the model, even if a manifest is regenerated | The notebook must be rebuilt per machine (it pins the author's local build) |
| Single-use inference gate: metadata must be valid, persisted and identical to memory; a run folder cannot be reused | Prevents running with incomplete metadata or appending a second pass to an old run | A repeat run needs a new run folder |
| Write-ahead logging of every planned call, plus interrupted, aborted and skipped calls, token ids and raw text | Partial runs stay fully accounted for; `KeyboardInterrupt` is recorded and re-raised | More complex runner (unit-tested, stub dry-run only) |
| Keep Milestone 6 prompts although the stated spacing no longer matches the resized image | Comparability with the original design | Known to be imperfect (see limitations) |
| Hermetic tests on a synthetic bundle; CT-derived artifacts are git-ignored | Tests run without the dataset and without publishing CT-derived images | Two tests skip without the locally built bundle |

## Reference measurements and their limitations

From ground-truth masks only ([`results/batch_measurements.csv`](results/batch_measurements.csv)): largest-tumor axial slice,
axes in mm. **These are 2D ellipse-fit proxies, not clinical measurements.**

| Case | Spacing (mm) | Major | Minor | QC | | Case | Spacing (mm) | Major | Minor | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5859 | 16.33 | 13.98 | pass | | 5 | 0.5859 | 28.77 | 23.39 | pass |
| 1 | 0.6523 | 26.72 | 25.13 | pass | | 6 | 0.7148 | 37.23 | 36.09 | pass |
| 2 | 0.7793 | 27.44 | 19.36 | flagged | | 7 | 0.6172 | 32.42 | 30.88 | flagged |
| 3 | 0.6816 | 22.57 | 21.47 | pass | | 8 | 0.5859 | 56.64 | 48.09 | pass |
| 4 | 0.5898 | 47.94 | 30.67 | flagged | | 9 | 0.6836 | 47.91 | 45.27 | flagged |

"flagged" is `ellipse_outside_bbox`. Four of ten cases are flagged; whether the rule is too strict for this data has not been investigated.

## Testing and validation

Run in a clean clone of the published commit, with no dataset and no local bundle:

| Suite | Command | Result |
|---|---|---|
| Preprocessing and validation | `python -m unittest discover -s tests` | 25 tests, all pass |
| Pilot harness | `python -m unittest discover -s validation/phase0_pilot5/tests` | 87 tests: 85 pass, 2 skipped (need the local, CT-derived bundle); all 87 pass where the bundle exists |

**Covered:** window formula and boundaries; dtypes and `uint16` wraparound; rejection of wrong offsets and blank or near-constant
images; stale-image rejection, including when a manifest is regenerated to match; gate failures (missing, invalid or mismatched
metadata, existing logs, planted lock files); write-ahead logging and fsync; interruption and early-stop accounting; hard-kill
detection; rerun refusal; parse-failure preservation; scored/exploratory separation.

**Not verified:** the intensity offset itself; the notebook's real model, processor and GPU cells (exercised only with stubs);
how Gemma's processor resizes these small crops; any model behaviour or accuracy.

## Roadmap

| Step | Scope | Status |
|---|---|---|
| Milestones 1-4 | Data loading, reference measurement, batch measurement, QC diagnosis | Done |
| Milestones 5-7 | Original VLM runs and analysis | **Invalid; withdrawn** |
| Phase 0 | Audit, corrected preprocessing, validation, tests, pilot harness | Done (pilot not run) |
| Next | Run the five-case smoke test; look for offset provenance; review `ellipse_outside_bbox` strictness | Pending |
| Later | Pinned dependencies, CLI and CI; more cases; prompts that describe the resized image; mask-provided and pixel-measurement baselines; written report | Not started |

## Setup and reproduction

Developed on macOS with Python 3.14.7; other versions are untested.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests                              # no dataset needed
python -m unittest discover -s validation/phase0_pilot5/tests
```

**Data.** The scripts need KiPA22 (about 400 MB, `YongchengYAO/KiPA22` on Hugging Face, CC BY-NC-4.0), downloaded by
`src/load_sample.py` into the git-ignored `data/`. Check the terms first.

```bash
cd src
python load_sample.py && python measure_tumor.py                  # one case
python batch_measure.py && python analyze_qc.py                   # 10 cases -> results/
cd ..
python src/validate_phase0_preprocessing.py                       # corrected exports -> validation/phase0/ (git-ignored)
```

**Pilot (not yet run).** `python validation/phase0_pilot5/build_pilot_bundle.py` builds a bundle and a notebook pinned to
*your* build. Follow the [pilot README](validation/phase0_pilot5/README_pilot.md). Running it uploads CT-derived images to
Google Colab and needs a Hugging Face token with accepted MedGemma terms, so read its data-handling notes first.

```
src/                        Preprocessing, reference measurement, QC, validation (local, no GPU)
tests/                      Preprocessing and validation tests
validation/phase0_pilot5/   Pilot library, bundle builder, notebook, tests, pilot README
notebooks/                  ORIGINAL Milestone 5-6 notebooks (their results are invalid)
results/ , outputs/         Original outputs; VLM-derived files are INVALID (see NOTICE files). CT-derived figures and blank slice images are not published
docs/PHASE0_AUDIT.md        Evidence, the fix, and what remains unverified
```

## Data, model, licensing, limitations and intended use

**Limitations and assumptions**

- No valid VLM result exists. Nothing here supports a claim about MedGemma, pixel spacing or tumor-size estimation.
- **Effective-spacing mismatch:** Experiment B states the native spacing (about 0.59-0.78 mm/px), but the model receives an image
  resized to about 896x896 (public model card), where a pixel spans roughly 0.09-0.12 mm.
- The checkpoint's preprocessor configuration is gated and unverified; the pilot records it at run time.
- Crops are small (116-176 px, field of view about 80-105 mm). The local zip holds 70 image/label pairs (an earlier README said
  49); only the first 10 were examined.
- One organ, dataset, model, prompt wording and window; zero-shot only; GPU decoding is not guaranteed bit-reproducible.
- **Design position, not a result:** where a mask exists, deterministic geometry is more auditable than a VLM estimate; a VLM is
  more plausibly a supporting layer than the source of truth.

**Licensing and terms**

- **Code:** [MIT](LICENSE), covering this repository's code and documentation only. **It grants no rights to the dataset or the model.**
  Dependencies are permissively licensed.
- **Dataset:** KiPA22, redistributed through MedVision on Hugging Face under **CC BY-NC-4.0**. It is not included here. You must
  download it yourself and comply with its terms.
- **CT-derived images:** new CT-derived validation images, pilot bundles and run folders are git-ignored. The CT-derived figures and
  the blank exported slice images (and their ZIP) that were part of the original experiment were **removed from the published
  history** because redistribution rights have **not been confirmed**; the author keeps local copies. Do not assume reuse rights.
  See [`docs/HISTORY.md`](docs/HISTORY.md).
- **Model:** `google/medgemma-1.5-4b-it` is gated; accept its terms yourself. They are separate from this repository's license.
- **MedVision** (CC BY 4.0) is credited for the methodology followed; its source files are not included.

**Intended use.** A research and portfolio prototype: not clinically validated, not for diagnosis, staging or treatment
decisions, and not a replacement for a radiologist's or a deterministic algorithm's measurement.
