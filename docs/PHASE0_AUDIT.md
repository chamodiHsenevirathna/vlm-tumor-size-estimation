# Phase 0 audit: why the original VLM results are invalid

**Summary.** The first VLM experiment (Milestones 5-7) fed MedGemma images that were essentially blank. A CT
intensity-handling bug saturated every exported slice to white. The reported accuracy numbers, the "spacing makes
estimates worse" observation, and the "repetitive predictions" interpretation therefore say nothing about MedGemma or
about pixel-spacing context. The experiment has not been rerun. Corrected inference is **pending**.

The reference measurements (ellipse fits to ground-truth masks) never used image intensities and are unaffected.

## Evidence

**1. The exported images were blank.** Applying the original formula (window center 40 HU, width 400 HU, applied directly
to the stored values) to the six slices that were sent to the model, plus case 4 (never sent), gives:

| Case | Unique grey levels | Pixels at 255 |
|---|---|---|
| 0, 2, 5 | 1 | 100% |
| 3 | 2 | 100% (1 non-white pixel of 29,241) |
| 4 | 83 | 99% (never exported originally; computed here) |
| 6 | 67 | 99.6% (90 non-white pixels) |
| 8 | 66 | 99.6% (128 non-white pixels) |

Re-running the original formula reproduces the committed PNGs pixel for pixel (cases 0, 2, 6, 8). The committed files
are 222-553 bytes each.

**2. The stored values are not Hounsfield units.** In all 10 locally extracted cases the NIfTI volumes are `uint16`
(values 0-2469, which cannot represent negative HU), `scl_slope` and `scl_inter` are NaN, and the header has no
description or extension. No dataset documentation describing the intensity encoding was found: the local zip has none,
the Hugging Face dataset card and README say nothing about it, and the KiPA22 dataset page returned HTTP 403.

**3. The data are consistent with `stored = HU + 1024`, but this is not proven.**

- Two histogram peaks per volume: fat at about 921 and soft tissue at about 1074 (means over 10 cases; sd 4-5), a
  separation of 140-166 raw units. Fat to soft tissue is about 140-160 HU, so the slope is about 1.
- Implied offsets: about 1021 (fat = -100 HU) and about 1024 (soft tissue = +50 HU).
- Kidney median intensity after subtracting 1024: about +80 to +170 HU, plausible for contrast-enhanced kidney.
- 7 of 10 volumes contain values floored at 0 (consistent with -1024 HU).
- These anchors only constrain the offset to roughly **1000-1040**. The tissue HU values are priors, not measurements.

## Correction

`src/ct_preprocessing.py` converts stored values to HU (`hu = stored - offset`, computed in float64 so `uint16` cannot
wrap) before windowing. The offset is a parameter; **1024 is a documented, dataset-specific hypothesis, not a fact.**
Both preprocessing scripts use this one module, and every export is validated before it is saved.

With the corrected pipeline the five representative cases (0, 2, 4, 6, 8) export 247-256 grey levels, grey standard
deviation 52-63, under 4% saturated pixels, 96-99% of pixels inside the HU window, and a kidney median of +80 to +139 HU
in those slices. A visual check showed recognisable kidney, tumor, fat and bone anatomy, with the ground-truth outlines
sitting on the lesions. (Those images are CT-derived and are not committed.)

## Validation checks that now exist

Content-aware (not just grey-level statistics): fraction of pixels inside the HU window, kidney-ROI median HU range,
tumor-ROI grey range and clipping, and an integrity check that the image equals a fresh windowing of the source slice.
Generic backstop: unique grey levels, standard deviation, entropy, saturation. A wrong offset is flagged by the
window-coverage check alone even with every generic threshold disabled (unit-tested).

## What was NOT verified

- The exact offset (see above). The checks cannot distinguish 1000 from 1024.
- Other cases in the zip (70 image/label pairs; only the first 10 were extracted and examined).
- How Gemma's processor resizes and normalises these small (116-176 px) crops. The checkpoint's preprocessor config is
  gated; the public model card says images are normalised to 896x896 and encoded to 256 tokens.
- Any corrected model behaviour. No inference has been run since the fix.

## Related design limitation (independent of the bug)

Experiment B states the native pixel spacing (about 0.59-0.78 mm per pixel), but the model receives an image resized to
roughly 896x896, where one pixel spans about 0.09-0.12 mm. The stated spacing does not describe the pixels the model
sees. The pilot keeps the original prompts for comparability and records this as a limitation.

## Status of the corrected pilot

A fail-closed five-case smoke-test pipeline (`validation/phase0_pilot5/`) is implemented and unit-tested, including
stub-based dry runs of the notebook. **It has not been run against a model.** It is designed as a smoke test, not an
accuracy study: three scored cases and two exploratory cases are far too few for any accuracy conclusion.
