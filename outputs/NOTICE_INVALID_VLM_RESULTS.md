# Notice: VLM outputs in this directory are INVALID

The raw model responses, `milestone7_findings.md` and the numeric analysis under `vlm_case0/` and `vlm_batch5/` come from
MedGemma runs whose input images were **blank** (saturated white) because of a CT intensity-preprocessing bug. The
conclusions written in `vlm_batch5/analysis/milestone7_findings.md` are **not supported** and must not be cited. The
`analysis/` charts plot those invalid numbers and are retained for transparency only.

**Not published.** The blank exported slice images (`vlm_batch5/case_*_slice.png`, `vlm_batch5/images.zip`,
`vlm_case0/slice_windowed.png`) and the reference-measurement figures (`batch_examples/`, `qc_analysis/`,
`sample_overlay.png`, `tumor_size_measurement.png`) were removed from the published history. The figures show CT slices
derived from KiPA22 (CC BY-NC-4.0), and their redistribution status has **not been confirmed**. The author keeps local
copies. Only `results/batch_measurements.csv` and `results/qc_analysis.csv` (reference and QC measurements from ground-truth masks) are unaffected by the bug; the other files in `results/` are invalid (see `results/NOTICE_INVALID_VLM_RESULTS.md`).

Details and evidence: [`docs/PHASE0_AUDIT.md`](../docs/PHASE0_AUDIT.md). What was removed and why:
[`docs/HISTORY.md`](../docs/HISTORY.md).
