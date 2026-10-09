# Notice: VLM outputs in this directory are INVALID

Everything under `vlm_case0/` and `vlm_batch5/` (the exported slice images, `images.zip`, raw model responses, and the
`analysis/` figures and `milestone7_findings.md`) comes from MedGemma runs whose input images were **blank**
(saturated white) because of a CT intensity-preprocessing bug. The files are preserved unmodified for transparency.
The conclusions written in `vlm_batch5/analysis/milestone7_findings.md` are **not supported** and must not be cited.

The reference-measurement figures (`batch_examples/`, `qc_analysis/`, `sample_overlay.png`,
`tumor_size_measurement.png`) come from ground-truth masks and are not affected by the bug. They show CT slices
derived from KiPA22 (CC BY-NC-4.0); their redistribution status has not been confirmed (see the README's licensing
section).

Details and evidence: [`docs/PHASE0_AUDIT.md`](../docs/PHASE0_AUDIT.md).
