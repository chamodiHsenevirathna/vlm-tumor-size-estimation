# Notice: VLM result files in this directory are INVALID

`vlm_batch5_results.csv`, `vlm_batch5_summary.csv` and `vlm_batch5_percase_comparison.csv` were produced by running
MedGemma on CT slice images that were **blank** (saturated white) because of a CT intensity-preprocessing bug. They
are kept unmodified for transparency and **must not be used as evidence** about MedGemma, pixel spacing, or
tumor-size estimation.

`batch_measurements.csv` and `qc_analysis.csv` are reference measurements computed from ground-truth segmentation
masks only. They do not depend on image intensities and are not affected by the bug.

Details and evidence: [`docs/PHASE0_AUDIT.md`](../docs/PHASE0_AUDIT.md).
