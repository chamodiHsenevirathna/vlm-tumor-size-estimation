> **WARNING: INVALID RESULTS. DO NOT CITE.** Everything below, and the charts in this folder, describes how MedGemma answered **blank (saturated-white) input images** produced by a CT intensity-preprocessing bug. The findings say nothing about MedGemma, pixel spacing or tumor-size estimation, and are **withdrawn**. The text is preserved unchanged for transparency. See [`docs/PHASE0_AUDIT.md`](../../../docs/PHASE0_AUDIT.md) and [`../../NOTICE_INVALID_VLM_RESULTS.md`](../../NOTICE_INVALID_VLM_RESULTS.md).

# Milestone 7 — VLM batch analysis findings

Source of truth: `results/vlm_batch5_results.csv` and `results/vlm_batch5_summary.csv` from the completed Milestone 6 batch (5 scored cases + 1 exploratory case, `google/medgemma-1.5-4b-it`, deterministic generation). No inference was rerun to produce this analysis.

## Main quantitative finding

- Average MAE, no spacing (A): **17.96 mm**
- Average MAE, with spacing (B): **25.81 mm**
- Average MRE, no spacing (A): **51.3%**
- Average MRE, with spacing (B): **87.5%**
- Scored cases improved by spacing: 0/5
- Scored cases worsened by spacing: 5/5

**In this pilot, explicitly providing pixel spacing worsened tumor-size estimation accuracy on all 5 scored cases** — both MAE and MRE increased substantially when the prompt included the true pixel spacing.

## Sample size caveat (do not overclaim)

**n=5 scored cases is a small pilot sample.** This result describes what happened in this specific pilot with this specific model, prompt wording, and windowed-image style — it does **not** establish a general conclusion about `google/medgemma-1.5-4b-it`, about MedGemma more broadly, or about vision-language models' ability to use spacing information in general. A larger, more varied batch (different tumor sizes, shapes, and pixel spacings) would be needed before drawing any general conclusion.

## Observed numeric pattern in the raw responses

The per-case predicted values themselves show a striking pattern, described here as an **observation**, not a proven causal mechanism:

- Experiment A produced 1 distinct (major, minor) prediction pair(s) across all 6 cases: [(15.0, 10.0)].
  -> Every Experiment A case received the *identical* predicted size, regardless of the case's actual reference measurement or image content.
- Ratio of Experiment A's prediction to Experiment B's prediction, per case (major_ratio, minor_ratio): [('0', 10.0, 10.0), ('3', 10.0, 10.0), ('5', 10.0, 10.0), ('6', 1.43, 1.82), ('8', 1.3, 0.87), ('2', 10.0, 10.0)].
- 4 of 6 cases show an EXACT 10x ratio between A's and B's predicted major and minor axis values (cases: ['0', '3', '5', '2']).

**Interpretation, stated carefully:** Experiment A's prediction did not vary at all across cases with different images and different true tumor sizes — this is consistent with the model producing a generic/templated guess rather than a measurement actually derived from the image content, at least for Experiment A's prompt wording. When spacing was added (Experiment B), most cases' predictions dropped to exactly 1/10th of Experiment A's values, a suspiciously clean ratio. One plausible (but **unverified**) explanation is that the model may be misinterpreting the stated pixel spacing (~0.6-0.7 mm, i.e. a number starting with a leading zero) as a cue to scale its answer down by a fixed factor, rather than using it as an actual unit-conversion multiplier the way the prompt intends. Two cases (6 and 8) deviate from the exact 10x pattern, which argues against a simple universal rule and toward something less predictable. This is offered as a hypothesis worth investigating further (e.g. with more cases, or prompt rewording), not as an established explanation.

## Exploratory case 2 (excluded from scored statistics)

- Case 2 (qc_flag=ellipse_outside_bbox): A -> (15.0, 10.0) mm, B -> (1.5, 1.0) mm, reference (27.44, 19.36) mm. Shows the same pattern as the scored cases (A constant, B ~10x smaller) — consistent with, but not proof of, the same underlying behavior.

## Per-case comparison table (scored cases)

| case | ref major/minor (mm) | A pred major/minor (mm) | B pred major/minor (mm) | MAE A | MAE B | MRE A | MRE B | spacing_effect |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.33/13.98 | 15.0/10.0 | 1.5/1.0 | 2.65 | 13.90 | 18.3% | 91.8% | worsened |
| 3 | 22.57/21.47 | 15.0/10.0 | 1.5/1.0 | 9.52 | 20.77 | 43.5% | 94.3% | worsened |
| 5 | 28.77/23.39 | 15.0/10.0 | 1.5/1.0 | 13.58 | 24.83 | 52.6% | 95.3% | worsened |
| 6 | 37.23/36.09 | 15.0/10.0 | 10.5/5.5 | 24.16 | 28.66 | 66.0% | 78.3% | worsened |
| 8 | 56.64/48.09 | 15.0/10.0 | 11.5/11.5 | 39.87 | 40.87 | 76.4% | 77.9% | worsened |

## Figures

- `mae_per_case.png` — MAE per scored case, A vs B
- `mre_per_case.png` — MRE per scored case, A vs B
- `ref_vs_pred.png` — reference vs predicted major axis, scored (circles) vs exploratory case 2 (triangles)
- `overall_avg_comparison.png` — average MAE/MRE across the 5 scored cases, A vs B