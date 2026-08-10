"""
Milestone 7: analysis and visualization of the Milestone 6 VLM batch results.

Reads the already-completed Milestone 6 outputs (results/vlm_batch5_results.csv,
results/vlm_batch5_summary.csv) exactly as produced by the Colab notebook --
this script does NOT call the model, does not modify those files, and does not
touch any Milestone 1-6 data, prompts, or reference measurements.

Produces:
  - results/vlm_batch5_percase_comparison.csv   (one row per scored case, A vs B side by side)
  - outputs/vlm_batch5/analysis/mae_per_case.png
  - outputs/vlm_batch5/analysis/mre_per_case.png
  - outputs/vlm_batch5/analysis/ref_vs_pred.png
  - outputs/vlm_batch5/analysis/overall_avg_comparison.png
  - outputs/vlm_batch5/analysis/milestone7_findings.md  (short written analysis)

Also inspects the raw predicted values for a systematic numeric pattern
(e.g. a constant/templated answer, or a fixed scale factor between
Experiment A and B) and reports it as an OBSERVED pattern -- not a proven
causal explanation -- in the findings file.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "vlm_batch5" / "analysis"

# Milestone 6's notebook download saved these with a trailing space before the
# extension (Colab file-picker artifact); support both spellings so this
# script works regardless of which one is present, without renaming/altering
# the original files.
def _find(basename: str) -> Path:
    candidates = [RESULTS_DIR / basename, RESULTS_DIR / f"{basename[:-4]} {basename[-4:]}"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find {basename} (checked {candidates})")


RESULTS_CSV = _find("vlm_batch5_results.csv")
SUMMARY_CSV = _find("vlm_batch5_summary.csv")
# batch_metadata.csv (Milestone 6's own preprocessing output) always has
# qc_flag per case; the results CSV from this particular run predates the
# qc_flag column being added to it, so qc_flag is joined in from here instead
# of assumed present on every row of vlm_batch5_results.csv.
METADATA_CSV = REPO_ROOT / "outputs" / "vlm_batch5" / "batch_metadata.csv"

EXP_A = "A_no_spacing"
EXP_B = "B_with_spacing"


def load_qc_flags() -> dict:
    with open(METADATA_CSV, newline="") as f:
        return {row["case_id"]: row["qc_flag"] for row in csv.DictReader(f)}


def load_results() -> list[dict]:
    qc_flags = load_qc_flags()
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["scored"] = r["scored"] == "True"
        if "qc_flag" not in r or not r["qc_flag"]:
            r["qc_flag"] = qc_flags.get(r["case_id"], "unknown")
        for k in ["pixel_spacing_mm", "major_mm_pred", "minor_mm_pred", "major_mm_ref", "minor_mm_ref",
                  "major_abs_error", "minor_abs_error", "major_rel_error", "minor_rel_error",
                  "mae_axes", "mre_axes"]:
            r[k] = float(r[k]) if r[k] not in ("", None) else None
    return rows


def build_percase_comparison(rows: list[dict]) -> list[dict]:
    """One row per scored case: Experiment A and B side by side."""
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["experiment"]] = r

    out = []
    for case_id, exps in sorted(by_case.items(), key=lambda kv: int(kv[0])):
        a, b = exps.get(EXP_A), exps.get(EXP_B)
        if a is None or b is None:
            continue
        out.append({
            "case_id": case_id,
            "scored": a["scored"],
            "qc_flag": a["qc_flag"],
            "major_mm_ref": a["major_mm_ref"],
            "minor_mm_ref": a["minor_mm_ref"],
            "major_mm_pred_A": a["major_mm_pred"],
            "minor_mm_pred_A": a["minor_mm_pred"],
            "major_mm_pred_B": b["major_mm_pred"],
            "minor_mm_pred_B": b["minor_mm_pred"],
            "mae_A": a["mae_axes"],
            "mae_B": b["mae_axes"],
            "mre_A": a["mre_axes"],
            "mre_B": b["mre_axes"],
            "spacing_effect": a["spacing_effect"],
        })
    return out


def save_percase_csv(percase: list[dict]) -> Path:
    out_path = RESULTS_DIR / "vlm_batch5_percase_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(percase[0].keys()))
        writer.writeheader()
        writer.writerows(percase)
    return out_path


def plot_mae_per_case(percase_scored: list[dict], out_path: Path) -> None:
    case_ids = [r["case_id"] for r in percase_scored]
    mae_a = [r["mae_A"] for r in percase_scored]
    mae_b = [r["mae_B"] for r in percase_scored]

    x = np.arange(len(case_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, mae_a, width, label="A: no spacing", color="#4C72B0")
    ax.bar(x + width / 2, mae_b, width, label="B: with spacing", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in case_ids])
    ax.set_ylabel("MAE across axes (mm)")
    ax.set_title("Mean Absolute Error per scored case: A vs B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mre_per_case(percase_scored: list[dict], out_path: Path) -> None:
    case_ids = [r["case_id"] for r in percase_scored]
    mre_a = [r["mre_A"] * 100 for r in percase_scored]
    mre_b = [r["mre_B"] * 100 for r in percase_scored]

    x = np.arange(len(case_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, mre_a, width, label="A: no spacing", color="#4C72B0")
    ax.bar(x + width / 2, mre_b, width, label="B: with spacing", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in case_ids])
    ax.set_ylabel("MRE across axes (%)")
    ax.set_title("Mean Relative Error per scored case: A vs B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ref_vs_pred(percase: list[dict], out_path: Path) -> None:
    """Reference vs predicted major axis, scored cases + exploratory case marked separately."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ref_scored = [r["major_mm_ref"] for r in percase if r["scored"]]
    pred_a_scored = [r["major_mm_pred_A"] for r in percase if r["scored"]]
    pred_b_scored = [r["major_mm_pred_B"] for r in percase if r["scored"]]
    ref_expl = [r["major_mm_ref"] for r in percase if not r["scored"]]
    pred_a_expl = [r["major_mm_pred_A"] for r in percase if not r["scored"]]
    pred_b_expl = [r["major_mm_pred_B"] for r in percase if not r["scored"]]

    max_val = max(ref_scored + ref_expl + pred_a_scored + pred_b_scored + pred_a_expl + pred_b_expl) * 1.1
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray", linewidth=1, label="Perfect prediction")

    ax.scatter(ref_scored, pred_a_scored, color="#4C72B0", marker="o", s=70, label="Scored, A (no spacing)")
    ax.scatter(ref_scored, pred_b_scored, color="#DD8452", marker="o", s=70, label="Scored, B (with spacing)")
    ax.scatter(ref_expl, pred_a_expl, color="#4C72B0", marker="^", s=90, edgecolor="black",
               label="Exploratory case 2, A")
    ax.scatter(ref_expl, pred_b_expl, color="#DD8452", marker="^", s=90, edgecolor="black",
               label="Exploratory case 2, B")

    ax.set_xlabel("Reference major axis (mm)")
    ax.set_ylabel("Predicted major axis (mm)")
    ax.set_title("Reference vs predicted major axis length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overall_avg(summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].bar(["A: no spacing", "B: with spacing"],
                [summary["scored_avg_mae_no_spacing"], summary["scored_avg_mae_with_spacing"]],
                color=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("Average MAE across axes (mm)")
    axes[0].set_title("Average MAE (5 scored cases)")

    axes[1].bar(["A: no spacing", "B: with spacing"],
                [summary["scored_avg_mre_no_spacing"] * 100, summary["scored_avg_mre_with_spacing"] * 100],
                color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("Average MRE across axes (%)")
    axes[1].set_title("Average MRE (5 scored cases)")

    fig.suptitle("Overall comparison: Experiment A vs B (n=5 scored cases)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_summary() -> dict:
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        raw = {row["metric"]: row["value"] for row in reader}
    summary = {}
    for k, v in raw.items():
        try:
            summary[k] = float(v)
        except ValueError:
            summary[k] = v
    return summary


def inspect_numeric_pattern(rows: list[dict]) -> str:
    """Look for a systematic pattern in the predicted values themselves
    (e.g. a constant/templated answer, or a fixed scale factor between A and
    B), and describe it in plain text. This is descriptive only."""
    a_rows = [r for r in rows if r["experiment"] == EXP_A]
    b_rows = [r for r in rows if r["experiment"] == EXP_B]

    a_pairs = {(r["major_mm_pred"], r["minor_mm_pred"]) for r in a_rows}
    b_pairs = [(r["case_id"], r["major_mm_pred"], r["minor_mm_pred"]) for r in b_rows]

    lines = []
    lines.append(f"- Experiment A produced {len(a_pairs)} distinct (major, minor) prediction pair(s) "
                  f"across all {len(a_rows)} cases: {sorted(a_pairs)}.")
    if len(a_pairs) == 1:
        lines.append("  -> Every Experiment A case received the *identical* predicted size, "
                      "regardless of the case's actual reference measurement or image content.")

    ratios = []
    for case_id, maj_b, min_b in b_pairs:
        a_row = next(r for r in a_rows if r["case_id"] == case_id)
        maj_a, min_a = a_row["major_mm_pred"], a_row["minor_mm_pred"]
        if maj_b and min_b:
            ratios.append((case_id, round(maj_a / maj_b, 2), round(min_a / min_b, 2)))
    lines.append(f"- Ratio of Experiment A's prediction to Experiment B's prediction, per case "
                 f"(major_ratio, minor_ratio): {ratios}.")

    exact_10x = [r for r in ratios if r[1] == 10.0 and r[2] == 10.0]
    lines.append(f"- {len(exact_10x)} of {len(ratios)} cases show an EXACT 10x ratio between A's and B's "
                 f"predicted major and minor axis values (cases: {[r[0] for r in exact_10x]}).")

    return "\n".join(lines)


def write_findings_md(percase: list[dict], summary: dict, pattern_text: str, out_path: Path) -> None:
    scored = [r for r in percase if r["scored"]]
    expl = [r for r in percase if not r["scored"]]

    lines = []
    lines.append("# Milestone 7 — VLM batch analysis findings\n")
    lines.append("Source of truth: `results/vlm_batch5_results.csv` and `results/vlm_batch5_summary.csv` "
                  "from the completed Milestone 6 batch (5 scored cases + 1 exploratory case, "
                  "`google/medgemma-1.5-4b-it`, deterministic generation). No inference was rerun to "
                  "produce this analysis.\n")

    lines.append("## Main quantitative finding\n")
    lines.append(f"- Average MAE, no spacing (A): **{summary['scored_avg_mae_no_spacing']:.2f} mm**")
    lines.append(f"- Average MAE, with spacing (B): **{summary['scored_avg_mae_with_spacing']:.2f} mm**")
    lines.append(f"- Average MRE, no spacing (A): **{summary['scored_avg_mre_no_spacing']*100:.1f}%**")
    lines.append(f"- Average MRE, with spacing (B): **{summary['scored_avg_mre_with_spacing']*100:.1f}%**")
    lines.append(f"- Scored cases improved by spacing: {int(summary['scored_cases_improved'])}/5")
    lines.append(f"- Scored cases worsened by spacing: {int(summary['scored_cases_worsened'])}/5")
    lines.append("")
    lines.append("**In this pilot, explicitly providing pixel spacing worsened tumor-size estimation "
                 "accuracy on all 5 scored cases** — both MAE and MRE increased substantially when the "
                 "prompt included the true pixel spacing.\n")

    lines.append("## Sample size caveat (do not overclaim)\n")
    lines.append("**n=5 scored cases is a small pilot sample.** This result describes what happened "
                 "in this specific pilot with this specific model, prompt wording, and windowed-image "
                 "style — it does **not** establish a general conclusion about `google/medgemma-1.5-4b-it`, "
                 "about MedGemma more broadly, or about vision-language models' ability to use spacing "
                 "information in general. A larger, more varied batch (different tumor sizes, shapes, "
                 "and pixel spacings) would be needed before drawing any general conclusion.\n")

    lines.append("## Observed numeric pattern in the raw responses\n")
    lines.append("The per-case predicted values themselves show a striking pattern, described here as an "
                 "**observation**, not a proven causal mechanism:\n")
    lines.append(pattern_text)
    lines.append("")
    lines.append("**Interpretation, stated carefully:** Experiment A's prediction did not vary at all "
                 "across cases with different images and different true tumor sizes — this is consistent "
                 "with the model producing a generic/templated guess rather than a measurement actually "
                 "derived from the image content, at least for Experiment A's prompt wording. When "
                 "spacing was added (Experiment B), most cases' predictions dropped to exactly 1/10th of "
                 "Experiment A's values, a suspiciously clean ratio. One plausible (but **unverified**) "
                 "explanation is that the model may be misinterpreting the stated pixel spacing "
                 "(~0.6-0.7 mm, i.e. a number starting with a leading zero) as a cue to scale its answer "
                 "down by a fixed factor, rather than using it as an actual unit-conversion multiplier "
                 "the way the prompt intends. Two cases (6 and 8) deviate from the exact 10x pattern, "
                 "which argues against a simple universal rule and toward something less predictable. "
                 "This is offered as a hypothesis worth investigating further (e.g. with more cases, or "
                 "prompt rewording), not as an established explanation.\n")

    lines.append("## Exploratory case 2 (excluded from scored statistics)\n")
    for r in expl:
        lines.append(f"- Case {r['case_id']} (qc_flag={r['qc_flag']}): "
                     f"A -> ({r['major_mm_pred_A']}, {r['minor_mm_pred_A']}) mm, "
                     f"B -> ({r['major_mm_pred_B']}, {r['minor_mm_pred_B']}) mm, "
                     f"reference ({r['major_mm_ref']}, {r['minor_mm_ref']}) mm. "
                     f"Shows the same pattern as the scored cases (A constant, B ~10x smaller) — "
                     f"consistent with, but not proof of, the same underlying behavior.")
    lines.append("")

    lines.append("## Per-case comparison table (scored cases)\n")
    lines.append("| case | ref major/minor (mm) | A pred major/minor (mm) | B pred major/minor (mm) | MAE A | MAE B | MRE A | MRE B | spacing_effect |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in scored:
        lines.append(f"| {r['case_id']} | {r['major_mm_ref']}/{r['minor_mm_ref']} | "
                     f"{r['major_mm_pred_A']}/{r['minor_mm_pred_A']} | "
                     f"{r['major_mm_pred_B']}/{r['minor_mm_pred_B']} | "
                     f"{r['mae_A']:.2f} | {r['mae_B']:.2f} | "
                     f"{r['mre_A']*100:.1f}% | {r['mre_B']*100:.1f}% | {r['spacing_effect']} |")
    lines.append("")

    lines.append("## Figures\n")
    lines.append("- `mae_per_case.png` — MAE per scored case, A vs B")
    lines.append("- `mre_per_case.png` — MRE per scored case, A vs B")
    lines.append("- `ref_vs_pred.png` — reference vs predicted major axis, scored (circles) vs exploratory case 2 (triangles)")
    lines.append("- `overall_avg_comparison.png` — average MAE/MRE across the 5 scored cases, A vs B")

    out_path.write_text("\n".join(lines))


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_results()
    summary = load_summary()
    percase = build_percase_comparison(rows)
    percase_scored = [r for r in percase if r["scored"]]

    percase_csv_path = save_percase_csv(percase)

    plot_mae_per_case(percase_scored, ANALYSIS_DIR / "mae_per_case.png")
    plot_mre_per_case(percase_scored, ANALYSIS_DIR / "mre_per_case.png")
    plot_ref_vs_pred(percase, ANALYSIS_DIR / "ref_vs_pred.png")
    plot_overall_avg(summary, ANALYSIS_DIR / "overall_avg_comparison.png")

    pattern_text = inspect_numeric_pattern(rows)
    findings_path = ANALYSIS_DIR / "milestone7_findings.md"
    write_findings_md(percase, summary, pattern_text, findings_path)

    print("=== Milestone 7 analysis complete ===")
    print(f"Per-case comparison CSV: {percase_csv_path}")
    print(f"Figures written to: {ANALYSIS_DIR}")
    print(f"Findings written to: {findings_path}")
    print()
    print("--- Observed numeric pattern ---")
    print(pattern_text)


if __name__ == "__main__":
    main()
