# History notes: what was changed before publication

The repository history was rewritten once, on 2026-10-10, before the repository was made public. This page says exactly
what changed, so the rewrite is not hidden. The nine development commits were kept in order, with their original author
names, author and committer dates, time zones, messages and co-author trailers. No commit was added, merged, reordered or
back-dated; commits after the table below are ordinary new commits with their real dates.

## What was removed

19 files were removed from every commit that contained them. They were all first added in the third commit. All remain on the author's machine; none is published.

| Path | Old blob (SHA-1, first 12) |
|---|---|
| `outputs/batch_examples/case_0_measurement.png` | `07fca316957e` |
| `outputs/batch_examples/case_1_measurement.png` | `4fd20677dfed` |
| `outputs/batch_examples/case_2_measurement.png` | `f795ccdf94c5` |
| `outputs/qc_analysis/case_0_qc.png` | `78ab73c1765b` |
| `outputs/qc_analysis/case_2_qc.png` | `4b30b133eaab` |
| `outputs/qc_analysis/case_4_qc.png` | `fca5941c6671` |
| `outputs/qc_analysis/case_7_qc.png` | `40db2ebbbb4c` |
| `outputs/qc_analysis/case_8_qc.png` | `93ac8f0aa6a3` |
| `outputs/qc_analysis/case_9_qc.png` | `a67161039a58` |
| `outputs/sample_overlay.png` | `f30f5675b21a` |
| `outputs/tumor_size_measurement.png` | `07fca316957e` |
| `outputs/vlm_batch5/case_0_slice.png` | `1dac635ad26d` |
| `outputs/vlm_batch5/case_2_slice.png` | `be5ee41974ac` |
| `outputs/vlm_batch5/case_3_slice.png` | `3d82026760c0` |
| `outputs/vlm_batch5/case_5_slice.png` | `c481fa3c3d8b` |
| `outputs/vlm_batch5/case_6_slice.png` | `9dd3a3da0e8d` |
| `outputs/vlm_batch5/case_8_slice.png` | `11d5d89fa56d` |
| `outputs/vlm_batch5/images.zip` | `158928c9cc93` |
| `outputs/vlm_case0/slice_windowed.png` | `1dac635ad26d` |

Why: the first 11 are figures that show CT slices derived from KiPA22 (CC BY-NC-4.0), and **redistribution rights have not
been confirmed**. The other 8 are the blank (all-white) exported slice images, their ZIP, and the single-case slice. They are
not CT-informative (they are the output of the intensity bug described in [PHASE0_AUDIT.md](PHASE0_AUDIT.md)), but they were
derived from the same data, so they were withheld as well until rights are confirmed. Their absence means the blank-image
evidence can no longer be opened as files; the audit documents the formula and the measured numbers so the check can be repeated
from the source data. The four numeric charts in `outputs/vlm_batch5/analysis/` are plots of numbers, not CT, and were kept.

## What was sanitized

- `results/qc_analysis.csv`, last column `viz_path`: absolute local home-directory paths (user name and project folder) were reduced
  to repo-relative paths (`outputs/qc_analysis/case_N_qc.png`) in every commit that had them. All other columns are unchanged.
  `src/analyze_qc.py` now writes relative paths.
- Commit metadata: the author's personal email address was replaced by the GitHub noreply address in every commit.
  Names are unchanged.

## What this changes

- **All commit hashes changed** (the trees changed). The table maps old to new. Hashes quoted before 2026-10-10 do not resolve here.
- **The signature on the initial commit was lost.** It was made in the GitHub web editor (committer `GitHub`) and signed by GitHub; any rewrite
  invalidates it. Its author, dates and message are unchanged.
- The first four commits are dated 2026-08-10 and the next five 2026-10-10; this is the real gap between the original study and its later audit and correction, and the dates are unchanged.
- Early commits therefore do not contain the figures that the original README and notices mention; the notices were updated.

## Commit map

| # | Old | New | Date (original) | Subject |
|---|---|---|---|---|
| 1 | `6f9cda8` | `5d8e1c8` | 2026-08-10 13:35:00 +0530 | Initial commit |
| 2 | `1d12aac` | `02f9c7a` | 2026-08-10 15:20:35 +0530 | Set up project structure |
| 3 | `5320cc5` | `b42144a` | 2026-08-10 21:04:15 +0530 | Complete MedVision-inspired VLM tumor size estimation study |
| 4 | `b7a0ea1` | `ab74b59` | 2026-08-10 21:13:17 +0530 | Add AI product framing and governance sections |
| 5 | `7a91c02` | `f5fbb50` | 2026-10-10 00:04:04 +0530 | Harden .gitignore against secrets and CT-derived material |
| 6 | `e75adbc` | `0c897c4` | 2026-10-10 00:04:04 +0530 | Fix CT intensity preprocessing and add validation tooling |
| 7 | `01dee03` | `9952fa9` | 2026-10-10 00:04:04 +0530 | Add fail-closed five-case pilot tooling (not yet run) |
| 8 | `5ba7457` | `1f222ae` | 2026-10-10 00:06:58 +0530 | Rewrite README honestly; document invalid VLM results and Phase 0 audit |
| 9 | `448c7d0` | `ede872c` | 2026-10-10 00:06:59 +0530 | Add MIT license for the code |

## Verification performed on the rewritten history

None of the 19 old blobs exists in the object store; `git log --all -- <path>` is empty for each path; no PNG/ZIP files other than the four
charts exist in any commit; no absolute home-directory path or personal email address appears in any commit's content or metadata;
`git fsck --strict` is clean; each rewritten commit's file list equals the original minus the 19 paths, and only
`results/qc_analysis.csv` differs in content; the unit tests pass at the rewritten tip.
