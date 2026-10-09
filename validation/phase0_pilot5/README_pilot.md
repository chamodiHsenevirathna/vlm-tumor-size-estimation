# Phase 0 five-case pilot (smoke test)

**This pilot does NOT validate tumor-measurement accuracy.** It checks that the pipeline now runs on valid,
non-blank CT images and records exactly how MedGemma behaves, with full raw logging. Three scored cases
(0, 6, 8) and two exploratory cases (2, 4) is far too small for any accuracy or spacing-effect conclusion.

Nothing here replaces, edits or supersedes the existing Milestone 5/6/7 results, notebooks, README or conclusions.

## Known limitations (also recorded in `manifest.json` and `run_metadata.json`)

- **Intensity offset 1024 is an UNVERIFIED hypothesis** (stored = HU + 1024). No dataset documentation confirms it;
  the data only constrains it to roughly 1000-1040.
- **Effective-spacing mismatch.** Experiment B states the native pixel spacing (e.g. 0.5859 mm/px), but the model
  receives an image resized to about 896x896 (public model card), where one pixel spans roughly 0.09-0.12 mm. The stated
  spacing does not describe the pixels the model sees. Prompts are kept identical to Milestone 6 (`m6-prompts-v1`) for
  comparability only.
- The checkpoint's preprocessor config is gated; the actual config is dumped into `run_metadata.json` at run time.
- Cases 2 and 4 carry the `ellipse_outside_bbox` QC flag; cases 3 and 5 from Milestone 6 are not included.
- References are 2D ellipse fits to ground-truth masks, not clinician measurements.

## Layout

```
build_pilot_bundle.py   builds bundle/, pilot5_images.zip and phase0_pilot5.ipynb (hashes pinned in the notebook)
pilot_lib.py            verification, prompts, parsing, metadata schema, logging (no GPU deps; unit-tested)
bundle/                 manifest.json, pilot_lib.py, images/case_{id}_slice_off1024.png  (zip contents)
pilot5_images.zip       the only file uploaded to Colab
phase0_pilot5.ipynb     the pilot notebook
tests/test_pilot_lib.py unit tests
runs/<run_id>/          (create after a run) downloaded run output goes here
```

## How the stale-image failure is prevented

1. New, distinct filenames (`case_N_slice_off1024.png`) in a new zip name, so old files cannot be picked up by name.
2. The notebook pins SHA-256 hashes of `manifest.json` and `pilot_lib.py`; any other bundle aborts before anything runs.
3. The manifest hashes every image and contains a blocklist of the known stale (blank, offset-0) exports; a blocklisted
   image is rejected even if someone regenerates the manifest to match.
4. Independent generic image checks (grey levels, spread, entropy, saturation) run again on the uploaded files.
5. Exact file inventory: any extra, missing or renamed file aborts.
6. The notebook refuses to start if `pilot5_bundle/` already exists, re-verifies right before inference, and refuses
   to run without a GPU.

## How incomplete metadata, reruns and lost attempts are prevented

- **Metadata gate.** Step 6 validates the metadata, persists it atomically, and only then exposes `run_meta`. Step 7
  calls `assert_ready_for_inference`, which blocks (touching nothing) unless the persisted `run_metadata.json` exists,
  is valid, and equals the in-memory metadata, the run id and folder name match, and the folder holds nothing else.
  The approval is single-use, in-process and bound to the metadata hash; a hand-made lock file is not an approval.
- **No second pass.** An existing raw log, run log, results, summary or lock blocks inference. A rerun must go through
  Step 6 again and gets a NEW run folder with a new unique run id. Records are never appended to an old run.
- **Write-ahead logging.** Every planned call is written (flushed + fsynced) to `raw_responses.jsonl` as
  `attempt_started` BEFORE the image is loaded or the model is called, then closed with `attempt_finished`
  (`generated` | `exception` | `interrupted` | `aborted`). A `plan` record lists all 10 planned calls up front.
- **Early stop / interruption.** If three consecutive calls fail, or you interrupt the cell, every remaining planned
  call is written as `attempt_skipped` with its reason, `run_metadata.json` is finalised (`stopped_early`,
  `interrupted`, `aborted`), and an interruption is re-raised. Step 8 reconciles the log against its plan and reports
  any attempt that has no terminal record (for example after a hard kill) as `unfinished` / `never_started`.
- **Every record carries the unique `run_id`;** a log with several run ids, several plans or unplanned calls is rejected.

## Scored vs exploratory

Scored (0, 6, 8) and exploratory (2, 4) rows are written to `results_scored.csv` and `results_exploratory.csv` as well as
the combined `results.csv` (which has a `group` column), and `summary.json` reports each group separately. Nothing is
pooled and no accuracy aggregate is computed. Any future scored calculation must start from `select_scored_rows` /
`require_scored_only`. Exploratory cases must never enter a scored calculation.

## Steps to run (needs your explicit go-ahead: this uploads five CT-derived images to Google Colab)

1. Local sanity check (no GPU, no upload), from the repo root:
   `.venv/bin/python -m unittest discover -s validation/phase0_pilot5/tests` and
   `.venv/bin/python -m unittest discover -s tests` - both must say OK.
2. Open `phase0_pilot5.ipynb` in Google Colab. Runtime -> T4 GPU.
3. Colab secret `HF_TOKEN` (read token, notebook access on); MedGemma terms accepted on your Hugging Face account.
4. Run Step 1-2, then **Runtime -> Restart session**, then continue from Step 3. Do not re-run the install cell.
5. Step 4: upload exactly `pilot5_images.zip`. Expect `BUNDLE VERIFIED: 3cc87e8a4cc6b584`, and look at the five
   displayed images (real CT slices, no outlines). Stop if anything looks blank.
6. Step 5: bf16 cell. Only if it hits CUDA out-of-memory, run the 4-bit fallback cell instead (recorded in metadata).
7. Steps 6-8: run in order (metadata is validated and persisted before inference; 10 planned calls; reconcile, parse
   and summary). If Step 7 is interrupted, still run Steps 8-9 to reconcile and download the partial log. To repeat
   a run, re-run Step 6 (new run folder) and then Step 7. Never re-run Step 7 alone: it will be refused.
8. Step 9: download `pilot5_run_<run_id>.zip` and unzip it into `validation/phase0_pilot5/runs/<run_id>/`.
   Never into `outputs/` or `results/`.

## Run output

`raw_responses.jsonl` (plan, then per call: `attempt_started`, `attempt_finished` or `attempt_skipped`, then
`run_finished`; each with the run id, prompt, raw text, text with special tokens, generated token ids, finish reason /
`hit_max_new_tokens`, token counts, timing, exception + traceback), `run.log`, `results.csv` + `results_scored.csv` +
`results_exploratory.csv` (one row per PLANNED call, including skipped ones), `parse_failures.jsonl` (every non-ok row),
`summary.json` (descriptive counts per group, no MAE/MRE averages), `run_metadata.json` (final status and counts),
`inference.lock`.
