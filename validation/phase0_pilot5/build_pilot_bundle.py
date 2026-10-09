"""
Build the isolated Phase 0 five-case pilot bundle. Writes ONLY inside validation/phase0_pilot5/:

  bundle/                 manifest.json, pilot_lib.py, images/case_{id}_slice_off1024.png
  pilot5_images.zip       deterministic zip of bundle/ (this is what gets uploaded to Colab)
  phase0_pilot5.ipynb     pilot notebook with the manifest and pilot_lib hashes PINNED inside it

Reads (never writes) data/batch_cases/, results/batch_measurements.csv, and stale_blocklist.json (committed SHA-256s of the
blank original exports). If the original exports still exist locally under outputs/ they are re-hashed and must agree with
the JSON; the build aborts otherwise. No inference, no network, no uploads.
Every image is validated with the shared ct_preprocessing checks (content-aware + generic) before it
is written; any failure aborts the build.
"""

import ast
import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import pilot_lib as pl  # noqa: E402
from ct_preprocessing import assess_export, window_ct_slice  # noqa: E402

BUNDLE_DIR = HERE / "bundle"
ZIP_PATH = HERE / "pilot5_images.zip"
NOTEBOOK_PATH = HERE / "phase0_pilot5.ipynb"
EXTRACT_DIR = REPO / "data" / "batch_cases"
BATCH_CSV = REPO / "results" / "batch_measurements.csv"
STALE_FILES = [REPO / "outputs" / "vlm_batch5" / f"case_{c}_slice.png" for c in ("0", "2", "3", "5", "6", "8")] + [
    REPO / "outputs" / "vlm_case0" / "slice_windowed.png"]
STALE_JSON = HERE / "stale_blocklist.json"
MIN_STALE_HASHES = 5
KIDNEY, TUMOR = 2, 4
RESIZE_PX = 896  # per the public MedGemma model card; the checkpoint's own preprocessor config was not accessible



def load_stale_blocklist(json_path: Path = STALE_JSON, local_files=STALE_FILES) -> list:
    """Return the validated stale-image SHA-256 blocklist, or raise (fail closed).

    The committed JSON is authoritative. Any original export still present locally is re-hashed and must be IN the JSON;
    a local file whose hash is missing from it means the blocklist is incomplete or wrong, so the build aborts."""
    try:
        data = json.loads(Path(json_path).read_text())
        hashes = data["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"stale-image blocklist unreadable ({json_path}): {exc!r}") from exc
    if not isinstance(hashes, list) or not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise RuntimeError("stale-image blocklist must be a list of lowercase 64-hex SHA-256 strings")
    if len(set(hashes)) != len(hashes) or len(hashes) < MIN_STALE_HASHES:
        raise RuntimeError(f"stale-image blocklist needs >= {MIN_STALE_HASHES} distinct hashes, got {len(set(hashes))}")
    local = {pl.sha256_file(p) for p in local_files if Path(p).exists()}
    unknown = sorted(local - set(hashes))
    if unknown:
        raise RuntimeError(f"local original exports are not in the committed blocklist: {[h[:12] for h in unknown]}")
    return sorted(hashes)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def load_reference() -> dict:
    with open(BATCH_CSV, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def build_case(case_id: str, ref: dict) -> tuple:
    row = ref[case_id]
    if row["status"] != "success":
        raise RuntimeError(f"case {case_id}: reference measurement status is {row['status']!r}")
    idx = int(row["selected_slice_idx"])
    img_nii = nib.load(str(EXTRACT_DIR / "image" / f"{case_id}.nii.gz"))
    raw = np.asanyarray(img_nii.dataobj[:, :, idx])
    lab = np.asanyarray(nib.load(str(EXTRACT_DIR / "label" / f"{case_id}.nii.gz")).dataobj[:, :, idx])
    header_spacing = float(img_nii.header.get_zooms()[0])
    prompt_spacing = round(float(ast.literal_eval(row["voxel_spacing_mm"])[0]), 6)  # same value path as Milestone 6 (4-dp CSV)

    grey = window_ct_slice(raw, pl.INTENSITY_OFFSET, pl.WINDOW_CENTER, pl.WINDOW_WIDTH)
    rep = assess_export(grey, raw, intensity_offset=pl.INTENSITY_OFFSET, center=pl.WINDOW_CENTER,
                        width=pl.WINDOW_WIDTH, kidney_mask=lab == KIDNEY, tumor_mask=lab == TUMOR)
    if not rep.ok:
        raise RuntimeError(f"case {case_id}: export failed validation: {rep.failures}")
    n = grey.shape[0]
    case = {
        "case_id": case_id,
        "group": "scored" if case_id in pl.SCORED_CASES else "exploratory",
        "qc_flag": row["medvision_qc_flag"],
        "filename": f"case_{case_id}_slice_off1024.png",
        "slice_idx": idx,
        "size_px": [int(grey.shape[0]), int(grey.shape[1])],  # PNG (width, height) after transpose
        "pixel_spacing_mm_for_prompt": prompt_spacing,
        "pixel_spacing_mm_header": header_spacing,
        "fov_mm": round(n * header_spacing, 2),
        "effective_spacing_at_896px_mm_assumed": round(header_spacing * n / RESIZE_PX, 4),
        "ref_major_mm": float(row["major_axis_mm"]),
        "ref_minor_mm": float(row["minor_axis_mm"]),
        "build_checks_passed": True,
        "build_content_metrics": {k: rep.metrics[k] for k in ("window_coverage", "kidney_median_hu",
                                                               "tumor_mean_grey", "tumor_clipped_fraction")},
    }
    return case, np.ascontiguousarray(grey.T)


# ------------------------------------------------------------------------------------ notebook

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build_notebook(msha: str, lsha: str, bundle_id: str) -> dict:
    cells = []
    cells.append(md("""
# Phase 0 pilot - five-case VLM smoke test (corrected CT preprocessing)

**SMOKE TEST ONLY. This notebook does NOT validate tumor-measurement accuracy.** It checks that the pipeline now
runs on valid (non-blank) images and records exactly how the model behaves, with full raw logging.

- Model: `google/medgemma-1.5-4b-it` (gated; needs accepted terms + HF token in Colab secret `HF_TOKEN`).
- Inputs: `pilot5_images.zip` built locally by `build_pilot_bundle.py` (CT slices, corrected offset 1024).
  **The intensity offset 1024 is an UNVERIFIED hypothesis.**
- Scored cases: 0, 6, 8. Exploratory cases: 2, 4 (QC-flagged). Prompts are identical to Milestone 6.
- **Limitation:** Experiment B states the native pixel spacing, but the model sees an ~896x896 resized image, so the
  stated spacing does not describe the pixels the model sees.
- Fail-closed: nothing runs unless the bundle's SHA-256 hashes match values pinned inside this notebook.

**Run cells top to bottom. Restart the runtime after the install cell, then continue from Step 3.**
"""))
    cells += [md("## Step 1 - Enable GPU\nRuntime -> Change runtime type -> T4 GPU."), code("!nvidia-smi")]
    cells += [md("## Step 2 - Install dependencies (versions are recorded in the run metadata)"), code("""
!pip install -q -U transformers accelerate bitsandbytes huggingface_hub "pillow==11.3.0"
import transformers, accelerate, bitsandbytes, PIL
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("pillow:", PIL.__version__)
"""), md("### IMPORTANT - restart the runtime now (Runtime -> Restart session), then continue from Step 3. "
          "Do not re-run the install cell.")]
    cells += [md("## Step 3 - Hugging Face login (Colab secret; no token is stored in this notebook)"), code("""
from huggingface_hub import login

try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    print("Loaded token from Colab secret 'HF_TOKEN'.")
except Exception:
    from getpass import getpass
    hf_token = getpass("Enter your Hugging Face token: ")

login(token=hf_token)
hf_token = None
""")]
    cells += [md("## Step 4 - Upload and VERIFY the bundle (fail-closed)\n"
                 "Upload exactly `pilot5_images.zip`. The notebook aborts if the bundle is missing, modified, contains "
                 "extra files, contains any known stale/blank image, or fails any image check. The hashes below are "
                 "pinned at build time."),
              code(f'''
import hashlib, sys, zipfile
from pathlib import Path
from google.colab import files

EXPECTED_MANIFEST_SHA256 = "{msha}"
EXPECTED_PILOT_LIB_SHA256 = "{lsha}"
EXPECTED_BUNDLE_ID = "{bundle_id}"
BUNDLE_ZIP_NAME = "pilot5_images.zip"
BUNDLE_DIR = Path("pilot5_bundle")

BUNDLE_VERIFIED = False
if BUNDLE_DIR.exists():
    raise RuntimeError("pilot5_bundle/ already exists from an earlier run. Use Runtime -> Disconnect and delete runtime, "
                       "then start again, so no stale files can be mixed in.")

uploaded = files.upload()
if list(uploaded) != [BUNDLE_ZIP_NAME]:
    raise RuntimeError(f"Upload exactly one file named {{BUNDLE_ZIP_NAME}}; got {{list(uploaded)}}")

with zipfile.ZipFile(BUNDLE_ZIP_NAME) as zf:
    for name in zf.namelist():
        if name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe path in zip: {{name}}")
    zf.extractall(BUNDLE_DIR)

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

if _sha(BUNDLE_DIR / "manifest.json") != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("manifest.json does not match the hash pinned in this notebook. WRONG OR STALE BUNDLE. Aborting.")
if _sha(BUNDLE_DIR / "pilot_lib.py") != EXPECTED_PILOT_LIB_SHA256:
    raise RuntimeError("pilot_lib.py does not match the hash pinned in this notebook. Aborting.")

sys.path.insert(0, str(BUNDLE_DIR))
import pilot_lib

bundle = pilot_lib.verify_bundle(BUNDLE_DIR, EXPECTED_MANIFEST_SHA256, EXPECTED_PILOT_LIB_SHA256)
if bundle["manifest"]["bundle_id"] != EXPECTED_BUNDLE_ID:
    raise RuntimeError("bundle_id mismatch. Aborting.")
BUNDLE_VERIFIED = True
print("BUNDLE VERIFIED:", EXPECTED_BUNDLE_ID)
print(pilot_lib.BANNER)
for c in bundle["cases"]:
    print(f"  case {{c['case_id']}} [{{c['group']}}, qc={{c['qc_flag']}}] {{c['filename']}} {{c['size_px']}} sha256={{c['sha256'][:12]}}...")
'''),
              code("""
# Look at exactly what will be sent (no overlays, no outlines). Confirm they are real CT slices, not blank.
from PIL import Image
from IPython.display import display

for c in bundle["cases"]:
    im = Image.open(BUNDLE_DIR / "images" / c["filename"])
    print(f"case {c['case_id']} ({c['group']}) {im.size} {im.mode}")
    display(im.resize((im.size[0] * 2, im.size[1] * 2), Image.NEAREST))
""")]
    cells += [md("## Step 5 - Load the model ONCE (bf16 first; 4-bit fallback cell below)\n"
                 "The loading mode is recorded in the run metadata. Results from the two modes may differ."),
              code("""
import gc
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-1.5-4b-it"
assert MODEL_ID == pilot_lib.MODEL_ID

for _name in ("model", "processor"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
QUANTIZATION = "bf16"
print("Loaded in bf16.")
"""),
              md("### 4-bit fallback (only if the bf16 cell above hit a CUDA out-of-memory error)"),
              code("""
import gc
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "google/medgemma-1.5-4b-it"
assert MODEL_ID == pilot_lib.MODEL_ID

for _name in ("model", "processor"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map="auto")
QUANTIZATION = "nf4-4bit"
print("Loaded in 4-bit (nf4).")
""")]
    cells += [md("## Step 6 - Record run metadata BEFORE any inference (aborts if incomplete)\n"
                 "Creates a NEW run folder with a unique run id, validates the metadata and persists it. "
                 "Step 7 refuses to start unless the persisted file exists, is valid and equals what is in memory."),
              code("""
import datetime, json, platform
import importlib.metadata as importlib_metadata
import torch, transformers, PIL

run_meta = None  # a failure anywhere below must leave NO usable metadata behind for Step 7
if not BUNDLE_VERIFIED:
    raise RuntimeError("bundle not verified")
if "QUANTIZATION" not in globals():
    raise RuntimeError("load the model first (Step 5)")
if not torch.cuda.is_available():
    raise RuntimeError("no GPU available; refusing to run on CPU")

def _ver(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "n/a"

RUN_ID = pilot_lib.new_run_id()
STARTED_UTC = pilot_lib.utc_now()
OUT_DIR = Path(f"pilot5_run_{RUN_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=False)

processor_cfg = json.loads(json.dumps(processor.image_processor.to_dict(), default=str))
env = {
    "model_id": MODEL_ID,
    "model_revision": getattr(model.config, "_commit_hash", None) or "unresolved",
    "quantization": QUANTIZATION,
    "device": f"cuda:{torch.cuda.get_device_name(0)}",
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "accelerate_version": _ver("accelerate"),
    "bitsandbytes_version": _ver("bitsandbytes"),
    "pillow_version": PIL.__version__,
    "processor_image_config": processor_cfg,
    "chat_template_format": "processor.apply_chat_template([{role:user, content:[image, text]}], add_generation_prompt=True, tokenize=True, return_dict=True)",
}
_candidate = pilot_lib.new_run_metadata(bundle["manifest"], bundle["manifest_sha256"], env, RUN_ID, STARTED_UTC)
pilot_lib.validate_run_metadata(_candidate)
pilot_lib.atomic_write_json(OUT_DIR / "run_metadata.json", _candidate)
run_meta = _candidate  # assigned only after validation AND persistence succeeded
print("run_id:", RUN_ID, "| model revision:", env["model_revision"], "| quantization:", QUANTIZATION)
print("processor image config:", json.dumps(processor_cfg, indent=2)[:1500])
""")]
    cells += [md("## Step 7 - Inference (10 planned calls: 5 cases x experiments A and B)\n"
                 "Gate first: refuses to run unless the persisted metadata is valid and equals the in-memory metadata, "
                 "and the run folder holds nothing else (no earlier raw log, results or lock). A rerun therefore needs a "
                 "NEW run (re-run Step 6), never an append.\n\n"
                 "Every planned call is written to `raw_responses.jsonl` BEFORE it runs (write-ahead, flushed + fsynced), "
                 "then closed with its outcome: generated, exception, interrupted or aborted. If the run stops early or you "
                 "interrupt it, every remaining planned call is recorded as skipped with the reason, the partial log is "
                 "preserved, and the interruption is re-raised."),
              code("""
import time, traceback
from PIL import Image

if not BUNDLE_VERIFIED:
    raise RuntimeError("bundle not verified")
# Re-verify immediately before inference: nothing may have changed since Step 4.
bundle = pilot_lib.verify_bundle(BUNDLE_DIR, EXPECTED_MANIFEST_SHA256, EXPECTED_PILOT_LIB_SHA256)
# Fail-closed gate (raises RunGateError and touches nothing if anything is wrong; otherwise claims the folder).
pilot_lib.assert_ready_for_inference(OUT_DIR, run_meta, RUN_ID)

MAX_NEW_TOKENS = pilot_lib.MAX_NEW_TOKENS
RAW_LOG = OUT_DIR / pilot_lib.RAW_LOG_NAME
log = pilot_lib.make_logger(OUT_DIR)

try:
    IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    if IMAGE_TOKEN_ID is None or IMAGE_TOKEN_ID == getattr(processor.tokenizer, "unk_token_id", None):
        IMAGE_TOKEN_ID = None
except Exception:
    IMAGE_TOKEN_ID = None
_eos = model.generation_config.eos_token_id
EOS_IDS = [] if _eos is None else (list(_eos) if isinstance(_eos, (list, tuple)) else [_eos])

def load_image(case):
    return Image.open(BUNDLE_DIR / "images" / case["filename"]).convert("RGB")

def generate_once(image, prompt_text):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=model.dtype if hasattr(model, "dtype") else torch.bfloat16)
    input_len = int(inputs["input_ids"].shape[-1])
    n_image_tokens = int((inputs["input_ids"] == IMAGE_TOKEN_ID).sum()) if IMAGE_TOKEN_ID is not None else None
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    ids = [int(t) for t in out[0][input_len:].tolist()]
    return {
        "raw_response": processor.decode(ids, skip_special_tokens=True),
        "raw_response_with_special_tokens": processor.decode(ids, skip_special_tokens=False),
        "generated_token_ids": ids,
        "n_generated_tokens": len(ids),
        "input_token_count": input_len,
        "image_token_count": n_image_tokens,
        "finish_reason": pilot_lib.determine_finish_reason(ids, EOS_IDS, MAX_NEW_TOKENS),
        "eos_token_ids": EOS_IDS,
    }

# Records every planned call write-ahead; KeyboardInterrupt is recorded, the partial log is kept, then re-raised.
outcome = pilot_lib.run_planned_calls(bundle["cases"], OUT_DIR, RUN_ID, load_image, generate_once, log=log)
print(outcome)
""")]
    cells += [md("## Step 8 - Reconcile the log, parse, and summarise (also works after an interrupted or partial run)\n"
                 "Every planned call appears in `results.csv` (skipped and unfinished ones included, with their reason). "
                 "Scored and exploratory cases are written to separate files and summarised separately; nothing is "
                 "pooled and no accuracy aggregate is computed."),
              code("""
RAW_LOG = OUT_DIR / pilot_lib.RAW_LOG_NAME  # always the CURRENT run folder, never a variable left over from an earlier run
events = pilot_lib.read_jsonl(RAW_LOG)
accounting = pilot_lib.verify_run_accounting(events, expected_run_id=RUN_ID)
rows = pilot_lib.build_result_rows(events, bundle["manifest"], expected_run_id=RUN_ID)
pilot_lib.write_split_results(OUT_DIR, rows)

non_ok = [r for r in rows if r["parse_status"] != "ok"]
with open(OUT_DIR / "parse_failures.jsonl", "w", encoding="utf-8") as f:
    for r in non_ok:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\\n")

summary = pilot_lib.descriptive_summary(rows, accounting)
pilot_lib.atomic_write_json(OUT_DIR / "summary.json", summary)
meta_now = pilot_lib.update_run_metadata(OUT_DIR, {"n_non_ok_rows": len(non_ok), "log_fully_accounted": accounting["all_accounted"]})

print(pilot_lib.BANNER)
print("run state:", meta_now["status"], "| fully accounted:", accounting["all_accounted"], "| planned:", accounting["n_planned"])
print({k: v for k, v in accounting.items() if k.startswith("n_")})
print(json.dumps(summary["groups"], indent=2))
for r in rows:
    print(r["seq"], r["group"], r["case_id"], r["experiment"], r["call_status"], r["parse_status"], r["major_mm_pred"], r["minor_mm_pred"], r["skip_reason"] or "")
""")]
    cells += [md("## Step 9 - Download the run folder\nPlace the unzipped folder at `validation/phase0_pilot5/runs/<run_id>/` in the "
                 "repo. Do not put it in `outputs/` or `results/`."),
              code("""
import shutil
archive = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print("created", archive)
files.download(archive)
""")]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


# ------------------------------------------------------------------------------------ main

def main() -> None:
    ref = load_reference()
    if BUNDLE_DIR.exists():
        assert BUNDLE_DIR.resolve().parent == HERE, "refusing to delete outside the pilot directory"
        shutil.rmtree(BUNDLE_DIR)
    (BUNDLE_DIR / "images").mkdir(parents=True)

    cases = []
    for cid in pl.CASE_ORDER:
        case, grey_t = build_case(cid, ref)
        path = BUNDLE_DIR / "images" / case["filename"]
        Image.fromarray(grey_t).save(path)
        case["sha256"] = pl.sha256_file(path)
        case["image_metrics"] = pl.image_metrics(np.array(Image.open(path)))
        cases.append(case)
        print(f"case {cid} [{case['group']}] {case['size_px']} sha={case['sha256'][:12]} coverage={case['build_content_metrics']['window_coverage']:.2f}")

    shutil.copyfile(HERE / "pilot_lib.py", BUNDLE_DIR / "pilot_lib.py")
    stale = load_stale_blocklist()

    status = git("status", "--porcelain")
    manifest = pl.assemble_manifest(
        cases, stale_sha256=stale,
        git_info={"commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(status)},
        ct_preprocessing_sha256=pl.sha256_file(REPO / "src" / "ct_preprocessing.py"),
        pilot_lib_sha256=pl.sha256_file(BUNDLE_DIR / "pilot_lib.py"),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    bundle_id = manifest["bundle_id"]
    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    msha = pl.sha256_file(BUNDLE_DIR / "manifest.json")
    lsha = pl.sha256_file(BUNDLE_DIR / "pilot_lib.py")
    pl.verify_bundle(BUNDLE_DIR, msha, lsha)  # the freshly built bundle must pass its own verification

    pl.write_deterministic_zip(BUNDLE_DIR, ZIP_PATH)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(msha, lsha, bundle_id), indent=1) + "\n")
    print(f"\nbundle_id={bundle_id}\nmanifest sha256={msha}\npilot_lib sha256={lsha}\nzip sha256={pl.sha256_file(ZIP_PATH)}")
    print(f"wrote {BUNDLE_DIR}, {ZIP_PATH.name}, {NOTEBOOK_PATH.name} (git tree dirty: {bool(status)})")


if __name__ == "__main__":
    main()
