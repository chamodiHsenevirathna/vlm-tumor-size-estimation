"""
Pure-Python helpers for the Phase 0 five-case pilot (no torch/transformers imports,
so everything here is unit-testable without a GPU). The Colab notebook imports this
file from the uploaded bundle after checking its SHA-256 against a pinned constant.

THIS PILOT IS A SMOKE TEST. It does not validate tumor-measurement accuracy.

Intensity offset 1024 is an UNVERIFIED hypothesis (stored = HU + 1024); dataset
provenance for the KiPA22 intensity encoding was not available.
"""

import csv
import datetime
import hashlib
import json
import math
import os
import re
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA_VERSION = "phase0_pilot5/2"
PROMPT_VERSION = "m6-prompts-v1"  # identical wording to notebooks/milestone6_vlm_batch5.ipynb
MODEL_ID = "google/medgemma-1.5-4b-it"
MAX_NEW_TOKENS = 300
GENERATION_SETTINGS = {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS}
INTENSITY_OFFSET = 1024.0
OFFSET_STATUS = "UNVERIFIED hypothesis (stored = HU + 1024); dataset provenance unavailable"
WINDOW_CENTER, WINDOW_WIDTH = 40.0, 400.0
SCORED_CASES = ["0", "6", "8"]
EXPLORATORY_CASES = ["2", "4"]
CASE_ORDER = SCORED_CASES + EXPLORATORY_CASES
EXPERIMENTS = ("A_no_spacing", "B_with_spacing")
EXPECTED_CALLS = len(CASE_ORDER) * len(EXPERIMENTS)
BANNER = ("SMOKE TEST ONLY: 5 cases, 3 scored. This pilot checks whether the pipeline now runs on valid "
          "images and how outputs behave. It does NOT validate tumor-measurement accuracy.")
REQUIRED_LIMITATION_IDS = ["no_accuracy_validation", "effective_spacing_mismatch", "offset_unverified",
                           "small_n", "preprocessor_config_unverified"]

# Generic grey-level backstop; must equal ct_preprocessing.QualityThresholds defaults (tested).
MIN_UNIQUE_LEVELS, MIN_GREY_STD, MIN_ENTROPY_BITS, MAX_SATURATED_FRACTION = 128, 20.0, 5.0, 0.25


class BundleVerificationError(RuntimeError):
    pass


class MetadataError(ValueError):
    pass


# ---------------------------------------------------------------- prompts (verbatim from Milestone 6)

def prompt_a():
    return (
        "You are shown an axial CT slice of a kidney. Identify the kidney tumor (if visible) and estimate its size.\n"
        "Report the length of the tumor's longest axis (major axis) and the length of its shortest perpendicular axis (minor axis), in millimeters.\n"
        "Respond only in this exact format:\n"
        '<answer>{"major_mm": X, "minor_mm": Y}</answer>\n'
        "where X and Y are numbers in millimeters."
    )


def prompt_b(pixel_spacing_mm):
    return (
        f"You are shown an axial CT slice of a kidney. This image has a pixel spacing of {pixel_spacing_mm} mm x {pixel_spacing_mm} mm per pixel\n"
        "(each pixel represents that many millimeters of real-world distance, in both the horizontal and vertical directions).\n"
        "Using this scale, identify the kidney tumor (if visible) and estimate its size.\n"
        "Report the length of the tumor's longest axis (major axis) and the length of its shortest perpendicular axis (minor axis), in millimeters.\n"
        "Respond only in this exact format:\n"
        '<answer>{"major_mm": X, "minor_mm": Y}</answer>\n'
        "where X and Y are numbers in millimeters."
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hashes() -> dict:
    return {"prompt_a_sha256": sha256_text(prompt_a()), "prompt_b_template_sha256": sha256_text(prompt_b("{S}"))}


LIMITATIONS = [
    {"id": "no_accuracy_validation",
     "text": "This pilot is a smoke test. It does not validate tumor-measurement accuracy, and nothing computed from it "
             "supports any conclusion about MedGemma's measurement ability or about the effect of pixel spacing."},
    {"id": "effective_spacing_mismatch",
     "text": "Experiment B states the NATIVE pixel spacing (e.g. 0.5859 mm/px) but the model receives an image resized to "
             "about 896x896, where one pixel spans roughly 0.09-0.12 mm. The stated spacing does not describe the pixels "
             "the model actually sees. Prompts are kept identical to Milestone 6 for comparability, not because this is correct."},
    {"id": "offset_unverified",
     "text": "The CT intensity offset (stored = HU + 1024) is an unverified, dataset-specific hypothesis; the data only "
             "constrains it to roughly 1000-1040 and no documentation confirms it."},
    {"id": "small_n",
     "text": "Five cases: 3 scored (0, 6, 8; pass_basic QC) and 2 exploratory (2, 4; ellipse_outside_bbox QC flag). "
             "Cases 3 and 5 from Milestone 6 are not included."},
    {"id": "preprocessor_config_unverified",
     "text": "The checkpoint's image preprocessor config was not accessible (gated). Resize/normalisation details are "
             "taken from the public model card (896x896, 256 image tokens) and are recorded at run time."},
    {"id": "reference_is_proxy",
     "text": "Reference measurements are ellipse fits to a single 2D ground-truth mask slice, not clinician measurements."},
]


def write_deterministic_zip(src, dest) -> None:
    """Zip a directory with fixed timestamps and sorted entries, so the same content gives the same bytes."""
    src = Path(src)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(src).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())


def assemble_manifest(cases: list, *, stale_sha256: list, git_info: dict, ct_preprocessing_sha256: str,
                      pilot_lib_sha256: str, created_utc: str) -> dict:
    """Build the bundle manifest from fully described cases. Shared by the real builder and the test fixtures."""
    bundle_id = hashlib.sha256("\n".join(f"{c['filename']}:{c['sha256']}" for c in cases).encode()).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_utc": created_utc,
        "banner": BANNER,
        "git": git_info,
        "pipeline": {
            "intensity_offset": INTENSITY_OFFSET, "offset_status": OFFSET_STATUS,
            "window_center": WINDOW_CENTER, "window_width": WINDOW_WIDTH, "window_unit": "HU",
            "ct_preprocessing_sha256": ct_preprocessing_sha256,
            "orientation": "slice[:, :, idx].T (radiological: anterior up, patient left on image right)",
            "model_input_has_overlays": False,
        },
        "groups": {"scored": SCORED_CASES, "exploratory": EXPLORATORY_CASES},
        "prompts": {"version": PROMPT_VERSION, **prompt_hashes(),
                    "note": "wording identical to notebooks/milestone6_vlm_batch5.ipynb"},
        "generation_settings": GENERATION_SETTINGS,
        "limitations": LIMITATIONS,
        "known_stale_sha256": sorted(stale_sha256),
        "pilot_lib_sha256": pilot_lib_sha256,
        "build_validation": {"all_passed": True, "checks": "ct_preprocessing.assess_export (content-aware + generic) per case"},
        "cases": cases,
    }


# ---------------------------------------------------------------- image metrics + bundle verification

def image_metrics(arr: np.ndarray) -> dict:
    p = np.bincount(arr.ravel(), minlength=256) / arr.size
    p = p[p > 0]
    return {
        "n_unique_levels": int(len(np.unique(arr))),
        "grey_std": float(arr.std()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "frac_ge_254": float((arr >= 254).mean()),
        "frac_le_1": float((arr <= 1).mean()),
    }


def generic_image_failures(m: dict) -> list:
    f = []
    if m["n_unique_levels"] < MIN_UNIQUE_LEVELS:
        f.append(f"n_unique_levels {m['n_unique_levels']} < {MIN_UNIQUE_LEVELS}")
    if m["grey_std"] < MIN_GREY_STD:
        f.append(f"grey_std {m['grey_std']:.1f} < {MIN_GREY_STD}")
    if m["entropy_bits"] < MIN_ENTROPY_BITS:
        f.append(f"entropy_bits {m['entropy_bits']:.2f} < {MIN_ENTROPY_BITS}")
    if m["frac_ge_254"] > MAX_SATURATED_FRACTION:
        f.append(f"frac_ge_254 {m['frac_ge_254']:.2f} > {MAX_SATURATED_FRACTION}")
    if m["frac_le_1"] > MAX_SATURATED_FRACTION:
        f.append(f"frac_le_1 {m['frac_le_1']:.2f} > {MAX_SATURATED_FRACTION}")
    return f


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def verify_bundle(bundle_dir, expected_manifest_sha256: str, expected_pilot_lib_sha256: str = None) -> dict:
    """Fail-closed verification of an extracted pilot bundle. Raises BundleVerificationError
    listing every problem found; returns the verified manifest and ordered cases otherwise."""
    bundle_dir = Path(bundle_dir)
    mpath = bundle_dir / "manifest.json"
    if not mpath.is_file():
        raise BundleVerificationError("manifest.json missing")
    msha = sha256_file(mpath)
    if msha != expected_manifest_sha256:
        raise BundleVerificationError(
            f"manifest sha256 {msha} != pinned {expected_manifest_sha256} (wrong or modified bundle)")
    try:
        manifest = json.loads(mpath.read_text())
    except json.JSONDecodeError as exc:
        raise BundleVerificationError(f"manifest.json is not valid JSON: {exc}")

    fails = []
    try:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            fails.append(f"schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}")

        pipe = manifest.get("pipeline", {})
        if pipe.get("intensity_offset") != INTENSITY_OFFSET:
            fails.append(f"intensity_offset {pipe.get('intensity_offset')!r} != {INTENSITY_OFFSET}")
        if pipe.get("offset_status") != OFFSET_STATUS or not str(pipe.get("offset_status", "")).startswith("UNVERIFIED"):
            fails.append("offset_status must mark the offset as UNVERIFIED")
        if (pipe.get("window_center"), pipe.get("window_width")) != (WINDOW_CENTER, WINDOW_WIDTH):
            fails.append("window center/width mismatch")

        groups = manifest.get("groups", {})
        if groups.get("scored") != SCORED_CASES or groups.get("exploratory") != EXPLORATORY_CASES:
            fails.append(f"groups must be scored={SCORED_CASES} exploratory={EXPLORATORY_CASES}, got {groups}")

        prompts = manifest.get("prompts", {})
        expect = prompt_hashes()
        if prompts.get("version") != PROMPT_VERSION:
            fails.append(f"prompt version {prompts.get('version')!r} != {PROMPT_VERSION!r}")
        for k, v in expect.items():
            if prompts.get(k) != v:
                fails.append(f"{k} does not match the prompts in pilot_lib.py")

        lim_ids = {x.get("id") for x in manifest.get("limitations", [])}
        for lid in REQUIRED_LIMITATION_IDS:
            if lid not in lim_ids:
                fails.append(f"required limitation {lid!r} missing from manifest")

        stale = set(manifest.get("known_stale_sha256", []))
        if not stale:
            fails.append("known_stale_sha256 is empty (stale-image guard disabled)")

        bv = manifest.get("build_validation", {})
        if bv.get("all_passed") is not True:
            fails.append("build_validation.all_passed is not true")

        cases = manifest.get("cases", [])
        ids = [c.get("case_id") for c in cases]
        if ids != CASE_ORDER:
            fails.append(f"case order {ids} != {CASE_ORDER}")

        # exact file inventory
        expected_files = {"manifest.json", "pilot_lib.py"} | {f"images/{c.get('filename')}" for c in cases}
        actual_files = {p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()}
        if actual_files != expected_files:
            fails.append(f"file inventory mismatch: unexpected={sorted(actual_files - expected_files)} "
                         f"missing={sorted(expected_files - actual_files)}")

        lib_path = bundle_dir / "pilot_lib.py"
        if lib_path.is_file():
            lsha = sha256_file(lib_path)
            if lsha != manifest.get("pilot_lib_sha256"):
                fails.append("pilot_lib.py sha256 differs from manifest")
            if expected_pilot_lib_sha256 is not None and lsha != expected_pilot_lib_sha256:
                fails.append("pilot_lib.py sha256 differs from the pinned value")

        seen = {}
        for c in cases:
            cid = c.get("case_id")
            tag = f"case {cid}"
            want_group = "scored" if cid in SCORED_CASES else "exploratory"
            if c.get("group") != want_group:
                fails.append(f"{tag}: group {c.get('group')!r} != {want_group!r}")
            if c.get("filename") != f"case_{cid}_slice_off1024.png":
                fails.append(f"{tag}: unexpected filename {c.get('filename')!r}")
            for k in ("pixel_spacing_mm_for_prompt", "ref_major_mm", "ref_minor_mm"):
                if not _finite_positive(c.get(k)):
                    fails.append(f"{tag}: {k} must be a finite positive number")
            path = bundle_dir / "images" / str(c.get("filename"))
            if not path.is_file():
                continue
            sha = sha256_file(path)
            if sha != c.get("sha256"):
                fails.append(f"{tag}: image sha256 differs from manifest")
            if sha in stale:
                fails.append(f"{tag}: image is a KNOWN STALE (blank, offset-0) export")
            if sha in seen:
                fails.append(f"{tag}: image identical to case {seen[sha]}")
            seen[sha] = cid
            try:
                with Image.open(path) as im:
                    im.load()
                    mode, size = im.mode, im.size
                    arr = np.array(im)
            except Exception as exc:  # noqa: BLE001
                fails.append(f"{tag}: cannot decode image ({exc})")
                continue
            if mode != "L" or arr.dtype != np.uint8 or arr.ndim != 2:
                fails.append(f"{tag}: expected 8-bit greyscale 2D, got mode {mode} dtype {arr.dtype} ndim {arr.ndim}")
                continue
            if list(size) != c.get("size_px"):
                fails.append(f"{tag}: size {list(size)} != manifest {c.get('size_px')}")
            m = image_metrics(arr)
            fails += [f"{tag}: {x}" for x in generic_image_failures(m)]
            for k, v in c.get("image_metrics", {}).items():
                if not math.isclose(m.get(k, float("nan")), v, rel_tol=1e-9, abs_tol=1e-9):
                    fails.append(f"{tag}: recomputed {k} {m.get(k)} != manifest {v}")
            if not c.get("build_checks_passed"):
                fails.append(f"{tag}: build-time content checks not recorded as passed")
    except (AttributeError, TypeError, KeyError) as exc:
        fails.append(f"malformed manifest ({type(exc).__name__}: {exc})")

    if fails:
        raise BundleVerificationError("; ".join(fails))
    return {"manifest": manifest, "manifest_sha256": msha, "cases": manifest["cases"]}


# ---------------------------------------------------------------- run metadata

REQUIRED_ENV_KEYS = ["model_id", "model_revision", "quantization", "device", "python_version", "torch_version",
                     "transformers_version", "accelerate_version", "bitsandbytes_version", "pillow_version",
                     "processor_image_config", "chat_template_format"]


def new_run_metadata(manifest: dict, manifest_sha256: str, env: dict, run_id: str, started_utc: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "started",
        "started_utc": started_utc,
        "banner": BANNER,
        "bundle": {"bundle_id": manifest.get("bundle_id"), "manifest_sha256": manifest_sha256,
                   "pilot_lib_sha256": manifest.get("pilot_lib_sha256"), "git_commit_at_build": manifest["git"]},
        "preprocessing": {
            "intensity_offset": manifest["pipeline"]["intensity_offset"],
            "offset_status": manifest["pipeline"]["offset_status"],
            "window_center_hu": manifest["pipeline"]["window_center"],
            "window_width_hu": manifest["pipeline"]["window_width"],
        },
        "inputs": [{k: c[k] for k in ("case_id", "group", "qc_flag", "filename", "sha256", "size_px",
                                       "pixel_spacing_mm_for_prompt", "pixel_spacing_mm_header", "fov_mm",
                                       "effective_spacing_at_896px_mm_assumed")} for c in manifest["cases"]],
        "model": {k: env[k] for k in ("model_id", "model_revision", "quantization", "device")},
        "environment": {k: env[k] for k in REQUIRED_ENV_KEYS
                        if k not in ("model_id", "model_revision", "quantization", "device")},
        "prompts": dict(manifest["prompts"]),
        "generation_settings": dict(GENERATION_SETTINGS),
        "limitations": manifest["limitations"],
    }


def validate_run_metadata(meta: dict) -> None:
    """Raise MetadataError unless the run metadata is complete and honest."""
    problems = []
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", str(meta.get("run_id", ""))):
        problems.append("run_id must be a unique id like 20260101T000000Z-1a2b3c4d")
    pre = meta.get("preprocessing", {})
    if pre.get("intensity_offset") != INTENSITY_OFFSET:
        problems.append("intensity_offset must be 1024")
    if not str(pre.get("offset_status", "")).startswith("UNVERIFIED"):
        problems.append("offset_status must be marked UNVERIFIED")
    if (pre.get("window_center_hu"), pre.get("window_width_hu")) != (WINDOW_CENTER, WINDOW_WIDTH):
        problems.append("window mismatch")
    model = meta.get("model", {})
    if model.get("model_id") != MODEL_ID:
        problems.append(f"model_id must be {MODEL_ID}")
    for k in ("model_revision", "quantization", "device"):
        if not model.get(k):
            problems.append(f"model.{k} missing (use 'unresolved' if it cannot be determined)")
    envd = meta.get("environment", {})
    for k in REQUIRED_ENV_KEYS:
        if k not in ("model_id", "model_revision", "quantization", "device") and k not in envd:
            problems.append(f"environment.{k} missing")
    if not isinstance(envd.get("processor_image_config"), dict) or not envd.get("processor_image_config"):
        problems.append("environment.processor_image_config must be a non-empty dict")
    if meta.get("generation_settings") != GENERATION_SETTINGS:
        problems.append("generation_settings differ from the registered settings")
    pr = meta.get("prompts", {})
    if pr.get("version") != PROMPT_VERSION:
        problems.append("prompt version missing/mismatched")
    if not meta.get("inputs") or [i["case_id"] for i in meta["inputs"]] != CASE_ORDER:
        problems.append("inputs must list the five cases in order")
    if not meta.get("bundle", {}).get("manifest_sha256"):
        problems.append("bundle.manifest_sha256 missing")
    if "no_accuracy_validation" not in {x.get("id") for x in meta.get("limitations", [])}:
        problems.append("limitations must state that accuracy is not validated")
    if problems:
        raise MetadataError("; ".join(problems))


# ---------------------------------------------------------------- run gate, durable logging, run orchestration

RAW_LOG_NAME = "raw_responses.jsonl"
RUN_LOG_NAME = "run.log"
METADATA_NAME = "run_metadata.json"
LOCK_NAME = "inference.lock"
RUN_ID_RE = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")
RUN_STATES = ("completed", "completed_with_failures", "stopped_early", "interrupted", "aborted")


_GATE_CLAIMS = {}  # in-process, single-use: run folder -> sha256 of the persisted metadata the gate approved


class RunGateError(RuntimeError):
    """Inference must not start (or continue): a precondition about the run folder or metadata failed."""


class RunLogError(ValueError):
    """The raw call log is inconsistent (mixed runs, duplicate plan, unknown sequence numbers, ...)."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_run_id() -> str:
    """Unique per Step 6 execution: UTC timestamp plus a random suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _fsync_dir(path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def append_jsonl(path, record: dict) -> None:
    """Append one JSON object as one line, flushed and fsynced (file, and directory when the file is new),
    so a crash cannot lose a record that was reported as written."""
    path = Path(path)
    is_new = not path.exists()
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    if is_new:
        _fsync_dir(path.parent)


def read_jsonl(path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON line ({exc})")
    return out


def atomic_write_json(path, obj) -> None:
    """Write JSON via a temp file + fsync + rename, so readers never see a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def make_logger(out_dir, echo=print):
    path = Path(out_dir) / RUN_LOG_NAME

    def log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        echo(line)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    return log


def determine_finish_reason(token_ids, eos_ids, max_new_tokens=MAX_NEW_TOKENS) -> str:
    if token_ids and token_ids[-1] in set(eos_ids or []):
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "length"
    return "unknown"


def assert_ready_for_inference(out_dir, run_meta, run_id) -> None:
    """Gate for inference. Raises RunGateError (touching nothing) unless:
      * run_meta is a dict created this session, valid, and identical to the PERSISTED run_metadata.json;
      * the folder name and both run ids match `run_id`;
      * the folder contains nothing but run_metadata.json (no raw log, run log, results, lock, ...).
    On success it atomically claims the run folder with an O_EXCL lock file, so a second pass over the
    same folder (a rerun of the inference cell) is refused."""
    problems = []
    out_dir = Path(out_dir)
    if not isinstance(run_meta, dict):
        raise RunGateError("run metadata missing: Step 6 did not complete in this session (run_meta is not a dict)")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise RunGateError(f"run_id {run_id!r} is not a valid unique run id")
    if not out_dir.is_dir():
        raise RunGateError(f"run folder {out_dir} does not exist")
    if out_dir.name != f"pilot5_run_{run_id}":
        problems.append(f"folder name {out_dir.name!r} does not match run_id {run_id!r}")

    try:
        validate_run_metadata(run_meta)
    except MetadataError as exc:
        problems.append(f"in-memory metadata invalid: {exc}")

    mpath = out_dir / METADATA_NAME
    persisted = None
    if not mpath.is_file():
        problems.append(f"{METADATA_NAME} was not persisted (Step 6 did not complete)")
    else:
        try:
            persisted = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"persisted {METADATA_NAME} is not valid JSON: {exc}")
    if isinstance(persisted, dict):
        try:
            validate_run_metadata(persisted)
        except MetadataError as exc:
            problems.append(f"persisted metadata invalid: {exc}")
        try:
            if persisted != json.loads(json.dumps(run_meta)):
                problems.append("persisted metadata differs from the in-memory metadata")
        except (TypeError, ValueError) as exc:
            problems.append(f"in-memory metadata is not JSON-serialisable: {exc}")
        if persisted.get("run_id") != run_id or run_meta.get("run_id") != run_id:
            problems.append("metadata run_id does not match the current run_id")
        if persisted.get("status") != "started":
            problems.append(f"persisted metadata status is {persisted.get('status')!r}, expected 'started'")
    elif mpath.is_file():
        problems.append("persisted metadata is not a JSON object")

    extra = sorted(p.name for p in out_dir.iterdir() if p.name != METADATA_NAME)
    if extra:
        problems.append(f"run artifacts already exist in this folder: {extra} (a rerun must use a new run folder)")
    if problems:
        raise RunGateError("; ".join(problems))

    try:
        fd = os.open(out_dir / LOCK_NAME, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunGateError("inference lock already exists for this run folder")
    try:
        os.write(fd, json.dumps({"run_id": run_id, "claimed_utc": utc_now()}).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(out_dir)
    _GATE_CLAIMS[str(out_dir.resolve())] = sha256_file(mpath)


def _exc_info(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def plan_calls(cases: list) -> list:
    plan, seq = [], 0
    for c in cases:
        for exp in EXPERIMENTS:
            seq += 1
            plan.append({"seq": seq, "case_id": c["case_id"], "group": c["group"], "experiment": exp})
    return plan


def build_prompt(case: dict, experiment: str) -> str:
    return prompt_a() if experiment == "A_no_spacing" else prompt_b(case["pixel_spacing_mm_for_prompt"])


def update_run_metadata(out_dir, updates: dict) -> dict:
    """Merge `updates` into the PERSISTED run_metadata.json (atomic rewrite) and return the new content."""
    path = Path(out_dir) / METADATA_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(updates)
    atomic_write_json(path, meta)
    return meta


def run_planned_calls(cases, out_dir, run_id, load_image, generate_fn, log=None, max_consecutive_failures=3) -> dict:
    """Run every planned call with write-ahead logging.

    Order of durable records in raw_responses.jsonl: one `plan` record listing every planned call; then, per
    call, `attempt_started` (written and fsynced BEFORE the image is loaded or the model is called) followed by
    exactly one `attempt_finished` (status generated | exception | interrupted | aborted). If execution stops
    early (consecutive failures, KeyboardInterrupt, other BaseException) every remaining planned call gets an
    `attempt_skipped` record with the reason, then `run_finished`. KeyboardInterrupt/BaseException are recorded,
    the partial log and run metadata are finalised, and the exception is re-raised.
    Every record carries `run_id`. `assert_ready_for_inference` must have been called first (it creates the lock)."""
    out_dir = Path(out_dir)
    raw = out_dir / RAW_LOG_NAME
    approved_sha = _GATE_CLAIMS.pop(str(out_dir.resolve()), None)  # single use: consumed even if a check below fails
    if approved_sha is None:
        raise RunGateError("no gate approval for this run folder in this process: call assert_ready_for_inference "
                           "first (a lock file alone is not enough, and an approval can be used only once)")
    lock = out_dir / LOCK_NAME
    try:
        lock_ok = json.loads(lock.read_text(encoding="utf-8")).get("run_id") == run_id
    except (OSError, ValueError, AttributeError):
        lock_ok = False
    if not lock_ok:
        raise RunGateError("inference lock missing or not for this run_id")
    if not (out_dir / METADATA_NAME).is_file() or sha256_file(out_dir / METADATA_NAME) != approved_sha:
        raise RunGateError("run metadata changed after the gate approved it; refusing to run")
    if raw.exists():
        raise RunGateError(f"{RAW_LOG_NAME} already exists; refusing to append a second pass to an existing run")
    log = log or (lambda msg: None)

    plan = plan_calls(cases)
    by_seq = {p["seq"]: c for p, c in zip(plan, [c for c in cases for _ in EXPERIMENTS])}
    append_jsonl(raw, {"event": "plan", "run_id": run_id, "schema_version": SCHEMA_VERSION, "ts_utc": utc_now(),
                       "n_planned": len(plan), "planned_calls": plan, "prompt_version": PROMPT_VERSION,
                       "max_new_tokens": MAX_NEW_TOKENS})
    log(f"run {run_id}: planned {len(plan)} calls. {BANNER}")

    counts = {"n_generated": 0, "n_exception": 0, "n_interrupted": 0, "n_aborted": 0, "n_skipped": 0}
    state, reraise, consecutive = None, None, 0

    def skip_rest(from_index: int, reason: str) -> None:
        for p in plan[from_index:]:
            append_jsonl(raw, {"event": "attempt_skipped", "run_id": run_id, "seq": p["seq"], "case_id": p["case_id"],
                               "group": p["group"], "experiment": p["experiment"], "reason": reason, "ts_utc": utc_now()})
            counts["n_skipped"] += 1
        log(f"skipped {len(plan) - from_index} remaining planned call(s): {reason}")

    for idx, p in enumerate(plan):
        case = by_seq[p["seq"]]
        prompt_text = build_prompt(case, p["experiment"])
        base = {"run_id": run_id, "seq": p["seq"], "case_id": p["case_id"], "group": p["group"],
                "experiment": p["experiment"], "prompt_version": PROMPT_VERSION, "prompt_text": prompt_text,
                "prompt_sha256": sha256_text(prompt_text), "image_file": case["filename"],
                "image_sha256": case["sha256"], "max_new_tokens": MAX_NEW_TOKENS}
        started_utc = utc_now()
        append_jsonl(raw, {**base, "event": "attempt_started", "started_utc": started_utc})  # write-ahead

        t0 = time.time()
        status, exc_info, result, image_size, interruption = "generated", None, {}, None, None
        try:
            image = load_image(case)
            image_size = list(getattr(image, "size", []) or [])
            result = generate_fn(image, prompt_text)
        except KeyboardInterrupt as exc:
            status, interruption, exc_info = "interrupted", exc, _exc_info(exc)
        except Exception as exc:  # noqa: BLE001 - recorded in full, never swallowed
            status, exc_info = "exception", _exc_info(exc)
        except BaseException as exc:  # noqa: BLE001 - e.g. SystemExit: record, then re-raise
            status, interruption, exc_info = "aborted", exc, _exc_info(exc)

        generation_keys = ("raw_response", "raw_response_with_special_tokens", "generated_token_ids",
                           "n_generated_tokens", "input_token_count", "image_token_count", "finish_reason",
                           "eos_token_ids")
        finished = {**base, "event": "attempt_finished", "status": status, "started_utc": started_utc,
                    "finished_utc": utc_now(), "latency_s": round(time.time() - t0, 3), "image_size_px": image_size,
                    "exception": exc_info}
        for k in generation_keys:
            finished[k] = result.get(k) if status == "generated" else None
        if status != "generated":
            finished["finish_reason"] = status
        finished["hit_max_new_tokens"] = (finished.get("finish_reason") == "length") if status == "generated" else None
        append_jsonl(raw, finished)
        counts[{"generated": "n_generated", "exception": "n_exception", "interrupted": "n_interrupted",
                "aborted": "n_aborted"}[status]] += 1
        log(f"call {p['seq']}/{len(plan)} case {p['case_id']} {p['experiment']}: {status} "
            f"finish={finished['finish_reason']} tokens={finished['n_generated_tokens']} latency={finished['latency_s']}s")

        if interruption is not None:
            state, reraise = ("interrupted" if status == "interrupted" else "aborted"), interruption
            skip_rest(idx + 1, status)
            break
        consecutive = consecutive + 1 if status == "exception" else 0
        if consecutive >= max_consecutive_failures:
            state = "stopped_early"
            skip_rest(idx + 1, f"stopped_after_{max_consecutive_failures}_consecutive_failures")
            break
    else:
        state = "completed" if counts["n_exception"] == 0 else "completed_with_failures"

    append_jsonl(raw, {"event": "run_finished", "run_id": run_id, "state": state, "ts_utc": utc_now(),
                       "n_planned": len(plan), **counts})
    update_run_metadata(out_dir, {"status": state, "finished_utc": utc_now(), "n_planned": len(plan), **counts})
    log(f"run {run_id} finished: {state} {counts}")
    outcome = {"state": state, "n_planned": len(plan), **counts}
    if reraise is not None:
        raise reraise
    return outcome


def verify_run_accounting(events: list, expected_run_id: str = None) -> dict:
    """Reconcile the raw log against its own plan. Raises RunLogError if the log mixes runs, has no or several
    plans (e.g. a rerun appended to it), or contains records for calls that were never planned. Hard-killed runs
    are reported (n_unfinished / n_never_started), not hidden."""
    run_ids = {e.get("run_id") for e in events}
    if not events or None in run_ids or len(run_ids) != 1:
        raise RunLogError(f"log must contain exactly one run id, found {sorted(map(str, run_ids))}")
    run_id = next(iter(run_ids))
    if expected_run_id is not None and run_id != expected_run_id:
        raise RunLogError(f"log run_id {run_id} != expected {expected_run_id}")
    plans = [e for e in events if e.get("event") == "plan"]
    if len(plans) != 1:
        raise RunLogError(f"expected exactly one plan record, found {len(plans)} (a second pass appended to this log?)")
    planned = {c["seq"]: c for c in plans[0]["planned_calls"]}

    started, finished, skipped = {}, {}, {}
    for e in events:
        kind = e.get("event")
        if kind in ("attempt_started", "attempt_finished", "attempt_skipped"):
            seq = e.get("seq")
            if seq not in planned:
                raise RunLogError(f"record for unplanned call seq={seq}")
            bucket = {"attempt_started": started, "attempt_finished": finished, "attempt_skipped": skipped}[kind]
            if seq in bucket:
                raise RunLogError(f"duplicate {kind} for seq={seq}")
            bucket[seq] = e
        elif kind not in ("plan", "run_finished"):
            raise RunLogError(f"unknown event {kind!r}")
    final = [e for e in events if e.get("event") == "run_finished"]
    if len(final) > 1:
        raise RunLogError("more than one run_finished record")

    per_seq, c = {}, {"n_generated": 0, "n_exception": 0, "n_interrupted": 0, "n_aborted": 0, "n_skipped": 0,
                      "n_unfinished": 0, "n_never_started": 0}
    for seq in sorted(planned):
        if seq in finished and seq in skipped:
            raise RunLogError(f"seq={seq} is both finished and skipped")
        if seq in finished:
            per_seq[seq] = finished[seq]["status"]
            c["n_" + per_seq[seq]] += 1
        elif seq in skipped:
            per_seq[seq] = "skipped"
            c["n_skipped"] += 1
        elif seq in started:
            per_seq[seq] = "unfinished"
            c["n_unfinished"] += 1
        else:
            per_seq[seq] = "never_started"
            c["n_never_started"] += 1
    return {"run_id": run_id, "n_planned": len(planned), "per_seq": per_seq, **c,
            "all_accounted": c["n_unfinished"] == 0 and c["n_never_started"] == 0,
            "run_finished_state": final[0]["state"] if final else None}


# ---------------------------------------------------------------- parsing (Milestone 6 logic, preserved + stricter)

REFUSAL_MARKERS = ["i cannot", "i can't", "i am not able", "i'm not able", "unable to provide",
                   "cannot provide a diagnosis"]
_ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


def _is_valid_length(v) -> bool:
    return math.isfinite(v) and v > 0


def parse_answer(raw_text):
    """Return (major_mm, minor_mm, status, n_answer_blocks). Status: ok | refused | parse_failed | invalid_numeric.
    Never edits values: invalid numbers stay visible. Only the first <answer> block is used."""
    if not isinstance(raw_text, str):
        return None, None, "parse_failed", 0
    blocks = _ANSWER_RE.findall(raw_text)
    if not blocks:
        low = raw_text.lower()
        return None, None, ("refused" if any(m in low for m in REFUSAL_MARKERS) else "parse_failed"), 0
    try:
        parsed = json.loads(blocks[0])
        major, minor = parsed.get("major_mm"), parsed.get("minor_mm")
        if (major is None or minor is None or isinstance(major, bool) or isinstance(minor, bool)
                or not isinstance(major, (int, float)) or not isinstance(minor, (int, float))):
            return None, None, "parse_failed", len(blocks)
        major, minor = float(major), float(minor)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return None, None, "parse_failed", len(blocks)
    if not (_is_valid_length(major) and _is_valid_length(minor)):
        return major, minor, "invalid_numeric", len(blocks)
    return major, minor, "ok", len(blocks)


# ---------------------------------------------------------------- results, split by group

RESULT_FIELDS = ["run_id", "seq", "case_id", "group", "qc_flag", "experiment", "pixel_spacing_mm_for_prompt",
                 "effective_spacing_at_896px_mm_assumed", "ref_major_mm", "ref_minor_mm",
                 "major_mm_pred", "minor_mm_pred", "major_abs_error_mm", "minor_abs_error_mm",
                 "major_lt_minor_flag", "call_status", "parse_status", "n_answer_blocks", "finish_reason",
                 "hit_max_new_tokens", "n_generated_tokens", "exception_type", "skip_reason", "image_sha256",
                 "raw_response"]


def build_result_rows(events: list, manifest: dict, expected_run_id: str = None) -> list:
    """One row per PLANNED call (including skipped/unfinished ones), from a reconciled log. Reference values are
    descriptive only; nothing is averaged here. Every row carries its `group`."""
    acct = verify_run_accounting(events, expected_run_id)
    plan = {c["seq"]: c for c in next(e for e in events if e.get("event") == "plan")["planned_calls"]}
    finished = {e["seq"]: e for e in events if e.get("event") == "attempt_finished"}
    skipped = {e["seq"]: e for e in events if e.get("event") == "attempt_skipped"}
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    rows = []
    for seq in sorted(plan):
        p, c = plan[seq], by_id[plan[seq]["case_id"]]
        if p["group"] != c["group"]:
            raise RunLogError(f"seq={seq}: logged group {p['group']!r} != manifest group {c['group']!r}")
        row = {"run_id": acct["run_id"], "seq": seq, "case_id": p["case_id"], "group": c["group"],
               "qc_flag": c["qc_flag"], "experiment": p["experiment"],
               "pixel_spacing_mm_for_prompt": c["pixel_spacing_mm_for_prompt"] if p["experiment"] == "B_with_spacing" else "",
               "effective_spacing_at_896px_mm_assumed": c["effective_spacing_at_896px_mm_assumed"],
               "ref_major_mm": c["ref_major_mm"], "ref_minor_mm": c["ref_minor_mm"],
               "major_mm_pred": None, "minor_mm_pred": None, "major_abs_error_mm": None, "minor_abs_error_mm": None,
               "major_lt_minor_flag": None, "call_status": acct["per_seq"][seq], "parse_status": "not_run",
               "n_answer_blocks": 0, "finish_reason": None, "hit_max_new_tokens": None, "n_generated_tokens": None,
               "exception_type": None, "skip_reason": (skipped.get(seq) or {}).get("reason"),
               "image_sha256": c["sha256"], "raw_response": None}
        r = finished.get(seq)
        if r is not None:
            row.update(finish_reason=r.get("finish_reason"), hit_max_new_tokens=r.get("hit_max_new_tokens"),
                       n_generated_tokens=r.get("n_generated_tokens"), raw_response=r.get("raw_response"),
                       exception_type=(r.get("exception") or {}).get("type"))
            if r["status"] == "generated":
                major, minor, pstatus, nblocks = parse_answer(r.get("raw_response"))
                row.update(major_mm_pred=major, minor_mm_pred=minor, parse_status=pstatus, n_answer_blocks=nblocks)
                if pstatus == "ok":
                    row.update(major_abs_error_mm=abs(major - c["ref_major_mm"]),
                               minor_abs_error_mm=abs(minor - c["ref_minor_mm"]), major_lt_minor_flag=major < minor)
            else:
                row["parse_status"] = r["status"]  # exception | interrupted | aborted
        rows.append(row)
    return rows


def select_scored_rows(rows: list) -> list:
    """The only rows eligible for any scored calculation. Exploratory rows are never returned."""
    return [r for r in rows if r["group"] == "scored"]


def require_scored_only(rows: list) -> list:
    """Guard for any future accuracy code: raises if exploratory (or unknown-group) rows are present."""
    bad = sorted({r.get("group") for r in rows if r.get("group") != "scored"})
    if bad:
        raise ValueError(f"scored calculation received rows from non-scored group(s): {bad}")
    return rows


def write_results_csv(path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_split_results(out_dir, rows: list) -> None:
    """results.csv (all rows, with a `group` column) plus separate scored / exploratory files, so nobody
    averages across groups by accident."""
    out_dir = Path(out_dir)
    write_results_csv(out_dir / "results.csv", rows)
    write_results_csv(out_dir / "results_scored.csv", [r for r in rows if r["group"] == "scored"])
    write_results_csv(out_dir / "results_exploratory.csv", [r for r in rows if r["group"] == "exploratory"])


def descriptive_summary(rows: list, accounting: dict) -> dict:
    """Counts and distinct-output checks only, reported SEPARATELY for scored and exploratory cases. Deliberately
    no MAE/MRE averages, no pooled statistics and no accuracy claim."""
    n_bad = sum(accounting[k] for k in ("n_skipped", "n_unfinished", "n_never_started", "n_interrupted", "n_aborted"))
    out = {"banner": BANNER, "run_id": accounting["run_id"], "run_finished_state": accounting["run_finished_state"],
           "accounting": {k: v for k, v in accounting.items() if k != "per_seq"},
           "complete": accounting["n_planned"] == EXPECTED_CALLS and n_bad == 0,
           "note": "Scored and exploratory cases are summarised separately and never pooled. "
                   "No accuracy aggregate is computed.", "groups": {}}
    for group in ("scored", "exploratory"):
        out["groups"][group] = {}
        for exp in EXPERIMENTS:
            sub = [r for r in rows if r["group"] == group and r["experiment"] == exp]
            counts, parse_counts = {}, {}
            for r in sub:
                counts[r["call_status"]] = counts.get(r["call_status"], 0) + 1
                parse_counts[r["parse_status"]] = parse_counts.get(r["parse_status"], 0) + 1
            ok = [r for r in sub if r["parse_status"] == "ok"]
            pairs = sorted({(r["major_mm_pred"], r["minor_mm_pred"]) for r in ok})
            out["groups"][group][exp] = {
                "n_planned": len(sub), "call_status_counts": counts, "parse_status_counts": parse_counts,
                "n_hit_max_new_tokens": sum(1 for r in sub if r["hit_max_new_tokens"]),
                "n_distinct_prediction_pairs": len(pairs), "distinct_prediction_pairs": pairs,
                "identical_prediction_across_ok_cases": len(ok) > 1 and len(pairs) == 1}
    return out
