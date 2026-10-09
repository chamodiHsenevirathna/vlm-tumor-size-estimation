"""Tests for the Phase 0 pilot bundle, verification, parsing, metadata and notebook.
Run from the repo root: python -m unittest discover -s validation/phase0_pilot5/tests -v"""

import ast
import copy
import json
import math
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PILOT = Path(__file__).resolve().parent.parent
REPO = PILOT.parent.parent
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(REPO / "src"))

import pilot_lib as pl  # noqa: E402
from ct_preprocessing import QualityThresholds  # noqa: E402

REAL_BUNDLE = PILOT / "bundle"      # only exists after build_pilot_bundle.py was run locally with the dataset
NOTEBOOK = PILOT / "phase0_pilot5.ipynb"
BUNDLE = None                        # set in setUpModule: a SYNTHETIC bundle (noise phantoms, not CT) built per test run
STALE_PNG = None                     # a blank "stale" export, also synthetic
_TMP = None
HAVE_REAL_BUNDLE = (REAL_BUNDLE / "manifest.json").is_file() and (PILOT / "pilot5_images.zip").is_file()


def make_synthetic_bundle(dest: Path, stale_dir: Path):
    """A structurally valid pilot bundle made of random-noise phantoms. These tests need no dataset and no CT data."""
    dest, stale_dir = Path(dest), Path(stale_dir)
    (dest / "images").mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    cases = []
    for k, cid in enumerate(pl.CASE_ORDER):
        n = 64 + 8 * k
        arr = np.random.default_rng(100 + k).integers(0, 256, (n, n), dtype=np.uint8)
        fname = f"case_{cid}_slice_off1024.png"
        Image.fromarray(arr).save(dest / "images" / fname)
        cases.append({
            "case_id": cid, "group": "scored" if cid in pl.SCORED_CASES else "exploratory",
            "qc_flag": "pass_basic" if cid in pl.SCORED_CASES else "ellipse_outside_bbox", "filename": fname,
            "slice_idx": 10 + k, "size_px": [n, n], "pixel_spacing_mm_for_prompt": round(0.5 + 0.05 * k, 4),
            "pixel_spacing_mm_header": 0.5 + 0.05 * k, "fov_mm": round(n * (0.5 + 0.05 * k), 2),
            "effective_spacing_at_896px_mm_assumed": round((0.5 + 0.05 * k) * n / 896, 4),
            "ref_major_mm": 21.37 + k, "ref_minor_mm": 14.82 + k, "build_checks_passed": True,
            "build_content_metrics": {"window_coverage": 0.97}, "sha256": pl.sha256_file(dest / "images" / fname),
            "image_metrics": pl.image_metrics(arr)})
    n0 = cases[0]["size_px"][0]
    stale = []
    for name, val in (("stale_white.png", 255), ("stale_black.png", 0)):
        Image.fromarray(np.full((n0, n0), val, np.uint8)).save(stale_dir / name)
        stale.append(pl.sha256_file(stale_dir / name))
    shutil.copyfile(PILOT / "pilot_lib.py", dest / "pilot_lib.py")
    manifest = pl.assemble_manifest(
        cases, stale_sha256=stale, git_info={"commit": "0" * 40, "working_tree_dirty": False},
        ct_preprocessing_sha256="0" * 64, pilot_lib_sha256=pl.sha256_file(dest / "pilot_lib.py"),
        created_utc="2026-01-01T00:00:00+00:00")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stale_dir / "stale_white.png"


def setUpModule():
    global _TMP, BUNDLE, STALE_PNG
    _TMP = tempfile.TemporaryDirectory()
    BUNDLE = Path(_TMP.name) / "bundle"
    STALE_PNG = make_synthetic_bundle(BUNDLE, Path(_TMP.name) / "stale")


def tearDownModule():
    _TMP.cleanup()


def case0_size() -> int:
    return json.loads((BUNDLE / "manifest.json").read_text())["cases"][0]["size_px"][0]


def sha(p):
    return pl.sha256_file(p)


class TempBundle:
    """Copy of the real bundle in a temp dir; helpers re-pin the manifest after a mutation."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "b"
        shutil.copytree(BUNDLE, self.dir)

    def cleanup(self):
        self._tmp.cleanup()

    @property
    def manifest(self):
        return json.loads((self.dir / "manifest.json").read_text())

    def save_manifest(self, m):
        (self.dir / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        return sha(self.dir / "manifest.json")

    def verify(self, pinned=None):
        return pl.verify_bundle(self.dir, pinned or sha(self.dir / "manifest.json"))

    def image(self, cid):
        return self.dir / "images" / f"case_{cid}_slice_off1024.png"


class BundleVerification(unittest.TestCase):
    def setUp(self):
        self.b = TempBundle()
        self.addCleanup(self.b.cleanup)

    def assertRejected(self, fragment, pinned=None):
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify(pinned)
        self.assertIn(fragment, str(cm.exception))

    def test_real_bundle_passes_and_is_ordered(self):
        out = self.b.verify()
        self.assertEqual([c["case_id"] for c in out["cases"]], ["0", "6", "8", "2", "4"])
        self.assertEqual([c["group"] for c in out["cases"]], ["scored"] * 3 + ["exploratory"] * 2)

    def test_wrong_pinned_manifest_hash(self):
        self.assertRejected("pinned", pinned="0" * 64)

    def test_modified_manifest_after_pinning(self):
        pinned = sha(self.b.dir / "manifest.json")
        m = self.b.manifest
        m["pipeline"]["window_width"] = 300.0
        self.b.save_manifest(m)
        self.assertRejected("pinned", pinned=pinned)

    def test_tampered_image_pixel(self):
        p = self.b.image("6")
        a = np.array(Image.open(p))
        a[10, 10] = 255 - a[10, 10]
        Image.fromarray(a).save(p)
        self.assertRejected("image sha256 differs from manifest")

    def test_stale_blank_image_swapped_in_with_original_manifest(self):
        shutil.copyfile(STALE_PNG, self.b.image("0"))
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify()
        msg = str(cm.exception)
        self.assertIn("KNOWN STALE", msg)
        self.assertIn("differs from manifest", msg)

    def test_stale_image_still_rejected_if_manifest_is_regenerated_to_match(self):
        shutil.copyfile(STALE_PNG, self.b.image("0"))
        m = self.b.manifest
        for c in m["cases"]:
            if c["case_id"] == "0":
                c["sha256"] = sha(self.b.image("0"))
                c["image_metrics"] = pl.image_metrics(np.array(Image.open(self.b.image("0"))))
        pinned = self.b.save_manifest(m)
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify(pinned)
        msg = str(cm.exception)
        self.assertIn("KNOWN STALE", msg)
        self.assertIn("n_unique_levels", msg)  # independent generic defence also fires

    def test_blank_image_with_manifest_regenerated(self):
        n0 = case0_size()
        Image.fromarray(np.full((n0, n0), 255, np.uint8)).save(self.b.image("0"))
        m = self.b.manifest
        for c in m["cases"]:
            if c["case_id"] == "0":
                c["sha256"] = sha(self.b.image("0"))
                c["image_metrics"] = pl.image_metrics(np.full((n0, n0), 255, np.uint8))
        pinned = self.b.save_manifest(m)
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify(pinned)
        self.assertIn("frac_ge_254", str(cm.exception))

    def test_extra_file_rejected(self):
        (self.b.dir / "images" / "case_0_slice.png").write_bytes(STALE_PNG.read_bytes())
        self.assertRejected("unexpected=['images/case_0_slice.png']")

    def test_missing_image_rejected(self):
        self.b.image("8").unlink()
        self.assertRejected("missing=")

    def test_rgb_image_rejected(self):
        p = self.b.image("2")
        Image.open(p).convert("RGB").save(p)
        self.assertRejected("expected 8-bit greyscale")

    def test_wrong_size_rejected(self):
        p = self.b.image("2")
        Image.open(p).resize((120, 120)).save(p)
        self.assertRejected("size")

    def test_pilot_lib_tampered(self):
        (self.b.dir / "pilot_lib.py").write_text((self.b.dir / "pilot_lib.py").read_text() + "\n# x\n")
        self.assertRejected("pilot_lib.py sha256 differs")

    def test_pinned_pilot_lib_hash_enforced(self):
        with self.assertRaises(pl.BundleVerificationError) as cm:
            pl.verify_bundle(self.b.dir, sha(self.b.dir / "manifest.json"), "f" * 64)
        self.assertIn("pinned", str(cm.exception))

    def _mutate(self, fn, fragment):
        m = self.b.manifest
        fn(m)
        pinned = self.b.save_manifest(m)
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify(pinned)
        self.assertIn(fragment, str(cm.exception))

    def test_offset_must_be_1024(self):
        self._mutate(lambda m: m["pipeline"].__setitem__("intensity_offset", 1000.0), "intensity_offset")

    def test_offset_must_be_marked_unverified(self):
        self._mutate(lambda m: m["pipeline"].__setitem__("offset_status", "verified"), "UNVERIFIED")

    def test_groups_must_match(self):
        self._mutate(lambda m: m["groups"].__setitem__("scored", ["0", "2", "6"]), "groups must be")

    def test_case_group_must_match(self):
        def f(m):
            m["cases"][3]["group"] = "scored"
        self._mutate(f, "group 'scored' != 'exploratory'")

    def test_prompt_hash_must_match(self):
        self._mutate(lambda m: m["prompts"].__setitem__("prompt_a_sha256", "0" * 64), "prompt_a_sha256")

    def test_required_limitations(self):
        def f(m):
            m["limitations"] = [x for x in m["limitations"] if x["id"] != "effective_spacing_mismatch"]
        self._mutate(f, "effective_spacing_mismatch")

    def test_stale_blocklist_must_not_be_empty(self):
        self._mutate(lambda m: m.__setitem__("known_stale_sha256", []), "stale-image guard disabled")

    def test_reference_must_be_positive(self):
        def f(m):
            m["cases"][0]["ref_major_mm"] = float("nan")
        self._mutate(f, "ref_major_mm")

    def test_duplicate_images_rejected(self):
        shutil.copyfile(self.b.image("0"), self.b.image("8"))
        m = self.b.manifest
        for c in m["cases"]:
            if c["case_id"] == "8":
                c["sha256"] = sha(self.b.image("0"))
        pinned = self.b.save_manifest(m)
        with self.assertRaises(pl.BundleVerificationError) as cm:
            self.b.verify(pinned)
        self.assertIn("identical to case", str(cm.exception))

    def test_malformed_manifest_fails_closed(self):
        (self.b.dir / "manifest.json").write_text("{not json")
        self.assertRejected("not valid JSON", pinned=sha(self.b.dir / "manifest.json"))

    def test_images_are_clean_model_inputs(self):
        for cid in pl.CASE_ORDER:
            a = np.array(Image.open(self.b.image(cid)))
            self.assertEqual(a.ndim, 2)  # greyscale only: no colour overlay can be present
            self.assertEqual(a.dtype, np.uint8)
            self.assertFalse(pl.generic_image_failures(pl.image_metrics(a)))


class ZipBundle(unittest.TestCase):
    def test_deterministic_zip_roundtrip_verifies_as_the_notebook_would(self):
        with tempfile.TemporaryDirectory() as t:
            z1, z2 = Path(t) / "a.zip", Path(t) / "b.zip"
            pl.write_deterministic_zip(BUNDLE, z1)
            pl.write_deterministic_zip(BUNDLE, z2)
            self.assertEqual(z1.read_bytes(), z2.read_bytes())  # same content -> same bytes
            out = Path(t) / "x"
            with zipfile.ZipFile(z1) as zf:
                self.assertEqual(sorted(zf.namelist()),
                                 sorted(p.relative_to(BUNDLE).as_posix() for p in BUNDLE.rglob("*") if p.is_file()))
                zf.extractall(out)
            pl.verify_bundle(out, sha(BUNDLE / "manifest.json"), sha(BUNDLE / "pilot_lib.py"))


class PromptsAndThresholds(unittest.TestCase):
    def test_prompts_identical_to_milestone6_notebook(self):
        with open(REPO / "notebooks" / "milestone6_vlm_batch5.ipynb") as f:
            nb = json.load(f)
        ns = {}
        exec("".join(nb["cells"][16]["source"]).split("print(")[0], ns)  # trusted repo code, defs only
        self.assertEqual(pl.prompt_a(), ns["prompt_a"]())
        for s in (0.5859, 0.7148, 0.5898):
            self.assertEqual(pl.prompt_b(s), ns["prompt_b"](s))
        self.assertEqual(pl.prompt_hashes()["prompt_a_sha256"][:16], "e750198d44e4f67f")
        self.assertEqual(pl.sha256_text(pl.prompt_b("{S}"))[:16], "ac324c1bb484b094")

    def test_prompts_never_contain_reference_values(self):
        m = json.loads((BUNDLE / "manifest.json").read_text())
        for c in m["cases"]:
            for text in (pl.prompt_a(), pl.prompt_b(c["pixel_spacing_mm_for_prompt"])):
                self.assertNotIn(str(c["ref_major_mm"]), text)
                self.assertNotIn(str(c["ref_minor_mm"]), text)

    def test_generic_thresholds_match_ct_preprocessing(self):
        t = QualityThresholds()
        self.assertEqual((pl.MIN_UNIQUE_LEVELS, pl.MIN_GREY_STD, pl.MIN_ENTROPY_BITS, pl.MAX_SATURATED_FRACTION),
                         (t.min_unique_levels, t.min_grey_std, t.min_entropy_bits, t.max_saturated_fraction))


class Parsing(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(pl.parse_answer('<answer>{"major_mm": 15, "minor_mm": 10.5}</answer>'), (15.0, 10.5, "ok", 1))

    def test_ok_with_surrounding_text_and_whitespace(self):
        r = pl.parse_answer('Sure.\n<answer> {"major_mm": 12, "minor_mm": 8} </answer>\nDone')
        self.assertEqual(r[2], "ok")

    def test_parse_failures_are_classified_not_dropped(self):
        for text, status in [("no tags here", "parse_failed"), ("", "parse_failed"),
                             ('<answer>{"major_mm": 1}</answer>', "parse_failed"),
                             ('<answer>{not json}</answer>', "parse_failed"),
                             ('<answer>{"major_mm": "big", "minor_mm": 2}</answer>', "parse_failed"),
                             ('<answer>{"major_mm": true, "minor_mm": 2}</answer>', "parse_failed"),
                             ('<answer>[1,2]</answer>', "parse_failed"),
                             ("I cannot provide a diagnosis.", "refused"),
                             ('<answer>{"major_mm": 15, "minor_mm"', "parse_failed")]:
            self.assertEqual(pl.parse_answer(text)[2], status, text)

    def test_invalid_numeric_keeps_values_visible(self):
        for body, vals in [('{"major_mm": 0, "minor_mm": 5}', (0.0, 5.0)), ('{"major_mm": -3, "minor_mm": 5}', (-3.0, 5.0)),
                           ('{"major_mm": NaN, "minor_mm": 5}', None), ('{"major_mm": Infinity, "minor_mm": 5}', None)]:
            major, minor, status, _ = pl.parse_answer(f"<answer>{body}</answer>")
            self.assertEqual(status, "invalid_numeric")
            if vals:
                self.assertEqual((major, minor), vals)

    def test_non_string_input(self):
        self.assertEqual(pl.parse_answer(None)[2], "parse_failed")

    def test_multiple_answer_blocks_counted_first_used(self):
        r = pl.parse_answer('<answer>{"major_mm": 1, "minor_mm": 2}</answer><answer>{"major_mm": 9, "minor_mm": 9}</answer>')
        self.assertEqual((r[0], r[1], r[3]), (1.0, 2.0, 2))


class MetadataSchema(unittest.TestCase):
    ENV = {"model_id": pl.MODEL_ID, "model_revision": "unresolved", "quantization": "bf16", "device": "cuda:T4",
           "python_version": "3.12", "torch_version": "2.x", "transformers_version": "4.x", "accelerate_version": "1.x",
           "bitsandbytes_version": "n/a", "pillow_version": "11.3.0", "processor_image_config": {"size": {"height": 896}},
           "chat_template_format": "x"}

    def meta(self):
        m = json.loads((BUNDLE / "manifest.json").read_text())
        return pl.new_run_metadata(m, "a" * 64, copy.deepcopy(self.ENV), "20260101T000000Z-1a2b3c4d", "t")

    def test_valid(self):
        meta = self.meta()
        pl.validate_run_metadata(meta)
        self.assertEqual(meta["preprocessing"]["intensity_offset"], 1024.0)
        self.assertTrue(meta["preprocessing"]["offset_status"].startswith("UNVERIFIED"))
        self.assertEqual(meta["generation_settings"], {"do_sample": False, "num_beams": 1, "max_new_tokens": 300})
        self.assertEqual(meta["preprocessing"]["window_center_hu"], 40.0)
        self.assertEqual(meta["preprocessing"]["window_width_hu"], 400.0)
        self.assertEqual(meta["prompts"]["version"], "m6-prompts-v1")
        self.assertEqual([i["case_id"] for i in meta["inputs"]], pl.CASE_ORDER)
        self.assertTrue(all("size_px" in i and "sha256" in i for i in meta["inputs"]))
        json.dumps(meta)  # serialisable

    def test_rejections(self):
        def bad(mut, frag):
            m = self.meta()
            mut(m)
            with self.assertRaises(pl.MetadataError) as cm:
                pl.validate_run_metadata(m)
            self.assertIn(frag, str(cm.exception))
        bad(lambda m: m["preprocessing"].__setitem__("offset_status", "verified"), "UNVERIFIED")
        bad(lambda m: m["preprocessing"].__setitem__("intensity_offset", 1000.0), "1024")
        bad(lambda m: m["environment"].pop("torch_version"), "torch_version")
        bad(lambda m: m["environment"].__setitem__("processor_image_config", {}), "processor_image_config")
        bad(lambda m: m["model"].__setitem__("model_revision", ""), "model_revision")
        bad(lambda m: m["model"].__setitem__("model_id", "other/model"), "model_id")
        bad(lambda m: m["generation_settings"].__setitem__("do_sample", True), "generation_settings")
        bad(lambda m: m["generation_settings"].__setitem__("max_new_tokens", 2000), "generation_settings")
        bad(lambda m: m["prompts"].__setitem__("version", "x"), "prompt version")
        bad(lambda m: m.__setitem__("limitations", []), "accuracy")
        bad(lambda m: m["inputs"].pop(), "five cases")


class CallLogging(unittest.TestCase):
    def test_jsonl_roundtrip_preserves_everything(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.jsonl"
            recs = [{"seq": 1, "raw_response": "<answer>x</answer>\nline2", "generated_token_ids": [1, 2, 3],
                     "exception": None, "unicode": "mm µ"},
                    {"seq": 2, "status": "exception", "exception": {"type": "OOM", "traceback": "Traceback\n..."}}]
            for r in recs:
                pl.append_jsonl(p, r)
            self.assertEqual(pl.read_jsonl(p), recs)
            self.assertEqual(len(p.read_text().splitlines()), 2)

    def test_corrupt_line_is_reported_not_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.jsonl"
            p.write_text('{"a": 1}\n{"b": \n')
            with self.assertRaises(ValueError):
                pl.read_jsonl(p)

    def test_finish_reason(self):
        self.assertEqual(pl.determine_finish_reason([5, 6, 1], [1, 106]), "eos")
        self.assertEqual(pl.determine_finish_reason(list(range(300)), [1]), "length")
        self.assertEqual(pl.determine_finish_reason([5, 6], [1]), "unknown")
        self.assertEqual(pl.determine_finish_reason([], []), "unknown")


GOOD = {"raw_response": '<answer>{"major_mm": 15, "minor_mm": 10}</answer>',
        "raw_response_with_special_tokens": '<answer>{"major_mm": 15, "minor_mm": 10}</answer><end_of_turn>',
        "generated_token_ids": [11, 12, 1], "n_generated_tokens": 3, "input_token_count": 300,
        "image_token_count": 256, "finish_reason": "eos", "eos_token_ids": [1, 106]}


class FakeImage:
    size = (138, 138)


def _load(case):
    return FakeImage()


class RunFixture:
    """A fresh run folder with persisted metadata (what Step 6 leaves behind)."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest = json.loads((BUNDLE / "manifest.json").read_text())
        self.run_id = pl.new_run_id()
        self.out = Path(self._tmp.name) / f"pilot5_run_{self.run_id}"
        self.out.mkdir()
        self.meta = pl.new_run_metadata(self.manifest, "a" * 64, copy.deepcopy(MetadataSchema.ENV), self.run_id, pl.utc_now())
        pl.atomic_write_json(self.out / "run_metadata.json", self.meta)

    def cleanup(self):
        self._tmp.cleanup()

    def listing(self):
        return sorted(p.name for p in self.out.iterdir())

    def gate(self):
        pl.assert_ready_for_inference(self.out, self.meta, self.run_id)

    def run(self, generate_fn, **kw):
        return pl.run_planned_calls(self.manifest["cases"], self.out, self.run_id, _load, generate_fn, **kw)

    def events(self):
        return pl.read_jsonl(self.out / "raw_responses.jsonl")

    def persisted(self):
        return json.loads((self.out / "run_metadata.json").read_text())


class CountingGenerate:
    def __init__(self, fail_on=(), exc=RuntimeError, result=GOOD):
        self.calls, self.fail_on, self.exc, self.result = 0, set(fail_on), exc, result

    def __call__(self, image, prompt):
        self.calls += 1
        if self.calls in self.fail_on:
            raise self.exc(f"stub failure on call {self.calls}")
        return dict(self.result)


class RunGate(unittest.TestCase):
    def setUp(self):
        self.f = RunFixture()
        self.addCleanup(self.f.cleanup)

    def assertBlocked(self, fragment=None):
        before = self.f.listing()
        gen = CountingGenerate()
        with self.assertRaises(pl.RunGateError) as cm:
            self.f.gate()
        if fragment:
            self.assertIn(fragment, str(cm.exception))
        self.assertEqual(self.f.listing(), before, "a blocked gate must not create any file (no lock, no log)")
        with self.assertRaises(pl.RunGateError):  # and inference itself still cannot start
            self.f.run(gen)
        self.assertEqual(gen.calls, 0, "model must never be called when the gate blocks")
        self.assertEqual(self.f.listing(), before)

    def test_valid_run_passes_and_claims_folder(self):
        self.f.gate()
        self.assertIn("inference.lock", self.f.listing())
        with self.assertRaises(pl.RunGateError):  # second gate on the same folder (rerun of Step 7)
            self.f.gate()

    def test_metadata_missing(self):
        (self.f.out / "run_metadata.json").unlink()
        self.assertBlocked("not persisted")

    def test_in_memory_metadata_missing(self):
        self.f.meta = None
        self.assertBlocked("run metadata missing")

    def test_invalid_in_memory_and_persisted_metadata(self):
        self.f.meta["preprocessing"]["offset_status"] = "verified"
        pl.atomic_write_json(self.f.out / "run_metadata.json", self.f.meta)  # identical but invalid
        self.assertBlocked("UNVERIFIED")

    def test_invalid_persisted_only(self):
        bad = copy.deepcopy(self.f.meta)
        bad["environment"].pop("torch_version")
        pl.atomic_write_json(self.f.out / "run_metadata.json", bad)
        self.assertBlocked("persisted metadata invalid")

    def test_metadata_mismatch_in_memory_changed(self):
        self.f.meta["model"]["quantization"] = "nf4-4bit"  # persisted file still says bf16
        self.assertBlocked("differs from the in-memory metadata")

    def test_metadata_mismatch_persisted_changed(self):
        changed = copy.deepcopy(self.f.meta)
        changed["environment"]["torch_version"] = "other"
        pl.atomic_write_json(self.f.out / "run_metadata.json", changed)
        self.assertBlocked("differs from the in-memory metadata")

    def test_persisted_metadata_corrupt(self):
        (self.f.out / "run_metadata.json").write_text("{not json")
        self.assertBlocked("not valid JSON")

    def test_status_must_be_started(self):
        pl.update_run_metadata(self.f.out, {"status": "completed"})
        self.f.meta["status"] = "completed"
        self.assertBlocked("expected 'started'")

    def test_run_id_and_folder_must_match(self):
        other = pl.new_run_id()
        self.f.meta["run_id"] = other
        pl.atomic_write_json(self.f.out / "run_metadata.json", self.f.meta)
        self.assertBlocked("run_id")

    def test_folder_name_must_match_run_id(self):
        new = self.f.out.parent / "pilot5_run_other"
        self.f.out.rename(new)
        self.f.out = new
        self.assertBlocked("folder name")

    def test_invalid_run_id(self):
        with self.assertRaises(pl.RunGateError):
            pl.assert_ready_for_inference(self.f.out, self.f.meta, "20260101T000000Z")

    def test_existing_artifacts_block_inference(self):
        for name in ("raw_responses.jsonl", "run.log", "results.csv", "summary.json", "parse_failures.jsonl",
                     "inference.lock", "stray.txt"):
            f = RunFixture()
            try:
                (f.out / name).write_text("x")
                before = f.listing()
                gen = CountingGenerate()
                with self.assertRaises(pl.RunGateError) as cm:
                    f.gate()
                self.assertIn(name, str(cm.exception))
                with self.assertRaises(pl.RunGateError):
                    f.run(gen)
                self.assertEqual(gen.calls, 0)
                self.assertEqual(f.listing(), before)
            finally:
                f.cleanup()

    def test_existing_raw_log_refused_even_with_lock(self):
        self.f.gate()
        (self.f.out / "raw_responses.jsonl").write_text("old\n")
        gen = CountingGenerate()
        with self.assertRaises(pl.RunGateError):
            self.f.run(gen)
        self.assertEqual(gen.calls, 0)
        self.assertEqual((self.f.out / "raw_responses.jsonl").read_text(), "old\n")  # nothing appended

    def test_planted_lock_file_is_not_an_approval(self):
        f = RunFixture()
        try:
            (f.out / "inference.lock").write_text(json.dumps({"run_id": f.run_id}))
            gen = CountingGenerate()
            with self.assertRaises(pl.RunGateError) as cm:
                f.run(gen)
            self.assertIn("no gate approval", str(cm.exception))
            self.assertEqual(gen.calls, 0)
            self.assertFalse((f.out / "raw_responses.jsonl").exists())
        finally:
            f.cleanup()

    def test_gate_approval_is_single_use_and_bound_to_the_metadata(self):
        self.f.gate()
        pl.update_run_metadata(self.f.out, {"status": "started", "tampered": True})  # changed after approval
        gen = CountingGenerate()
        with self.assertRaises(pl.RunGateError) as cm:
            self.f.run(gen)
        self.assertIn("changed after the gate", str(cm.exception))
        self.assertEqual(gen.calls, 0)
        with self.assertRaises(pl.RunGateError):  # approval already consumed
            self.f.run(gen)
        self.assertEqual(gen.calls, 0)

    def test_inference_without_gate_refused(self):
        gen = CountingGenerate()
        with self.assertRaises(pl.RunGateError):
            self.f.run(gen)
        self.assertEqual(gen.calls, 0)

    def test_metadata_run_id_format_validated(self):
        bad = copy.deepcopy(self.f.meta)
        bad["run_id"] = "20260101T000000Z"
        with self.assertRaises(pl.MetadataError):
            pl.validate_run_metadata(bad)
        self.assertNotEqual(pl.new_run_id(), pl.new_run_id())


class RunExecution(unittest.TestCase):
    def setUp(self):
        self.f = RunFixture()
        self.addCleanup(self.f.cleanup)
        self.f.gate()

    def test_completed_run_is_fully_recorded(self):
        out = self.f.run(CountingGenerate())
        self.assertEqual(out["state"], "completed")
        ev = self.f.events()
        kinds = [e["event"] for e in ev]
        self.assertEqual(kinds.count("plan"), 1)
        self.assertEqual(kinds.count("attempt_started"), 10)
        self.assertEqual(kinds.count("attempt_finished"), 10)
        self.assertEqual(kinds[-1], "run_finished")
        self.assertTrue(all(e["run_id"] == self.f.run_id for e in ev))
        acct = pl.verify_run_accounting(ev, self.f.run_id)
        self.assertTrue(acct["all_accounted"])
        self.assertEqual(acct["n_generated"], 10)
        self.assertEqual(self.f.persisted()["status"], "completed")
        fin = [e for e in ev if e["event"] == "attempt_finished"][0]
        self.assertEqual(fin["generated_token_ids"], [11, 12, 1])
        self.assertEqual(fin["raw_response_with_special_tokens"], GOOD["raw_response_with_special_tokens"])
        self.assertEqual(fin["image_size_px"], [138, 138])

    def test_attempt_is_durably_logged_before_the_model_runs(self):
        raw = self.f.out / "raw_responses.jsonl"
        seen = []

        def gen(image, prompt):
            ev = pl.read_jsonl(raw)  # what is on disk at the moment the "model" is called
            seq = max(e["seq"] for e in ev if e["event"] == "attempt_started")
            seen.append((seq, any(e["event"] == "attempt_finished" and e["seq"] == seq for e in ev),
                         ev[-1]["event"], ev[-1].get("prompt_text") == prompt))
            return dict(GOOD)
        self.f.run(gen)
        self.assertEqual(len(seen), 10)
        self.assertTrue(all(not finished and last == "attempt_started" and same for _, finished, last, same in seen))

    def test_records_are_fsynced(self):
        import unittest.mock as mock
        with mock.patch.object(pl.os, "fsync", wraps=pl.os.fsync) as fs:
            self.f.run(CountingGenerate())
        n_lines = len(self.f.events())
        self.assertGreaterEqual(fs.call_count, n_lines)

    def test_early_stop_accounts_for_every_remaining_planned_call(self):
        gen = CountingGenerate(fail_on=range(1, 11))
        out = self.f.run(gen)
        self.assertEqual(out["state"], "stopped_early")
        self.assertEqual(gen.calls, 3)
        ev = self.f.events()
        acct = pl.verify_run_accounting(ev, self.f.run_id)
        self.assertEqual((acct["n_exception"], acct["n_skipped"], acct["n_planned"]), (3, 7, 10))
        self.assertTrue(acct["all_accounted"])
        skipped = [e for e in ev if e["event"] == "attempt_skipped"]
        self.assertEqual([e["seq"] for e in skipped], list(range(4, 11)))
        self.assertTrue(all(e["reason"] == "stopped_after_3_consecutive_failures" for e in skipped))
        exc = [e for e in ev if e["event"] == "attempt_finished"][0]["exception"]
        self.assertEqual(exc["type"], "RuntimeError")
        self.assertIn("stub failure", exc["traceback"])
        self.assertIn("RuntimeError", exc["traceback"])
        self.assertEqual(self.f.persisted()["status"], "stopped_early")
        rows = pl.build_result_rows(ev, self.f.manifest, self.f.run_id)
        self.assertEqual([r["call_status"] for r in rows], ["exception"] * 3 + ["skipped"] * 7)
        self.assertEqual(len(rows), 10)

    def test_intermittent_failures_do_not_stop_the_run(self):
        out = self.f.run(CountingGenerate(fail_on={2, 4, 6, 8}))
        self.assertEqual(out["state"], "completed_with_failures")
        acct = pl.verify_run_accounting(self.f.events())
        self.assertEqual((acct["n_exception"], acct["n_generated"], acct["n_skipped"]), (4, 6, 0))

    def test_image_load_failure_is_recorded_as_an_attempt(self):
        def bad_load(case):
            if case["case_id"] == "0":
                raise OSError("cannot open image")
            return FakeImage()
        pl.run_planned_calls(self.f.manifest["cases"], self.f.out, self.f.run_id, bad_load, CountingGenerate())
        ev = self.f.events()
        first = [e for e in ev if e["event"] == "attempt_finished"][0]
        self.assertEqual((first["status"], first["exception"]["type"]), ("exception", "OSError"))

    def test_keyboard_interrupt_is_recorded_preserved_and_reraised(self):
        gen = CountingGenerate(fail_on={4}, exc=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.f.run(gen)
        ev = self.f.events()
        acct = pl.verify_run_accounting(ev, self.f.run_id)
        self.assertEqual(acct["per_seq"][4], "interrupted")
        self.assertEqual([acct["per_seq"][i] for i in (1, 2, 3)], ["generated"] * 3)
        self.assertEqual([acct["per_seq"][i] for i in range(5, 11)], ["skipped"] * 6)
        self.assertTrue(acct["all_accounted"])
        started4 = [e for e in ev if e["event"] == "attempt_started" and e["seq"] == 4]
        fin4 = [e for e in ev if e["event"] == "attempt_finished" and e["seq"] == 4][0]
        self.assertEqual(len(started4), 1)
        self.assertEqual((fin4["status"], fin4["exception"]["type"]), ("interrupted", "KeyboardInterrupt"))
        self.assertTrue(all(e["reason"] == "interrupted" for e in ev if e["event"] == "attempt_skipped"))
        self.assertEqual(ev[-1]["state"], "interrupted")
        self.assertEqual(self.f.persisted()["status"], "interrupted")
        rows = pl.build_result_rows(ev, self.f.manifest, self.f.run_id)
        self.assertEqual((rows[3]["call_status"], rows[3]["parse_status"]), ("interrupted", "interrupted"))
        self.assertEqual(len(rows), 10)
        # the first three calls' raw responses are intact
        self.assertEqual(rows[0]["raw_response"], GOOD["raw_response"])

    def test_interrupt_during_image_load(self):
        def load(case):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            pl.run_planned_calls(self.f.manifest["cases"], self.f.out, self.f.run_id, load, CountingGenerate())
        acct = pl.verify_run_accounting(self.f.events())
        self.assertEqual((acct["n_interrupted"], acct["n_skipped"]), (1, 9))

    def test_other_base_exceptions_are_recorded_as_aborted_and_reraised(self):
        with self.assertRaises(SystemExit):
            self.f.run(CountingGenerate(fail_on={2}, exc=SystemExit))
        acct = pl.verify_run_accounting(self.f.events())
        self.assertEqual((acct["n_aborted"], acct["n_skipped"], acct["n_generated"]), (1, 8, 1))
        self.assertEqual(self.f.persisted()["status"], "aborted")

    def test_hard_kill_leaves_unfinished_attempt_visible(self):
        """Simulate SIGKILL after the write-ahead record of call 2: no terminal record exists for it."""
        raw = self.f.out / "raw_responses.jsonl"
        plan = pl.plan_calls(self.f.manifest["cases"])
        rid = self.f.run_id
        pl.append_jsonl(raw, {"event": "plan", "run_id": rid, "n_planned": 10, "planned_calls": plan})
        pl.append_jsonl(raw, {"event": "attempt_started", "run_id": rid, "seq": 1, **{k: plan[0][k] for k in ("case_id", "group", "experiment")}})
        pl.append_jsonl(raw, {"event": "attempt_finished", "run_id": rid, "seq": 1, "status": "generated", "raw_response": GOOD["raw_response"], "finish_reason": "eos"})
        pl.append_jsonl(raw, {"event": "attempt_started", "run_id": rid, "seq": 2, **{k: plan[1][k] for k in ("case_id", "group", "experiment")}})
        ev = pl.read_jsonl(raw)
        acct = pl.verify_run_accounting(ev, rid)
        self.assertEqual((acct["n_unfinished"], acct["n_never_started"], acct["all_accounted"]), (1, 8, False))
        rows = pl.build_result_rows(ev, self.f.manifest, rid)
        self.assertEqual([r["call_status"] for r in rows[:3]], ["generated", "unfinished", "never_started"])
        self.assertFalse(pl.descriptive_summary(rows, acct)["complete"])

    def test_rerun_cannot_append_a_second_pass(self):
        self.f.run(CountingGenerate())
        raw = self.f.out / "raw_responses.jsonl"
        before = raw.read_bytes()
        gen = CountingGenerate()
        with self.assertRaises(pl.RunGateError):
            self.f.gate()
        with self.assertRaises(pl.RunGateError):
            self.f.run(gen)
        self.assertEqual(gen.calls, 0)
        self.assertEqual(raw.read_bytes(), before)

    def test_logs_that_mix_runs_or_plans_are_rejected(self):
        self.f.run(CountingGenerate())
        ev = self.f.events()
        other = [dict(e, run_id=pl.new_run_id()) for e in ev[:3]]
        with self.assertRaises(pl.RunLogError):
            pl.verify_run_accounting(ev + other)
        with self.assertRaises(pl.RunLogError):
            pl.verify_run_accounting(ev + [e for e in ev if e["event"] == "plan"])  # second plan = second pass
        with self.assertRaises(pl.RunLogError):
            pl.verify_run_accounting(ev + [e for e in ev if e["event"] == "attempt_started"][:1])  # duplicate
        with self.assertRaises(pl.RunLogError):
            pl.verify_run_accounting(ev + [dict(ev[1], seq=99)])
        with self.assertRaises(pl.RunLogError):
            pl.verify_run_accounting(ev, expected_run_id=pl.new_run_id())


class ResultsAndSummary(unittest.TestCase):
    def setUp(self):
        self.f = RunFixture()
        self.addCleanup(self.f.cleanup)
        self.f.gate()

    def finish(self, gen):
        out = self.f.run(gen)
        ev = self.f.events()
        acct = pl.verify_run_accounting(ev, self.f.run_id)
        rows = pl.build_result_rows(ev, self.f.manifest, self.f.run_id)
        return out, ev, acct, rows

    def test_rows_cover_every_planned_call_with_group_and_parse_status(self):
        responses = {1: GOOD, 2: dict(GOOD, raw_response="garbled"), 3: dict(GOOD, raw_response="I cannot provide a diagnosis"),
                     4: dict(GOOD, raw_response='<answer>{"major_mm": 0, "minor_mm": 5}</answer>'),
                     5: dict(GOOD, raw_response='<answer>{"major_mm": 1', finish_reason="length")}

        def gen(image, prompt):
            gen.n += 1
            return dict(responses.get(gen.n, GOOD))
        gen.n = 0
        _, _, _, rows = self.finish(gen)
        self.assertEqual(len(rows), 10)
        self.assertEqual([r["parse_status"] for r in rows[:5]], ["ok", "parse_failed", "refused", "invalid_numeric", "parse_failed"])
        self.assertEqual(rows[1]["raw_response"], "garbled")
        self.assertTrue(rows[4]["hit_max_new_tokens"])
        self.assertIsNone(rows[1]["major_abs_error_mm"])
        self.assertAlmostEqual(rows[0]["major_abs_error_mm"], abs(15 - rows[0]["ref_major_mm"]))
        self.assertEqual([r["group"] for r in rows], ["scored"] * 6 + ["exploratory"] * 4)
        self.assertEqual(rows[0]["pixel_spacing_mm_for_prompt"], "")
        self.assertNotEqual(rows[1]["pixel_spacing_mm_for_prompt"], "")

    def test_summary_is_split_by_group_and_never_pooled(self):
        _, _, acct, rows = self.finish(CountingGenerate())
        s = pl.descriptive_summary(rows, acct)
        self.assertEqual(set(s["groups"]), {"scored", "exploratory"})
        for exp in pl.EXPERIMENTS:
            self.assertEqual(s["groups"]["scored"][exp]["n_planned"], 3)
            self.assertEqual(s["groups"]["exploratory"][exp]["n_planned"], 2)
        self.assertNotIn("per_experiment", s)
        self.assertTrue(s["complete"])
        text = json.dumps(s).lower()
        for forbidden in ("mae", "mre", "average", "mean_"):
            self.assertNotIn(forbidden, text)
        self.assertIn("never pooled", s["note"])
        self.assertIn("does not validate tumor-measurement accuracy", s["banner"].lower().replace("not validate", "not validate"))

    def test_exploratory_failures_do_not_leak_into_scored_summary(self):
        # calls 7,8,9 (all exploratory) fail -> early stop; call 10 skipped. Scored group must be untouched.
        _, _, acct, rows = self.finish(CountingGenerate(fail_on={7, 8, 9}))
        s = pl.descriptive_summary(rows, acct)
        for exp in pl.EXPERIMENTS:
            self.assertEqual(s["groups"]["scored"][exp]["call_status_counts"], {"generated": 3})
        ex = s["groups"]["exploratory"]
        self.assertEqual(ex["A_no_spacing"]["call_status_counts"], {"exception": 2})
        self.assertEqual(ex["B_with_spacing"]["call_status_counts"], {"exception": 1, "skipped": 1})
        self.assertFalse(s["complete"])

    def test_split_csv_files_keep_groups_apart(self):
        _, _, _, rows = self.finish(CountingGenerate())
        pl.write_split_results(self.f.out, rows)
        import csv as _csv
        def read(name):
            with open(self.f.out / name, newline="") as fh:
                return list(_csv.DictReader(fh))
        self.assertEqual({r["group"] for r in read("results_scored.csv")}, {"scored"})
        self.assertEqual({r["group"] for r in read("results_exploratory.csv")}, {"exploratory"})
        self.assertEqual((len(read("results.csv")), len(read("results_scored.csv")), len(read("results_exploratory.csv"))), (10, 6, 4))
        self.assertEqual(list(read("results.csv")[0].keys()), pl.RESULT_FIELDS)

    def test_scored_guards(self):
        _, _, _, rows = self.finish(CountingGenerate())
        scored = pl.select_scored_rows(rows)
        self.assertEqual(len(scored), 6)
        self.assertTrue(all(r["group"] == "scored" for r in scored))
        self.assertEqual(pl.require_scored_only(scored), scored)
        with self.assertRaises(ValueError):
            pl.require_scored_only(rows)
        with self.assertRaises(ValueError):
            pl.require_scored_only([dict(rows[0], group="unknown")])

    def test_group_disagreement_between_log_and_manifest_is_rejected(self):
        self.f.run(CountingGenerate())
        manifest = copy.deepcopy(self.f.manifest)
        manifest["cases"][4]["group"] = "scored"  # case 2 mislabelled in the manifest
        with self.assertRaises(pl.RunLogError):
            pl.build_result_rows(self.f.events(), manifest, self.f.run_id)

    def test_incomplete_run_is_flagged_not_complete(self):
        gen = CountingGenerate(fail_on={5}, exc=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.f.run(gen)
        ev = self.f.events()
        acct = pl.verify_run_accounting(ev)
        s = pl.descriptive_summary(pl.build_result_rows(ev, self.f.manifest), acct)
        self.assertFalse(s["complete"])
        self.assertEqual(s["run_finished_state"], "interrupted")


class NotebookStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(NOTEBOOK) as f:
            cls.nb = json.load(f)
        cls.text = "\n".join("".join(c["source"]) for c in cls.nb["cells"])

    def test_valid_notebook_without_outputs(self):
        self.assertEqual(self.nb["nbformat"], 4)
        for c in self.nb["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])

    def test_code_cells_parse_as_python(self):
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
            try:
                ast.parse(src)
            except SyntaxError as exc:  # pragma: no cover
                self.fail(f"cell {i} does not parse: {exc}")

    def test_fail_closed_gates_present(self):
        for needle in ("verify_bundle", "BUNDLE_VERIFIED", "WRONG OR STALE BUNDLE", "already exists",
                       "refusing to run on CPU", "validate_run_metadata", "exist_ok=False",
                       "assert_ready_for_inference", "run_planned_calls", "atomic_write_json", "run_meta = None"):
            self.assertIn(needle, self.text)
        t = self.text
        self.assertGreaterEqual(t.count("verify_bundle("), 2)  # bootstrap + again right before inference
        self.assertLess(t.index("verify_bundle("), t.index("model.generate"))
        self.assertLess(t.index("pilot_lib.assert_ready_for_inference(OUT_DIR, run_meta, RUN_ID)"), t.index("run_planned_calls("))
        self.assertLess(t.index("pilot_lib.assert_ready_for_inference(OUT_DIR, run_meta, RUN_ID)"), t.index("model.generate"))
        # Step 6 ordering: validate -> persist -> only then expose run_meta
        self.assertLess(t.index("pilot_lib.validate_run_metadata(_candidate)"), t.index('atomic_write_json(OUT_DIR / "run_metadata.json"'))
        self.assertLess(t.index('atomic_write_json(OUT_DIR / "run_metadata.json"'), t.index("run_meta = _candidate"))
        self.assertLess(t.index("run_meta = None"), t.index("RUN_ID = pilot_lib.new_run_id()"))

    def test_notebook_has_no_own_loop_that_could_bypass_write_ahead_logging(self):
        self.assertEqual(self.text.count("model.generate("), 1)
        self.assertNotIn("append_jsonl", self.text)           # only the library writes the call log
        self.assertNotIn("for case in bundle", self.text)

    def test_library_write_ahead_and_interrupt_handling(self):
        src = (PILOT / "pilot_lib.py").read_text()
        body = src[src.index("def run_planned_calls"):src.index("def verify_run_accounting")]
        self.assertLess(body.index('"event": "attempt_started"'), body.index("generate_fn(image, prompt_text)"))
        self.assertLess(body.index('"event": "attempt_started"'), body.index("load_image(case)"))
        for needle in ("except KeyboardInterrupt", "except BaseException", "raise reraise", "attempt_skipped",
                       "stopped_after_", "update_run_metadata(out_dir"):
            self.assertIn(needle, body)

    def test_logging_and_generation_settings(self):
        for needle in ("generated_token_ids", "raw_response_with_special_tokens", "finish_reason", "do_sample=False",
                       "num_beams=1", "max_new_tokens=MAX_NEW_TOKENS", "parse_failures.jsonl", "write_split_results",
                       "run_metadata.json", "verify_run_accounting", "make_logger"):
            self.assertIn(needle, self.text)
        self.assertEqual(self.text.count("model.generate("), 1)

    def test_honesty_statements(self):
        for needle in ("SMOKE TEST ONLY", "does NOT validate tumor-measurement accuracy", "UNVERIFIED",
                       "stated spacing does not describe the pixels the model sees"):
            self.assertIn(needle, self.text)

    def test_no_secrets_and_no_writes_to_existing_result_dirs(self):
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", self.text))
        code_text = "\n".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code")
        self.assertNotRegex(code_text, r"(outputs|results)/")
        self.assertIn('userdata.get("HF_TOKEN")', self.text)
        self.assertIn("hf_token = None", self.text)

    def test_reference_values_never_enter_prompts_or_notebook(self):
        self.assertNotIn("ref_major_mm", self.text)
        self.assertNotIn("ref_minor_mm", self.text)
        src = (PILOT / "pilot_lib.py").read_text()
        self.assertIn('prompt_b(case["pixel_spacing_mm_for_prompt"])', src[src.index("def build_prompt"):src.index("def update_run_metadata")])


@unittest.skipUnless(HAVE_REAL_BUNDLE, "needs the locally built bundle (run build_pilot_bundle.py with the dataset)")
class LocalBuildConsistency(unittest.TestCase):
    """Only meaningful on the machine that built the bundle: the committed notebook pins that build's hashes."""

    @classmethod
    def setUpClass(cls):
        with open(NOTEBOOK) as f:
            cls.text = "\n".join("".join(c["source"]) for c in json.load(f)["cells"])

    def test_pinned_hashes_match_built_artifacts(self):
        pins = dict(re.findall(r'^(EXPECTED_\w+) = "([0-9a-f]+)"', self.text, re.M))
        self.assertEqual(pins["EXPECTED_MANIFEST_SHA256"], sha(REAL_BUNDLE / "manifest.json"))
        self.assertEqual(pins["EXPECTED_PILOT_LIB_SHA256"], sha(REAL_BUNDLE / "pilot_lib.py"))
        self.assertEqual(pins["EXPECTED_BUNDLE_ID"], json.loads((REAL_BUNDLE / "manifest.json").read_text())["bundle_id"])
        self.assertEqual(sha(REAL_BUNDLE / "pilot_lib.py"), sha(PILOT / "pilot_lib.py"))

    def test_real_zip_and_bundle_verify_with_the_notebook_pins(self):
        pins = dict(re.findall(r'^(EXPECTED_\w+) = "([0-9a-f]+)"', self.text, re.M))
        with tempfile.TemporaryDirectory() as t:
            with zipfile.ZipFile(PILOT / "pilot5_images.zip") as zf:
                zf.extractall(Path(t) / "pilot5_bundle")
            pl.verify_bundle(Path(t) / "pilot5_bundle", pins["EXPECTED_MANIFEST_SHA256"], pins["EXPECTED_PILOT_LIB_SHA256"])


if __name__ == "__main__":
    unittest.main()
