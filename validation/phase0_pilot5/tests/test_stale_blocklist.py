"""Tests for the committed stale-image blocklist and the builder's fail-closed loader.
Run from the repo root: python -m unittest discover -s validation/phase0_pilot5/tests -v"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent
REPO = PILOT.parent.parent
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(REPO / "src"))

import build_pilot_bundle as bb  # noqa: E402

# Independently recorded from the original blank exports (also pinned in the author's first local manifest).
EXPECTED = {
    "186e9061cf09bc4fc6e913e5756b0fb4469e83823ed87d4e050081682cfc3849",
    "1c1833c9389efc85f624652fc9c1710f1ec292ebd36f43f8d26da383be4cc78a",
    "1f42141de41680e4be86354ebb6fec132cfdbafbea470417fa8004ffc3ee1387",
    "6d33bbd4301c4cb93f1a71057a8e5e0142d4b50152eb66982e82af6ce4ba79b8",
    "9d9f405f8b3231f8779a0d77a3829e48c3a0a3263fee4dd8ea5ae9958b98cc2f",
    "a9b8a077e9c10b19584ea2b2a56484634c42a186185f8a563888369f014bc068",
}


class TestStaleBlocklist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, obj, name="b.json"):
        p = self.dir / name
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
        return p

    def test_committed_blocklist_matches_expected_without_local_images(self):
        missing = [self.dir / "nope.png"]  # a fresh clone: the removed PNGs do not exist
        self.assertEqual(set(bb.load_stale_blocklist(bb.STALE_JSON, missing)), EXPECTED)

    def test_default_stale_files_may_be_absent_or_consistent(self):
        # works both in a fresh clone (files absent) and on the author's machine (files present and matching)
        self.assertEqual(set(bb.load_stale_blocklist()), EXPECTED)

    def test_local_original_must_be_in_blocklist(self):
        rogue = self.dir / "rogue.png"
        rogue.write_bytes(b"not in the list")
        with self.assertRaisesRegex(RuntimeError, "not in the committed blocklist"):
            bb.load_stale_blocklist(bb.STALE_JSON, [rogue])

    def test_matching_local_original_accepted(self):
        h = "a" * 64
        f = self.dir / "x.png"
        f.write_bytes(b"x")
        real = hashlib.sha256(b"x").hexdigest()
        p = self.write({"sha256": [real] + [c * 64 for c in "bcdef"]})
        self.assertIn(real, bb.load_stale_blocklist(p, [f]))
        self.assertNotIn(h, bb.load_stale_blocklist(p, [f]))

    def test_fail_closed_on_bad_blocklists(self):
        good = [c * 64 for c in "abcdef"]
        cases = {
            "missing file": self.dir / "absent.json",
            "invalid json": self.write("{not json", "bad.json"),
            "no key": self.write({"other": good}, "nokey.json"),
            "not a list": self.write({"sha256": "abc"}, "nl.json"),
            "short hash": self.write({"sha256": good[:5] + ["abc"]}, "short.json"),
            "uppercase": self.write({"sha256": good[:5] + ["A" * 64]}, "up.json"),
            "too few": self.write({"sha256": good[:4]}, "few.json"),
            "duplicates": self.write({"sha256": good[:5] + [good[0]]}, "dup.json"),
            "empty": self.write({"sha256": []}, "empty.json"),
        }
        for name, path in cases.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                bb.load_stale_blocklist(path, [])

    def test_original_exports_still_hash_to_blocklist_when_present(self):
        present = [p for p in bb.STALE_FILES if p.exists()]
        if not present:
            self.skipTest("original blank exports are not published")
        self.assertEqual({bb.pl.sha256_file(p) for p in present}, EXPECTED)


if __name__ == "__main__":
    unittest.main()
