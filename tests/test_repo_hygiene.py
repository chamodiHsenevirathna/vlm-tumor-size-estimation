"""Guards against reintroducing absolute local paths. Run: python -m unittest discover -s tests -v"""

import csv
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class TestRelativePaths(unittest.TestCase):
    def test_qc_csv_has_only_relative_viz_paths(self):
        with open(REPO / "results" / "qc_analysis.csv", newline="") as f:
            paths = [row["viz_path"] for row in csv.DictReader(f)]
        self.assertTrue(paths)
        for p in paths:
            self.assertFalse(Path(p).is_absolute() or "/Users/" in p or "\\" in p, p)

    def test_analyze_qc_writes_repo_relative_viz_path(self):
        src = (REPO / "src" / "analyze_qc.py").read_text()
        self.assertIn('out_path.relative_to(REPO_ROOT).as_posix()', src)
        self.assertNotIn('"viz_path": str(out_path)', src)


if __name__ == "__main__":
    unittest.main()
