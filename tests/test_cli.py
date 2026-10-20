import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT


class CliTest(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, "-m", "sit", *args], cwd=ROOT,
                              text=True, capture_output=True, timeout=10)

    def test_demo_runs_end_to_end_without_vision_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.call("demo", "--output", tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            paths = list(Path(tmp).rglob("trafico.csv"))
            self.assertEqual(len(paths), 1)
            with paths[0].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            completed = [r for r in rows if r["event_type"] == "ZONA_TRAFICO" and r["complete"] == "true"]
            self.assertEqual(len(completed), 1)
            self.assertEqual(float(completed[0]["duration_s"]), 4)

    def test_check_distinguishes_structure_from_runtime(self):
        result = self.call("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("estructura", result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads((ROOT / "config/cameras.example.json").read_text())
            raw["model"]["weights"] = "missing.pt"
            raw["model"]["tracker"] = str(ROOT / "config/tracking/bytetrack.yaml")
            raw["cameras"][0]["zones"] = [str(ROOT / "config/zones/zona_salida.json")]
            path = Path(tmp) / "cameras.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            result = self.call("check", "--config", str(path), "--runtime")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Falta el modelo local", result.stderr)

    def test_importing_all_modules_has_no_camera_or_model_side_effects(self):
        code = ("import importlib,pkgutil,sys,sit; "
                "[importlib.import_module(m.name) for m in pkgutil.walk_packages(sit.__path__,sit.__name__+'.')]; "
                "assert not {'cv2','torch','ultralytics','numpy'}.intersection(sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
