import csv
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import FakePipeline, camera, config, frame, observation
from sit.domain.types import StreamBoundary
from sit.main import run_cameras


class MultiCameraTest(unittest.TestCase):
    def test_coordinator_uses_independent_models_and_camera_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = camera(camera_id="cam_1"), camera(camera_id="cam_2")
            cfg = dataclasses.replace(config(first), cameras=(first, second), data_dir=Path(tmp))
            models = []

            def pipeline_factory(_settings):
                instance = FakePipeline([(observation(),), (observation(point=(120, 50)),),
                                         (observation(point=(125, 50)),)])
                instance.device = "fake-cpu"
                models.append(instance)
                return instance

            class FakeStream:
                def __init__(self, cam, _stop):
                    self.camera = cam

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    pass

                def __iter__(self):
                    return iter([frame(1, 0, camera_id=self.camera.camera_id),
                                 frame(2, 1, camera_id=self.camera.camera_id),
                                 frame(3, 2, camera_id=self.camera.camera_id), StreamBoundary("eof")])

            with patch("sit.perception.ultralytics_pipeline.UltralyticsPipeline", pipeline_factory):
                with patch("sit.capture.rtsp.CaptureStream", FakeStream):
                    self.assertEqual(run_cameras(cfg, (first, second), None), 0)
            self.assertEqual(len(models), 2)
            self.assertIsNot(models[0], models[1])
            paths = list(Path(tmp).rglob("trafico.csv"))
            self.assertEqual(len(paths), 2)
            sessions = set()
            for path in paths:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([r["event_type"] for r in rows], ["ENTRADA_ZONA", "ZONA_TRAFICO"])
                self.assertEqual(rows[0]["camera_id"], path.parent.name)
                self.assertEqual(rows[1]["complete"], "true")
                sessions.add(rows[0]["session_id"])
            self.assertEqual(len(sessions), 2)
