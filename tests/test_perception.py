import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sit.config import ModelConfig
from sit.perception.ultralytics_pipeline import UltralyticsPipeline


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    task = "segment"

    def __init__(self, _path):
        self.calls = []
        self.result = SimpleNamespace(boxes=SimpleNamespace(id=Tensor([7]), cls=Tensor([2])),
                                      masks=SimpleNamespace(xy=[[(0, 0), (10, 0), (10, 10), (0, 10)]]))

    def track(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class PerceptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weights = Path(self.tmp.name) / "fixture.pt"
        self.weights.write_bytes(b"fixture, not a serialized model")
        self.settings = ModelConfig(self.weights, Path("bytetrack.yaml"))
        self.cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    def make(self, **kwargs):
        return UltralyticsPipeline(self.settings, model_factory=FakeModel, torch_module=self.cpu, **kwargs)

    def test_segment_masks_map_to_domain_observations_and_cpu_disables_fp16(self):
        pipeline = self.make()
        observation, = pipeline.infer(object())
        self.assertEqual((observation.track_id, observation.class_id, observation.center), (7, 2, (5, 5)))
        self.assertEqual(observation.label, "Auto")
        self.assertEqual(pipeline.model.calls[0]["device"], "cpu")
        self.assertFalse(pipeline.model.calls[0]["half"])
        self.assertTrue(pipeline.model.calls[0]["persist"])
        self.assertEqual(pipeline.model.calls[0]["classes"], [2, 3, 5, 7])

    def test_reset_replaces_tracker_owner(self):
        pipeline = self.make()
        previous = pipeline.model
        pipeline.reset()
        self.assertIsNot(pipeline.model, previous)

    def test_detect_only_model_is_rejected(self):
        class DetectModel(FakeModel):
            task = "detect"
        with self.assertRaisesRegex(ValueError, "segmentación"):
            UltralyticsPipeline(self.settings, model_factory=DetectModel, torch_module=self.cpu)

    def test_missing_masks_are_not_silently_treated_as_empty_traffic(self):
        pipeline = self.make()
        pipeline.model.result.masks = None
        with self.assertRaisesRegex(ValueError, "sin máscaras"):
            pipeline.infer(object())

    def test_no_tracker_ids_is_a_valid_empty_result(self):
        pipeline = self.make()
        pipeline.model.result.boxes.id = None
        self.assertEqual(pipeline.infer(object()), ())

    def test_missing_weights_fails_before_model_factory_is_called(self):
        def never_load(_path):
            raise AssertionError("No debe intentar descargar/cargar un modelo faltante")
        with self.assertRaisesRegex(ValueError, "Falta el modelo"):
            UltralyticsPipeline(dataclasses.replace(self.settings, weights=Path("missing.pt")),
                                model_factory=never_load, torch_module=self.cpu)
