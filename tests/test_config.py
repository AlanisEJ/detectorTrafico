import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import ROOT, camera
from sit.config import ModelConfig, load_config


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cameras.json"
        self.raw = json.loads((ROOT / "config/cameras.example.json").read_text())
        self.raw["model"]["tracker"] = str(ROOT / "config/tracking/bytetrack.yaml")
        self.raw["cameras"][0]["zones"] = [str(ROOT / "config/zones/zona_salida.json")]

    def load(self):
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        return load_config(self.path)

    def test_sample_is_valid_without_model_or_environment(self):
        cfg = load_config(ROOT / "config/cameras.example.json")
        self.assertEqual([c.camera_id for c in cfg.cameras], ["cam_1"])
        self.assertEqual(cfg.cameras[0].resolution, (1280, 720))

    def test_paths_resolve_from_config_file_not_cwd(self):
        self.raw["model"]["weights"] = "../weights/model.pt"
        cfg = self.load()
        self.assertEqual(cfg.model.weights, (self.path.parent / "../weights/model.pt").resolve())

    def test_credential_is_only_resolved_at_runtime(self):
        cfg = self.load()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SIT_CAM_1_RTSP"):
                cfg.cameras[0].resolve_source()
        secret = "rtsp://test_user:sample_password@127.0.0.1/video"
        with patch.dict(os.environ, {"SIT_CAM_1_RTSP": secret}):
            self.assertEqual(cfg.cameras[0].resolve_source(), secret)
            self.assertNotIn("sample_password", repr(cfg))

    def test_rejects_plaintext_credentials(self):
        self.raw["cameras"][0]["source"]["url"] = "rtsp://example.invalid"
        with self.assertRaisesRegex(ValueError, "fuera del JSON"):
            self.load()

    def test_rejects_wrong_resolution_and_duplicate_camera(self):
        self.raw["cameras"][0]["resolution"] = [640, 360]
        with self.assertRaises(ValueError):
            self.load()
        self.raw["cameras"][0]["resolution"] = [1280, 720]
        self.raw["cameras"].append(self.raw["cameras"][0].copy())
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_invalid_thresholds_and_capture_settings(self):
        for key, value in (("frame_skip", 0), ("buffer_size", True), ("read_timeout_ms", -1)):
            with self.subTest(key=key):
                self.raw["cameras"][0][key] = value
                with self.assertRaises(ValueError):
                    self.load()
                del self.raw["cameras"][0][key]
        self.raw["exit_confirm_s"] = self.raw["lost_timeout_s"]
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_unsafe_camera_path(self):
        self.raw["cameras"][0]["id"] = "../escape"
        with self.assertRaises(ValueError):
            self.load()

    def test_hash_verification_and_missing_weights(self):
        weights = self.path.parent / "fixture.pt"
        weights.write_bytes(b"fixture only; never deserialized")
        cfg = ModelConfig(weights, Path("tracker.yaml"), hashlib.sha256(weights.read_bytes()).hexdigest())
        cfg.verify_weights()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            dataclasses.replace(cfg, sha256="0" * 64).verify_weights()
        with self.assertRaisesRegex(ValueError, "Falta el modelo"):
            dataclasses.replace(cfg, weights=Path("missing.pt")).verify_weights()

    def test_local_video_requires_an_existing_file(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(camera(), video_path=Path("missing.mp4")).resolve_source()
