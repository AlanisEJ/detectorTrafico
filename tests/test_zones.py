import dataclasses
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, zone
from sit.domain.zones import Zone, centroid, load_zone, save_zone


class ZonesTest(unittest.TestCase):
    def test_inside_outside_and_edges(self):
        z = zone()
        for point in ((50, 50), (0, 0), (100, 100), (0, 50)):
            self.assertTrue(z.contains(point))
        self.assertFalse(z.contains((101, 50)))

    def test_rejects_degenerate_crossed_and_out_of_range_polygons(self):
        bad = [((1, 1), (2, 2)), ((1, 1), (2, 2), (3, 3)),
               ((0, 0), (100, 100), (0, 100), (100, 0)),
               ((0, 0), (160, 0), (0, 100)),
               ((0, 0), (100, 0), (0, 0)),
               ((0, 0), (float("nan"), 0), (0, 100))]
        for points in bad:
            with self.subTest(points=points), self.assertRaises(ValueError):
                dataclasses.replace(zone(), points=points)

    def test_centroid_both_winding_orders(self):
        self.assertEqual(centroid(zone().points), (50, 50))
        self.assertEqual(centroid(tuple(reversed(zone().points))), (50, 50))

    def test_json_roundtrip_and_original_zone_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zone.json"
            save_zone(path, zone())
            self.assertEqual(load_zone(path), zone())
            self.assertFalse(path.with_suffix(".json.tmp").exists())
        for name, count in (("zona_cam1", 9), ("zona_cam2", 7), ("zona_salida", 19)):
            self.assertEqual(len(load_zone(ROOT / "config/zones" / f"{name}.json").points), count)
