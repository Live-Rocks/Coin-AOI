"""Unit tests for v13 dataset handling and the fixed smoke gate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v13_pipeline import (  # noqa: E402
    box_iou,
    load_dataset_descriptor,
    match_detections,
    polygon_to_box,
    smoke_image_passed,
)


class V13PipelineTests(unittest.TestCase):
    def test_dataset_paths_resolve_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            config = repository_root / "datasets" / "v13" / "data.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "path: data/export\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                "names: [dent, scratch, stain_corrosion]\n"
            )
            root, splits, names = load_dataset_descriptor(config, repository_root)

        self.assertEqual(root, (repository_root / "data" / "export").resolve())
        self.assertEqual(splits["train"], root / "train" / "images")
        self.assertEqual(names, ["dent", "scratch", "stain_corrosion"])

    def test_polygon_to_box(self) -> None:
        actual = polygon_to_box([0.1, 0.2, 0.5, 0.2, 0.5, 0.8])
        for value, expected in zip(actual, (0.3, 0.5, 0.4, 0.6), strict=True):
            self.assertAlmostEqual(value, expected)

    def test_same_class_iou_match_passes(self) -> None:
        truth = [{"class_id": 1, "xyxy": [0, 0, 10, 10]}]
        predictions = [{"class_id": 1, "xyxy": [1, 1, 9, 9]}]
        result = match_detections(truth, predictions, 0.5)
        self.assertTrue(smoke_image_passed(truth, result))
        self.assertAlmostEqual(result["matches"][0]["iou"], 0.64)

    def test_wrong_class_and_low_iou_are_misses(self) -> None:
        truth = [{"class_id": 1, "xyxy": [0, 0, 10, 10]}]
        for prediction in (
            {"class_id": 0, "xyxy": [0, 0, 10, 10]},
            {"class_id": 1, "xyxy": [9, 9, 19, 19]},
        ):
            result = match_detections(truth, [prediction], 0.5)
            self.assertFalse(smoke_image_passed(truth, result))
            self.assertEqual(result["unmatched_ground_truth_indices"], [0])

    def test_normal_image_false_positive_fails(self) -> None:
        result = match_detections([], [{"class_id": 0, "xyxy": [0, 0, 1, 1]}], 0.5)
        self.assertFalse(smoke_image_passed([], result))
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
