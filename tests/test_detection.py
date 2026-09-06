"""Regression checks for errors found in the original detection pipeline."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from detection import (CustomObjectDetectionModel, ObjectDetectionDataset, box_iou,
                       composite_loss, detection_metrics, generate_dataset, train)
from smoke_assets import create_assets


class MetricsTests(unittest.TestCase):
    def test_class_zero_and_iou_boundary(self):
        box = [.5, .5, .4, .4]
        target = [(c, box) for c in range(3)]
        metrics = detection_metrics([(c, .9, box) for c in range(3)], target)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["map50_all_points"], 1)
        self.assertAlmostEqual(box_iou(box, box), 1)
        self.assertEqual(box_iou(box, [.5, .5, -.4, .4]), 0)
        boundary = detection_metrics([(0, 1., [.5, .5, .25, .5])], [(0, [.5, .5, .5, .5])])
        self.assertEqual(boundary["true_positives"], 1)

    def test_wrong_class_missed_box_and_abstention(self):
        box = [.5, .5, .2, .2]
        predictions = [(1, .9, box), (1, .8, [.1, .1, .1, .1]), None]
        result = detection_metrics(predictions, [(0, box), (1, box), (2, box)])
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["map50_all_points"], 0)
        self.assertAlmostEqual(result["mean_iou"], 1/3)
        self.assertEqual(result["samples"], 3)

    def test_ap_ranking_penalizes_confident_false_positive(self):
        box = [.5, .5, .2, .2]
        result = detection_metrics([(0, .9, box), (0, .8, box)], [(1, box), (0, box)])
        self.assertEqual(result["ap50_by_class"]["Waldo"], .5)
        self.assertEqual(result["map50_all_points"], .25)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        objects, backgrounds = create_assets(self.root / "assets")
        self.dataset = self.root / "dataset"
        self.manifest = generate_dataset(objects, backgrounds, self.dataset, (6, 3, 3), size=64)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_disjoint_valid_labels_and_deterministic_validation(self):
        groups = [set(r["sha256"] for r in rows) for rows in self.manifest["backgrounds"].values()]
        self.assertFalse(groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])
        for split in ["train", "val", "test"]:
            ds = ObjectDetectionDataset(self.dataset, split, size=64)
            for image, target in ds:
                self.assertEqual(tuple(image.shape), (3, 64, 64))
                self.assertEqual(tuple(target.shape), (5,))
            self.assertTrue(torch.equal(ds[0][0], ds[0][0]))
        with self.assertRaises(ValueError):
            ObjectDetectionDataset(self.dataset, "val", augment=True)

    def test_flip_updates_box_and_missing_label_fails(self):
        ds = ObjectDetectionDataset(self.dataset, "train", size=64, augment=True)
        with patch.object(random, "random", return_value=1):
            _, original = ds[0]
        with patch.object(random, "random", return_value=0):
            _, mirrored = ds[0]
        self.assertAlmostEqual(float(mirrored[1]), 1-float(original[1]), places=6)
        self.assertTrue(torch.equal(mirrored[2:], original[2:]))
        (self.dataset / "train" / "labels" / "00000.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            ds[0]

    def test_model_has_valid_boxes_and_both_heads_receive_gradients(self):
        model = CustomObjectDetectionModel(pretrained=False)
        logits, boxes = model(torch.rand(2, 3, 64, 64))
        self.assertTrue(torch.all(boxes[:, :2] - boxes[:, 2:]/2 >= 0))
        self.assertTrue(torch.all(boxes[:, :2] + boxes[:, 2:]/2 <= 1))
        target = torch.tensor([[0, .5, .5, .2, .3], [2, .4, .4, .2, .2]])
        loss, _, _ = composite_loss(logits, boxes, target)
        loss.backward()
        for head in [model.classifier, model.regressor]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in head.parameters()), 0)

    def test_box_head_can_fit_a_fixed_batch_without_collapsing(self):
        torch.manual_seed(7)
        model = CustomObjectDetectionModel(pretrained=False).eval()
        # Exercise the real regression head on stable, large-magnitude backbone
        # features: this catches the saturation observed in the first pilot.
        features = torch.rand(4, 512, 7, 7) * 20
        wanted = torch.tensor([[.3,.4,.1,.25], [.6,.5,.15,.2],
                               [.4,.7,.12,.25], [.7,.3,.1,.2]])
        optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=1e-4)
        def prediction():
            raw = model.regressor(features).sigmoid()
            wh = raw[:, 2:]
            return torch.cat((wh/2 + raw[:, :2]*(1-wh), wh), 1)
        initial = torch.nn.functional.smooth_l1_loss(prediction(), wanted).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(prediction(), wanted)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * .2)
        self.assertGreater(prediction()[:, 2:].min().item(), .01)

    def test_training_resume_and_dataset_mismatch(self):
        output = self.root / "run"
        train(self.dataset, output, epochs=1, batch_size=3, size=64, pretrained=False, device="cpu")
        _, history, metrics = train(self.dataset, output, epochs=2, batch_size=3,
                                    size=64, pretrained=False, resume=True, device="cpu")
        self.assertEqual(len(history), 2)
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["precision"], metrics["recall"])
        manifest_path = self.dataset / "manifest.json"
        manifest_path.write_text(manifest_path.read_text() + " ")
        with self.assertRaises(ValueError):
            train(self.dataset, output, epochs=3, batch_size=3, size=64,
                  pretrained=False, resume=True, device="cpu")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
