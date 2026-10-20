import unittest

import torch
from torch import nn

from src.train import build_model, choose_device, make_transforms


class ProjectTests(unittest.TestCase):
    def test_transform_pair(self):
        train_tf, eval_tf = make_transforms(150)
        self.assertNotEqual(train_tf, eval_tf)

    def test_model_has_six_class_head_and_freeze_option(self):
        # Uses the default pretrained weights cache/network if not already available.
        model = build_model(num_classes=6, freeze_backbone=True)
        self.assertEqual(model.fc.out_features, 6)
        self.assertTrue(all(not p.requires_grad for p in model.layer1.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))

    def test_device_is_supported(self):
        self.assertIn(choose_device().type, {"cpu", "cuda", "mps"})


if __name__ == "__main__":
    unittest.main()
