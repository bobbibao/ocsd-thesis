"""OCSD v3 pipeline wiring: synthetic model, no image-quality claim."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ocsd import ocsd_v3


def sketch_rgb():
    img = np.full((512, 512, 3), 255, np.uint8)
    img[100:200, 60:62] = 0; img[100:102, 60:200] = 0; img[198:200, 60:200] = 0; img[100:200, 198:200] = 0
    img[300:420, 300:302] = 0; img[300:302, 300:460] = 0
    return img


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def generate(condition, seed, negative, **kw):
            self.calls.append((condition, seed, kw)); return np.full((512, 512, 3), 40, np.uint8), dict(unet_sample_calls=1)
        self.model = SimpleNamespace(generate=generate)

    def test_sketch_to_image_uses_parsed_silhouettes_and_frozen_settings(self):
        pipe = ocsd_v3.OCSDv3(self.model, config=ocsd_v3.with_overrides(layout_only=True, refine=False), device='cpu')
        image, log = pipe(sketch_rgb(), 'a red box, a blue hook', 'on a table', seed=3)
        self.assertEqual(image.shape, (512, 512, 3))
        condition, seed, kw = self.calls[0]
        self.assertEqual(seed, 3); self.assertEqual(len(condition['objects']), 2)
        self.assertTrue(all('silhouette_rle' in o for o in condition['objects']))
        self.assertTrue(kw['silhouette']); self.assertTrue(kw['silhouette_layout_only']); self.assertFalse(kw['scribble'])
        self.assertEqual(log['method'], 'ocsd_v3'); self.assertTrue(log['ocsd_v3_config']['layout_only'])
        # Object order is left to right, so phrases follow the sketch.
        self.assertEqual([o['phrase'] for o in condition['objects']], ['a red box', 'a blue hook'])

    def test_refiner_runs_on_the_base_image_when_configured(self):
        seen = []
        def fake_refine(pipe, pixels, caption, seed, strength, steps, device='cuda'):
            seen.append((int(pixels[0, 0, 0]), caption, seed, strength, steps)); return pixels+1, dict(refiner_steps=steps)
        pipe = ocsd_v3.OCSDv3(self.model, refiner=object(), refiner_revision='a'*40,
                              config=ocsd_v3.with_overrides(refine=True), device='cpu')
        with patch('ocsd.direct_instances.refine', side_effect=fake_refine):
            image, log = pipe(sketch_rgb(), 'a red box, a blue hook', 'on a table', seed=1)
        self.assertEqual(int(image[0, 0, 0]), 41); self.assertEqual(seen[0][0], 40)
        self.assertEqual(seen[0][2:], (1, .35, 20)); self.assertEqual(log['refiner_revision'], 'a'*40)
        missing = ocsd_v3.OCSDv3(self.model, config=ocsd_v3.with_overrides(refine=True), device='cpu')
        with self.assertRaisesRegex(RuntimeError, 'refiner'):
            missing(sketch_rgb(), 'a red box, a blue hook', 'on a table')

    def test_phrase_count_must_match_objects(self):
        pipe = ocsd_v3.OCSDv3(self.model, device='cpu')
        with self.assertRaisesRegex(ValueError, 'phrases'):
            pipe(sketch_rgb(), 'a red box', 'on a table')


    def test_demo_lists_v3_first_and_shows_parsed_silhouettes(self):
        from ocsd import app
        self.assertEqual(list(app.METHODS.values())[0], 'ocsd_v3')
        scene = app.build_scene_from_input(sketch_rgb(), 'a red box, a blue hook', 'on a table')
        overlay = app.silhouette_overlay(scene)
        self.assertEqual(overlay.shape, (512, 512, 3))
        # The closed square is filled (interior coloured), the open hook stays thin.
        self.assertFalse((overlay[150, 130] == 255).all())
        self.assertTrue((overlay[380, 400] == 255).all())

if __name__ == '__main__': unittest.main()
