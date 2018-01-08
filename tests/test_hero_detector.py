import unittest
import os.path as path
import math

import cv2

from features.hero_detector import HeroDetector


class TestHeroDetector(unittest.TestCase):

    def setUp(self):
        self.hero_detector = HeroDetector()
        self.frame_dir = path.join('..', 'service_files', 'test_hero_detector_frames')

    def test_no_hero(self):
        frame = cv2.imread(path.join(self.frame_dir, 'no_hero.png'))
        self.assertEqual((0, 'None'), self.hero_detector.check(frame))

    def test_clear_hero(self):
        frame = cv2.imread(path.join(self.frame_dir, 'LeeSin.png'))
        prob, name = self.hero_detector.check(frame)
        self.assertEqual((1, 'LeeSin'), (math.ceil(prob), 'LeeSin'))

    def test_noisy_hero(self):
        frame = cv2.imread(path.join(self.frame_dir, 'Draven.png'))
        prob, name = self.hero_detector.check(frame)
        self.assertEqual((1, 'Draven'), (math.ceil(prob), 'Draven'))
