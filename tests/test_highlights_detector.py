import unittest
import os.path as path

import cv2

from features.highlights_detector import HighlightsDetector


class TestHighlightsDetector(unittest.TestCase):

    def setUp(self):
        self.hl_detector = HighlightsDetector()
        self.frame_dir = path.join('..', 'service_files', 'test_highlights_detector_frames')

    def test_1080p(self):
        frame = cv2.imread(path.join(self.frame_dir, '1080.jpg'))
        highlights = [('First Touches', 'XMIMOSHIKUFU')]
        self.assertEqual(highlights, self.hl_detector.check(frame))

    def test_720p(self):
        frame = cv2.imread(path.join(self.frame_dir, '720.png'))
        highlights = [('Bicycle Hits', 'YOU'), ('Epic Saves', 'YOU'), ('Shorts on Goal', 'TMON')]
        self.assertEqual(highlights, self.hl_detector.check(frame))
