import unittest

from features import FeatureDetector

class TestFeatureDetector(unittest.TestCase):
    def setUp(self):
        self.feature = FeatureDetector()

    def test_check_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.feature.check(None)
