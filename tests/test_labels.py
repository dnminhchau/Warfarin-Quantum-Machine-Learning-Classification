
import unittest

from scripts.make_labels import label_dose


class TestWarfarinLabels(unittest.TestCase):

    def test_low_dose(self):
        self.assertEqual(label_dose(20.9), "Low")
        self.assertEqual(label_dose(21), "Low")

    def test_intermediate_dose(self):
        self.assertEqual(label_dose(21.1), "Intermediate")
        self.assertEqual(label_dose(48.9), "Intermediate")

    def test_high_dose(self):
        self.assertEqual(label_dose(49), "High")
        self.assertEqual(label_dose(49.1), "High")

    def test_invalid_dose(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    label_dose(value)


if __name__ == "__main__":
    unittest.main()
