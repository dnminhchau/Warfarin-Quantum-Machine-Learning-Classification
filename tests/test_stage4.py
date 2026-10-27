import unittest

import numpy as np
import pandas as pd

from src.preprocessing import ALL_STAGE4_FEATURES, build_preprocessor


class Stage4PreprocessingTests(unittest.TestCase):
    def test_preprocessor_runs_and_keeps_missing_distinct(self):
        X_train = pd.DataFrame(
            {
                "Gender": ["male", "female", None, "female"],
                "Age": ["50 - 59", "60 - 69", None, "70 - 79"],
                "Height (cm)": [170.0, np.nan, 165.0, 180.0],
                "Weight (kg)": [70.0, 80.0, np.nan, 90.0],
                "Diabetes": [0.0, 1.0, np.nan, 0.0],
                "Congestive Heart Failure and/or Cardiomyopathy": [0.0, np.nan, 1.0, 0.0],
                "Valve Replacement": [np.nan, 0.0, 1.0, 0.0],
                "Current Smoker": [0.0, 1.0, np.nan, 0.0],
                "CYP2C9 consensus": ["*1/*1", "*1/*2", None, "*1/*1"],
                "VKORC1 -1639 consensus": ["A/G", "G/G", None, "A/A"],
            }
        )

        pre = build_preprocessor(ALL_STAGE4_FEATURES)
        Xt = pre.fit_transform(X_train)
        self.assertEqual(Xt.shape[0], 4)
        self.assertTrue(np.isfinite(Xt).all())

        binary_encoder = pre.named_steps["columns"].named_transformers_["binary"].named_steps["onehot"]
        # At least one binary column saw -1 (missing) as its own category.
        self.assertTrue(any(-1.0 in cats for cats in binary_encoder.categories_))

    def test_unknown_category_at_validation_does_not_crash(self):
        X_train = pd.DataFrame(
            {
                "Gender": ["male", "female"],
                "Age": ["50 - 59", "60 - 69"],
                "Height (cm)": [170.0, 165.0],
                "Weight (kg)": [70.0, 80.0],
                "Diabetes": [0.0, 1.0],
                "Congestive Heart Failure and/or Cardiomyopathy": [0.0, 1.0],
                "Valve Replacement": [0.0, 1.0],
                "Current Smoker": [0.0, 1.0],
                "CYP2C9 consensus": ["*1/*1", "*1/*2"],
                "VKORC1 -1639 consensus": ["A/G", "G/G"],
            }
        )
        X_val = X_train.iloc[[0]].copy()
        X_val.loc[:, "CYP2C9 consensus"] = "RARE_UNSEEN_GENOTYPE"

        pre = build_preprocessor(ALL_STAGE4_FEATURES)
        pre.fit(X_train)
        Xt = pre.transform(X_val)
        self.assertEqual(Xt.shape[0], 1)
        self.assertTrue(np.isfinite(Xt).all())


if __name__ == "__main__":
    unittest.main()
