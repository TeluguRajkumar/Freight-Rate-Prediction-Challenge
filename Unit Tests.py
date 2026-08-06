##UNIT TESTS

"""test_freight_rate_pipeline.py -- run with: pytest -v"""

import numpy as np
import pandas as pd
import pytest

from freight_rate_pipeline import (
    FeatureEngineer,
    FEATURE_ORDER,
    assert_valid_december_output,
    assert_valid_predictions_output,
    inverse_target,
    make_target,
    DECEMBER_COLUMNS,
    DECEMBER_FIXED,
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_LOAD_IDS,
)


@pytest.fixture
def tiny_train_df() -> pd.DataFrame:
    return pd.DataFrame({
        "pickup": ["Richmond", "Savannah", "Richmond", "Boston"],
        "delivery": ["Baltimore", "LA", "Baltimore", "NYC"],
        "distance": [274.3, 2406.2, 300.0, 215.0],
        "equipment": ["Dry Van", "Reefer", "Flatbed", "Dry Van"],
        "weight": [30658.0, np.nan, 20000.0, 15000.0],
        "date": ["2025-01-01", "2025-06-15", "2025-03-10", "2025-09-01"],
        "posted_rate": [645.41, 5000.0, 700.0, 480.0],
    })


class TestFeatureEngineer:
    def test_fit_before_transform_raises(self, tiny_train_df):
        with pytest.raises(RuntimeError):
            FeatureEngineer().transform(tiny_train_df)

    def test_output_columns_match_feature_order(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        features = engineer.transform(tiny_train_df)
        assert list(features.columns) == FEATURE_ORDER

    def test_missing_weight_is_imputed_with_train_median_and_flagged(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        features = engineer.transform(tiny_train_df)
        expected_median = tiny_train_df["weight"].median()
        assert features.loc[1, "weight"] == pytest.approx(expected_median)
        assert features.loc[1, "weight_missing"] == 1
        assert features.loc[0, "weight_missing"] == 0

    def test_unseen_category_maps_to_unknown_not_a_crash(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        unseen = pd.DataFrame({
            "pickup": ["Nowhereville"], "delivery": ["Baltimore"],
            "distance": [100.0], "equipment": ["Dry Van"],
            "weight": [10000.0], "date": ["2025-12-05"],
        })
        features = engineer.transform(unseen)
        assert features.loc[0, "pickup"] == "UNKNOWN"

    def test_non_positive_distance_raises(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        bad = tiny_train_df.copy()
        bad.loc[0, "distance"] = 0
        with pytest.raises(ValueError, match="distance"):
            engineer.transform(bad)

    def test_unparseable_date_raises(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        bad = tiny_train_df.copy()
        bad.loc[0, "date"] = "not-a-date"
        with pytest.raises(ValueError, match="date"):
            engineer.transform(bad)

    def test_cyclic_features_are_bounded(self, tiny_train_df):
        engineer = FeatureEngineer().fit(tiny_train_df)
        features = engineer.transform(tiny_train_df)
        for col in ("doy_sin_1", "doy_cos_1", "doy_sin_2", "doy_cos_2", "dow_sin", "dow_cos"):
            assert features[col].between(-1.0001, 1.0001).all()


class TestTargetTransform:
    def test_round_trip_is_close(self):
        posted_rate = pd.Series([645.41, 5000.0, 120.0])
        distance = pd.Series([274.3, 2406.2, 90.0])
        log_rpm = make_target(posted_rate, distance)
        recovered = inverse_target(log_rpm.to_numpy(), distance.to_numpy())
        np.testing.assert_allclose(recovered, posted_rate.to_numpy(), rtol=1e-6)

    def test_inverse_target_floors_at_positive_value(self):
        # A wildly negative log-prediction must never invert to <= 0.
        result = inverse_target(np.array([-50.0]), np.array([100.0]))
        assert (result > 0).all()


class TestOutputValidation:
    def _valid_predictions(self) -> pd.DataFrame:
        ids = [f"TE-{i:06d}" for i in range(1, EXPECTED_VALIDATION_ROWS + 1)]
        return pd.DataFrame({"load_id": ids, "predicted_rate": np.full(EXPECTED_VALIDATION_ROWS, 1000.0)})

    def test_valid_predictions_pass(self):
        assert_valid_predictions_output(self._valid_predictions())  # should not raise

    def test_wrong_row_count_raises(self):
        bad = self._valid_predictions().iloc[:-1]
        with pytest.raises(ValueError, match="rows"):
            assert_valid_predictions_output(bad)

    def test_duplicate_load_id_raises(self):
        bad = self._valid_predictions()
        bad.loc[1, "load_id"] = bad.loc[0, "load_id"]
        with pytest.raises(ValueError, match="duplicate"):
            assert_valid_predictions_output(bad)

    def test_non_positive_rate_raises(self):
        bad = self._valid_predictions()
        bad.loc[0, "predicted_rate"] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            assert_valid_predictions_output(bad)

    def test_wrong_column_order_raises(self):
        bad = self._valid_predictions()[["predicted_rate", "load_id"]]
        with pytest.raises(ValueError, match="columns"):
            assert_valid_predictions_output(bad)

    def _valid_december(self) -> pd.DataFrame:
        dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
        return pd.DataFrame({
            "pickup": [DECEMBER_FIXED["pickup"]] * 31,
            "delivery": [DECEMBER_FIXED["delivery"]] * 31,
            "distance": [DECEMBER_FIXED["distance"]] * 31,
            "equipment": [DECEMBER_FIXED["equipment"]] * 31,
            "weight": [DECEMBER_FIXED["weight"]] * 31,
            "date": dates.strftime("%Y-%m-%d"),
            "predicted_rate": np.linspace(700, 800, 31),
        })[DECEMBER_COLUMNS]

    def test_valid_december_passes(self):
        assert_valid_december_output(self._valid_december())  # should not raise

    def test_december_wrong_fixed_value_raises(self):
        bad = self._valid_december()
        bad.loc[0, "pickup"] = "Somewhere Else"
        with pytest.raises(ValueError, match="pickup"):
            assert_valid_december_output(bad)

    def test_december_missing_a_date_raises(self):
        bad = self._valid_december().iloc[:-1]
        with pytest.raises(ValueError, match="one row per day"):
            assert_valid_december_output(bad)

    def test_december_duplicate_date_raises(self):
        bad = self._valid_december().copy()
        bad.loc[1, "date"] = bad.loc[0, "date"]
        with pytest.raises(ValueError, match="duplicate"):
            assert_valid_december_output(bad)
