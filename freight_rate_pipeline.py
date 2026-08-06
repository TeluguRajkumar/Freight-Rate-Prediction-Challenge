"""
freight_rate_pipeline.py

End-to-end, leakage-safe training + inference pipeline for the
Freight Rate Prediction Challenge.

Design summary
---------------
* Target   : log1p(posted_rate / distance)  ->  "rate per mile" in log space.
             Inverted at prediction time: rate = expm1(pred) * distance.
* Model    : sklearn HistGradientBoostingRegressor with native categorical
             support (pickup / delivery / equipment), no one-hot blow-up.
* Features : distance (+log), weight (+missing flag), month, is_weekend,
             day-of-year Fourier harmonics (captures seasonality and lets
             the model generalize "December looks like January" without
             needing literal chronological extrapolation), day-of-week
             Fourier terms.
* Excluded : market_index, quote_signal (near-zero correlation with the
             target AND absent from december_chart_inputs.csv -- keeping
             one consistent feature set for both required outputs avoids
             having to forecast-the-forecast-input).
* Split    : TimeSeriesSplit over chronologically sorted data (never
             random k-fold) because validation/december dates are in the
             future relative to training -- this is a forecasting task.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

LOGGER = logging.getLogger("freight_rate_pipeline")
RANDOM_STATE = 42

EXPECTED_VALIDATION_ROWS = 12_000
EXPECTED_LOAD_IDS = {f"TE-{i:06d}" for i in range(1, EXPECTED_VALIDATION_ROWS + 1)}

DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
DECEMBER_FIXED = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32_000.0,
}
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]

NUMERIC_FEATURES = [
    "distance", "log_distance", "weight", "weight_missing",
    "month", "is_weekend",
    "doy_sin_1", "doy_cos_1", "doy_sin_2", "doy_cos_2",
    "dow_sin", "dow_cos",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_TRAIN_COLUMNS = {
    "pickup", "delivery", "distance", "equipment", "weight", "date", "posted_rate",
}
REQUIRED_INFERENCE_COLUMNS = {"pickup", "delivery", "distance", "equipment", "weight", "date"}

RATE_FLOOR = 0.01


def load_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: file not found at {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label}: file at {path} contains zero rows")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing required columns {sorted(missing)}")
    return frame


@dataclass
class _FitState:
    weight_median: float
    known_pickup: frozenset
    known_delivery: frozenset
    known_equipment: frozenset


class FeatureEngineer:
    HARMONICS = 2

    def __init__(self) -> None:
        self._state: Optional[_FitState] = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        if "weight" not in df.columns:
            raise ValueError("fit(): input frame must contain a 'weight' column")
        self._state = _FitState(
            weight_median=float(pd.to_numeric(df["weight"], errors="coerce").median()),
            known_pickup=frozenset(df["pickup"].astype(str).unique()),
            known_delivery=frozenset(df["delivery"].astype(str).unique()),
            known_equipment=frozenset(df["equipment"].astype(str).unique()),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform().")
        state = self._state
        out = pd.DataFrame(index=df.index)

        distance = pd.to_numeric(df["distance"], errors="coerce")
        if distance.isna().any() or (distance <= 0).any():
            raise ValueError("distance must be a positive, non-null number for every row.")
        out["distance"] = distance.astype(float)
        out["log_distance"] = np.log1p(out["distance"])

        weight = pd.to_numeric(df["weight"], errors="coerce")
        out["weight_missing"] = weight.isna().astype(np.int8)
        out["weight"] = weight.fillna(state.weight_median).astype(float)

        date = pd.to_datetime(df["date"], errors="coerce")
        if date.isna().any():
            raise ValueError("date column contains value(s) that could not be parsed.")
        day_of_year = date.dt.dayofyear.astype(float)
        day_of_week = date.dt.dayofweek.astype(float)
        out["month"] = date.dt.month.astype(float)
        out["is_weekend"] = (day_of_week >= 5).astype(np.int8)
        for h in range(1, self.HARMONICS + 1):
            angle = 2.0 * np.pi * h * day_of_year / 365.25
            out[f"doy_sin_{h}"] = np.sin(angle)
            out[f"doy_cos_{h}"] = np.cos(angle)
        dow_angle = 2.0 * np.pi * day_of_week / 7.0
        out["dow_sin"] = np.sin(dow_angle)
        out["dow_cos"] = np.cos(dow_angle)

        out["equipment"] = self._safe_category(df["equipment"], state.known_equipment)
        out["pickup"] = self._safe_category(df["pickup"], state.known_pickup)
        out["delivery"] = self._safe_category(df["delivery"], state.known_delivery)

        return out[FEATURE_ORDER]

    @staticmethod
    def _safe_category(series: pd.Series, known_values: frozenset) -> pd.Categorical:
        text = series.astype(str)
        safe = text.where(text.isin(known_values), other="UNKNOWN")
        categories = sorted(known_values) + ["UNKNOWN"]
        return pd.Categorical(safe, categories=categories)


def make_target(posted_rate: pd.Series, distance: pd.Series) -> pd.Series:
    rate_per_mile = posted_rate.astype(float) / distance.astype(float)
    return np.log1p(rate_per_mile)


def inverse_target(log_rate_per_mile: np.ndarray, distance: np.ndarray) -> np.ndarray:
    rate_per_mile = np.clip(np.expm1(log_rate_per_mile), RATE_FLOOR, None)
    return rate_per_mile * distance


def build_model(**overrides) -> HistGradientBoostingRegressor:
    params = dict(
        loss="squared_error",
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_iter=400,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def _dollar_rmse_scorer(estimator, X: pd.DataFrame, y_log_rpm: np.ndarray) -> float:
    pred = inverse_target(estimator.predict(X), X["distance"].to_numpy())
    true = inverse_target(y_log_rpm, X["distance"].to_numpy())
    return -float(np.sqrt(mean_squared_error(true, pred)))


def tune_hyperparameters(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, n_iter: int = 25
) -> HistGradientBoostingRegressor:
    param_distributions = {
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 40, 80],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "max_iter": [200, 400, 600],
    }
    search = RandomizedSearchCV(
        estimator=build_model(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring=_dollar_rmse_scorer,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y)
    LOGGER.info("Best CV params: %s (dollar RMSE=%.2f)", search.best_params_, -search.best_score_)
    return build_model(**search.best_params_)


def backtest_metrics(train_df: pd.DataFrame, model_params: dict, n_splits: int = 5) -> dict[str, float]:
    train_df = train_df.sort_values("date").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in splitter.split(train_df):
        fold_train, fold_test = train_df.iloc[train_idx], train_df.iloc[test_idx]

        engineer = FeatureEngineer().fit(fold_train)
        X_train = engineer.transform(fold_train)
        y_train = make_target(fold_train["posted_rate"], fold_train["distance"])
        X_test = engineer.transform(fold_test)

        model = build_model(**model_params)
        model.fit(X_train, y_train)

        pred_rate = inverse_target(model.predict(X_test), X_test["distance"].to_numpy())
        true_rate = fold_test["posted_rate"].to_numpy()

        fold_metrics.append({
            "rmse": float(np.sqrt(mean_squared_error(true_rate, pred_rate))),
            "mae": float(mean_absolute_error(true_rate, pred_rate)),
            "mape": float(mean_absolute_percentage_error(true_rate, pred_rate)),
            "r2": float(r2_score(true_rate, pred_rate)),
        })

    return {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ("rmse", "mae", "mape", "r2")
    }


def assert_valid_predictions_output(predictions: pd.DataFrame) -> None:
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("predictions must have exactly columns ['load_id', 'predicted_rate'] in that order")
    if len(predictions) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"predictions must contain exactly {EXPECTED_VALIDATION_ROWS:,} rows")
    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        raise ValueError("predictions contains missing or duplicate load_id values")
    submitted = set(predictions["load_id"].astype(str))
    if submitted != EXPECTED_LOAD_IDS:
        missing = EXPECTED_LOAD_IDS - submitted
        extra = submitted - EXPECTED_LOAD_IDS
        raise ValueError(f"load_id mismatch (missing={len(missing)}, extra={len(extra)})")
    rates = pd.to_numeric(predictions["predicted_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates).all():
        raise ValueError("predicted_rate contains non-numeric or non-finite values")
    if (rates <= 0).any():
        raise ValueError("predicted_rate contains non-positive values")


def assert_valid_december_output(december: pd.DataFrame) -> None:
    if list(december.columns) != DECEMBER_COLUMNS:
        raise ValueError(f"December output must keep columns {DECEMBER_COLUMNS} in that exact order")
    dates = pd.to_datetime(december["date"], errors="coerce")
    if dates.isna().any() or len(december) != 31 or set(dates) != set(DECEMBER_DATES):
        raise ValueError("December output must contain exactly one row per day, 2025-12-01..2025-12-31")
    if dates.duplicated().any():
        raise ValueError("December output contains duplicate dates")
    for column, expected in DECEMBER_FIXED.items():
        values = december[column]
        ok = (values == expected).all() if isinstance(expected, str) else np.isclose(values, expected).all()
        if not ok:
            raise ValueError(f"December '{column}' must equal {expected!r} for every row")
    rates = pd.to_numeric(december["predicted_rate"], errors="coerce")
    if rates.isna().any() or (rates <= 0).any():
        raise ValueError("December predicted_rate must be positive and numeric for every row")


def generate_validation_predictions(
    model: HistGradientBoostingRegressor,
    engineer: FeatureEngineer,
    validation_df: pd.DataFrame,
    template_df: pd.DataFrame,
) -> pd.DataFrame:
    features = engineer.transform(validation_df)
    predicted_rate = inverse_target(model.predict(features), features["distance"].to_numpy())

    predictions = pd.DataFrame({
        "load_id": validation_df["load_id"].astype(str),
        "predicted_rate": predicted_rate,
    })
    filled_template = template_df[["load_id"]].merge(predictions, on="load_id", how="left")
    if filled_template["predicted_rate"].isna().any():
        raise ValueError("Could not produce a prediction for every load_id in the template.")
    assert_valid_predictions_output(filled_template)
    return filled_template


def generate_december_predictions(
    model: HistGradientBoostingRegressor,
    engineer: FeatureEngineer,
    december_df: pd.DataFrame,
) -> pd.DataFrame:
    result = december_df.copy()
    features = engineer.transform(result)
    result["predicted_rate"] = inverse_target(model.predict(features), features["distance"].to_numpy())
    result = result[DECEMBER_COLUMNS].sort_values("date").reset_index(drop=True)
    assert_valid_december_output(result)
    return result


def run_pipeline(
    train_test_path: Path,
    validation_path: Path,
    validation_template_path: Path,
    december_path: Path,
    output_dir: Path,
    tune: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(train_test_path, REQUIRED_TRAIN_COLUMNS, "train_test.csv").sort_values("date")
    validation_df = load_csv(validation_path, REQUIRED_INFERENCE_COLUMNS | {"load_id"}, "validation.csv")
    template_df = load_csv(validation_template_path, {"load_id", "predicted_rate"}, "validation_predictions_template.csv")
    december_df = load_csv(december_path, REQUIRED_INFERENCE_COLUMNS, "december_chart_inputs.csv")

    if len(template_df) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"Template must have {EXPECTED_VALIDATION_ROWS:,} rows, found {len(template_df)}")
    if len(december_df) != 31:
        raise ValueError(f"december_chart_inputs.csv must have 31 rows, found {len(december_df)}")

    engineer = FeatureEngineer().fit(train_df)
    X_train = engineer.transform(train_df)
    y_train = make_target(train_df["posted_rate"], train_df["distance"])

    if tune:
        model = tune_hyperparameters(X_train, y_train)
    else:
        model = build_model()

    LOGGER.info("Running honest fold-wise backtest (refits per fold; may take a minute)...")
    metrics = backtest_metrics(train_df, model.get_params())
    LOGGER.info(
        "Backtest -- RMSE=$%.2f  MAE=$%.2f  MAPE=%.2f%%  R2=%.4f",
        metrics["rmse"], metrics["mae"], metrics["mape"] * 100, metrics["r2"],
    )

    LOGGER.info("Refitting final model on all of train_test.csv...")
    model.fit(X_train, y_train)

    validation_predictions = generate_validation_predictions(model, engineer, validation_df, template_df)
    december_predictions = generate_december_predictions(model, engineer, december_df)

    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    december_predictions.to_csv(output_dir / "december_chart_inputs.csv", index=False)
    LOGGER.info("Saved validation_predictions.csv and december_chart_inputs.csv to %s", output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freight rate training + inference pipeline")
    parser.add_argument("--train-test", type=Path, default=Path("data/train_test.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.csv"))
    parser.add_argument("--validation-template", type=Path, default=Path("data/validation_predictions_template.csv"))
    parser.add_argument("--december", type=Path, default=Path("data/december_chart_inputs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--no-tune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_pipeline(
            train_test_path=args.train_test,
            validation_path=args.validation,
            validation_template_path=args.validation_template,
            december_path=args.december,
            output_dir=args.output_dir,
            tune=not args.no_tune,
        )
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
