"""
model_utils.py
---------------
Training, persistence, prediction, and SHAP-explanation helpers for
the RoadGuard AI risk model.

MODEL INPUT FORMAT
-------------------
A pandas DataFrame with exactly the columns in
`data_utils.NUMERIC_FEATURES + data_utils.CATEGORICAL_FEATURES`.
Categorical columns must be strings (engineer_features() handles this).

MODEL OUTPUT FORMAT
--------------------
predict_risk_score(df) -> np.ndarray of float risk scores on a 0-100 scale.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data_utils import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_SCALED,
)

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def train_and_save(df: pd.DataFrame) -> dict:
    """Train the RandomForest risk-regression model and persist all
    artifacts to disk. Returns the evaluation metrics dict."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    X = df[ALL_FEATURES]
    y = df[TARGET_SCALED]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # NOTE: tuned deliberately smaller/shallower than a default RF grid
    # search would suggest. Testing showed n_estimators=300/depth=None
    # scores the same (R2 ~0.88) as this smaller config but produces a
    # ~70MB pickle vs ~7MB here — with no accuracy benefit, since the
    # underlying signal (visibility/traffic_density/peak_hour) is not
    # complex enough to need deep trees.
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)

    preds = model.predict(X_test_t)
    metrics = {
        "model_type": "RandomForestRegressor (scikit-learn)",
        "task_type": "regression",
        "target": f"{TARGET_SCALED} (0-100 scale)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "R2": round(float(r2_score(y_test, preds)), 4),
    }

    feature_names = list(preprocessor.get_feature_names_out())

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    return metrics


def model_files_exist() -> bool:
    return os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH)


def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return preprocessor, model, feature_names, metrics


def predict_risk_score(df: pd.DataFrame, preprocessor, model) -> np.ndarray:
    """df must contain ALL_FEATURES columns. Returns risk scores (0-100),
    clipped to a valid range."""
    X = df[ALL_FEATURES]
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)
    return np.clip(preds, 0, 100)


def shap_explanation(df_row: pd.DataFrame, preprocessor, model, feature_names,
                      max_display: int = 8):
    """
    Compute SHAP values for a single-row DataFrame using a TreeExplainer
    on the fitted RandomForestRegressor. Returns a DataFrame of the
    top contributing (transformed) features and their SHAP values,
    sorted by absolute impact.

    Falls back gracefully (returns None) if shap is not installed or
    the explanation fails, so the app never crashes on this step.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        X_t = preprocessor.transform(df_row[ALL_FEATURES])
        # RandomForestRegressor -> TreeExplainer works directly on the raw model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_row = np.array(shap_values).reshape(-1)

        result = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_row,
        })
        result["abs_value"] = result["shap_value"].abs()
        result = result.sort_values("abs_value", ascending=False).head(max_display)
        return result.drop(columns="abs_value").reset_index(drop=True)
    except Exception:
        return None


def raw_feature_from_transformed(name: str) -> str:
    """Map a one-hot/passthrough transformed feature name (e.g.
    'cat__weather_fog' or 'num__hour') back to its original raw
    feature name (e.g. 'weather' or 'hour') for grouping/display."""
    if "__" in name:
        prefix, rest = name.split("__", 1)
    else:
        rest = name
    for feat in CATEGORICAL_FEATURES:
        if rest.startswith(feat + "_"):
            return feat
    for feat in NUMERIC_FEATURES:
        if rest == feat:
            return feat
    return rest


# ---------------------------------------------------------------------
# Display-label helpers (UI-only — does not affect model input/output)
# ---------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    "hour": "Hour of Day",
    "lanes": "Number of Lanes",
    "temperature": "Temperature",
    "is_weekend": "Weekend",
    "is_peak_hour": "Peak Hour",
    "traffic_signal": "Traffic Signal Present",
    "latitude": "Location Latitude",
    "longitude": "Location Longitude",
    "road_type": "Road Type",
    "weather": "Weather",
    "visibility": "Visibility",
    "traffic_density": "Traffic Density",
    "day_of_week": "Day of Week",
    "city": "City",
    "state": "State",
    "has_festival": "Festival / Holiday",
}


def humanize_feature_name(transformed_name: str) -> str:
    """
    Convert a pipeline-transformed feature name (e.g. 'cat__visibility_low',
    'num__is_peak_hour') into a human-readable label for display in the
    SHAP explanation chart (e.g. 'Low Visibility', 'Peak Hour').

    Purely cosmetic — does not change any SHAP values, model input, or
    model output. `raw_feature_from_transformed` (used elsewhere for
    grouping) is unaffected and still operates on the original names.
    """
    if "__" in transformed_name:
        prefix, rest = transformed_name.split("__", 1)
    else:
        prefix, rest = "num", transformed_name

    raw = raw_feature_from_transformed(transformed_name)
    base_label = FEATURE_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())

    if prefix == "num":
        return base_label

    # One-hot encoded column: rest = "<raw>_<value>" -> isolate the value
    value = rest[len(raw) + 1:] if rest.startswith(raw + "_") else rest
    value_label = value.replace("_", " ").title()

    if raw == "has_festival":
        return "Festival Day" if value == "1" else "No Festival"
    if raw in ("visibility", "traffic_density"):
        return f"{value_label} {base_label}"
    if raw == "weather":
        return f"{value_label} Weather"
    if raw == "road_type":
        return f"{value_label} Road"
    return f"{base_label}: {value_label}"


def contribution_by_group(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a SHAP result into the Weather / Road / Traffic /
    Infrastructure contribution groups, as normalized percentages of
    total absolute SHAP impact.

    IMPORTANT — caller contract: this function aggregates over exactly
    the rows present in `shap_df`. It does NOT independently guarantee
    "all features" — that depends entirely on what the caller passes
    in. `shap_explanation()` defaults to `max_display=8` (a top-N
    slice); if you pass that directly here, the percentages will only
    reflect those top 8 features, not the full feature set.

    To get percentages that reflect the model's FULL feature set (the
    intended behavior for the "Contribution by Category" summary),
    callers must first call:

        shap_df_full = shap_explanation(row, preprocessor, model,
                                         feature_names,
                                         max_display=len(feature_names))

    and pass `shap_df_full` here — then separately take
    `shap_df_full.head(8)` for the top-contributors bar chart, if a
    truncated display list is also wanted. app.py's
    render_shap_section() follows exactly this pattern.
    """
    from src.data_utils import FEATURE_GROUPS

    df = shap_df.copy()
    df["raw_feature"] = df["feature"].apply(raw_feature_from_transformed)
    df["group"] = df["raw_feature"].map(FEATURE_GROUPS).fillna("Other")
    df["abs_value"] = df["shap_value"].abs()

    grouped = df.groupby("group")["abs_value"].sum()
    total = grouped.sum()
    if total == 0:
        return pd.DataFrame(columns=["group", "pct"])
    pct = (grouped / total * 100).round(1).reset_index()
    pct.columns = ["group", "pct"]
    return pct.sort_values("pct", ascending=False)


def shap_plain_english_summary(shap_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """
    Turn the top-N SHAP contributors into simple, non-technical
    sentences for a general audience, e.g.:
        "High Traffic Density increased the predicted risk."
        "High Visibility decreased the predicted risk."

    This describes the SHAP attribution for ONE specific prediction —
    it does not imply causation, and does not claim the feature causes
    higher/lower risk in general.
    """
    if shap_df is None or len(shap_df) == 0:
        return []
    ordered = shap_df.reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index)
    out = []
    for _, r in ordered.head(top_n).iterrows():
        label = humanize_feature_name(r["feature"])
        verb = "increased" if r["shap_value"] >= 0 else "decreased"
        out.append(f"{label} {verb} the predicted risk for this scenario.")
    return out
