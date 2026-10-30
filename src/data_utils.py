"""
data_utils.py
-------------
Shared data-loading, feature-engineering, and hotspot-aggregation
utilities for RoadGuard AI.

IMPORTANT — DATA LEAKAGE POLICY
================================
This dataset's `risk_score` was found (during dataset analysis) to be
almost fully explained by post-accident outcome columns
(`accident_severity`, `casualties`, `vehicles_involved`, `cause`).
A model trained on those columns would not be usable at prediction
time, because those facts are only known AFTER an accident happens.

Therefore, the ML model in this project is trained ONLY on features
that are known BEFORE or AT the time of a potential accident:

    location   -> city, state, latitude, longitude
    time       -> hour, day_of_week, is_weekend, is_peak_hour
    road       -> road_type, lanes, traffic_signal
    environment-> weather, visibility, temperature
    traffic    -> traffic_density
    calendar   -> has_festival (derived from `festival`)

`accident_severity`, `casualties`, `vehicles_involved`, and `cause`
are NEVER used as model inputs. `cause` and `accident_severity` are
still used for descriptive analytics (e.g. "top causes at this
hotspot"), which is safe because that's historical reporting, not a
live prediction input.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "accidents.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
HOTSPOTS_PATH = os.path.join(MODELS_DIR, "hotspots.csv")

# ---------------------------------------------------------------------
# Feature lists (single source of truth — used by train_model.py,
# app.py, and model_utils.py so nothing ever gets out of sync)
# ---------------------------------------------------------------------
NUMERIC_FEATURES = [
    "hour",
    "lanes",
    "temperature",
    "is_weekend",
    "is_peak_hour",
    "traffic_signal",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "day_of_week",
    "city",
    "state",
    "has_festival",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_RAW = "risk_score"          # 0-1 scale in the raw CSV
TARGET_SCALED = "risk_score_100"   # 0-100 scale used everywhere in the app

# Maps each model feature to a display "contribution group" used in the
# Hotspot view + SHAP explanation (Weather / Road / Traffic / Infrastructure)
FEATURE_GROUPS = {
    "weather": "Weather",
    "visibility": "Weather",
    "temperature": "Weather",
    "road_type": "Road",
    "lanes": "Road",
    "traffic_density": "Traffic",
    "is_peak_hour": "Traffic",
    "hour": "Traffic",
    "day_of_week": "Traffic",
    "is_weekend": "Traffic",
    "traffic_signal": "Infrastructure",
    "has_festival": "Infrastructure",
    "city": "Infrastructure",
    "state": "Infrastructure",
    "latitude": "Infrastructure",
    "longitude": "Infrastructure",
}

RISK_THRESHOLDS = [
    (0, 25, "LOW", "🟢"),
    (25, 50, "MEDIUM", "🟡"),
    (50, 75, "HIGH", "🟠"),
    (75, 101, "CRITICAL", "🔴"),
]

HOUR_BUCKETS = [
    (0, 6, "Late Night (12 AM - 6 AM)"),
    (6, 12, "Morning (6 AM - 12 PM)"),
    (12, 17, "Afternoon (12 PM - 5 PM)"),
    (17, 21, "Evening (5 PM - 9 PM)"),
    (21, 24, "Night (9 PM - 12 AM)"),
]


def risk_level(score_0_100: float) -> tuple[str, str]:
    """Return (level_label, emoji) for a 0-100 risk score."""
    for low, high, label, emoji in RISK_THRESHOLDS:
        if low <= score_0_100 < high:
            return label, emoji
    return "CRITICAL", "🔴"


def hour_bucket(hour: int) -> str:
    for low, high, label in HOUR_BUCKETS:
        if low <= hour < high:
            return label
    return "Unknown"


# ---------------------------------------------------------------------
# Loading & feature engineering
# ---------------------------------------------------------------------
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived, leakage-safe columns. Never mutates leakage columns."""
    df = df.copy()

    # Basic hygiene — guard against unexpected missing/odd values so the
    # app never crashes on a slightly different CSV.
    df["festival"] = df.get("festival", pd.Series(index=df.index, dtype=object))
    df["has_festival"] = df["festival"].notna().astype(int).astype(str)

    if TARGET_RAW in df.columns:
        df[TARGET_SCALED] = (df[TARGET_RAW].astype(float) * 100).round(1)

    df["hour_bucket"] = df["hour"].apply(hour_bucket)

    # Coerce categorical feature columns to string so OneHotEncoder never
    # trips on mixed types.
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def load_processed_data(path: str = DATA_PATH) -> pd.DataFrame:
    return engineer_features(load_raw_data(path))


# ---------------------------------------------------------------------
# Hotspot clustering
# ---------------------------------------------------------------------
def assign_hotspot_clusters(df: pd.DataFrame, clusters_per_city: int = 3) -> pd.Series:
    """
    Assign each accident record to a coordinate-based hotspot cluster
    (K-MEANS clustering on latitude/longitude, run separately per city —
    this is a geographic grouping algorithm, unrelated to the Random
    Forest risk-prediction model). Returns a Series of hotspot_id
    strings aligned to df's index (e.g. "Chennai - Cluster 2").

    Exposed as its own function (rather than only inline inside
    build_hotspots()) so other callers — e.g. the Hotspot Intelligence
    page's "Where x When" view — can look up which individual accident
    records belong to a specific cluster, without recomputing or
    duplicating the aggregate hotspot table.
    """
    ids = pd.Series(index=df.index, dtype=object)
    for city, sub in df.groupby("city"):
        n_clusters = min(clusters_per_city, max(1, sub.shape[0] // 20))
        n_clusters = max(1, n_clusters)
        coords = sub[["latitude", "longitude"]].values
        if n_clusters == 1 or len(sub) < n_clusters:
            labels = np.zeros(len(sub), dtype=int)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(coords)
        ids.loc[sub.index] = [f"{city} - Cluster {l + 1}" for l in labels]
    return ids


def build_hotspots(df: pd.DataFrame, predicted_risk_col: str = "predicted_risk",
                    clusters_per_city: int = 3) -> pd.DataFrame:
    """
    Group accident records into geographic hotspots.

    NOTE: The dataset has no official administrative "zone" column, so
    we do NOT invent zone names/boundaries. Instead we cluster each
    city's accidents by (latitude, longitude) using KMEANS (see
    assign_hotspot_clusters()), and label each cluster
    "<City> - Cluster <n>". This is clearly a coordinate-based
    grouping ("AI-derived geographic risk cluster"), not an official
    traffic-authority zone.

    (Behavior/output is unchanged from earlier versions of this
    function — the clustering loop was only extracted into
    assign_hotspot_clusters() for reuse; same random_state, same
    algorithm, same result.)
    """
    df = df.copy()
    df["hotspot_id"] = assign_hotspot_clusters(df, clusters_per_city=clusters_per_city)

    agg_rows = []
    for hotspot_id, sub in df.groupby("hotspot_id"):
        risk_col = predicted_risk_col if predicted_risk_col in sub.columns else TARGET_SCALED
        mean_risk = sub[risk_col].mean()
        level, emoji = risk_level(mean_risk)

        # High-risk period: hour bucket with highest mean risk at this hotspot
        bucket_means = sub.groupby("hour_bucket")[risk_col].mean().sort_values(ascending=False)
        high_risk_period = bucket_means.index[0] if len(bucket_means) else "N/A"

        # Peak day of week: day with the most accidents at this hotspot
        peak_day = (
            sub["day_of_week"].value_counts().index[0]
            if "day_of_week" in sub.columns and len(sub) else "N/A"
        )

        top_causes = (
            sub["cause"].value_counts().head(3).index.tolist()
            if "cause" in sub.columns else []
        )

        agg_rows.append({
            "hotspot_id": hotspot_id,
            "city": sub["city"].iloc[0],
            "state": sub["state"].iloc[0],
            "latitude": sub["latitude"].mean(),
            "longitude": sub["longitude"].mean(),
            "avg_risk_score": round(mean_risk, 1),
            "risk_level": level,
            "risk_emoji": emoji,
            "historical_accident_count": len(sub),
            "high_risk_period": high_risk_period,
            "peak_day": peak_day,
            "top_causes": ", ".join(top_causes),
            "dominant_weather": sub["weather"].mode().iloc[0] if "weather" in sub.columns and len(sub) else "N/A",
            "dominant_traffic_density": sub["traffic_density"].mode().iloc[0] if "traffic_density" in sub.columns and len(sub) else "N/A",
            "dominant_road_type": sub["road_type"].mode().iloc[0] if "road_type" in sub.columns and len(sub) else "N/A",
            "dominant_visibility": sub["visibility"].mode().iloc[0] if "visibility" in sub.columns and len(sub) else "N/A",
            "pct_traffic_signal": round(sub["traffic_signal"].mean(), 2) if "traffic_signal" in sub.columns and len(sub) else 0.0,
            "pct_peak_hour": round(sub["is_peak_hour"].mean(), 2) if "is_peak_hour" in sub.columns and len(sub) else 0.0,
        })

    hotspots = pd.DataFrame(agg_rows).sort_values("avg_risk_score", ascending=False).reset_index(drop=True)
    return hotspots


def load_hotspots() -> Optional[pd.DataFrame]:
    if os.path.exists(HOTSPOTS_PATH):
        return pd.read_csv(HOTSPOTS_PATH)
    return None


# ---------------------------------------------------------------------
# Hotspot Priority Score ("Road Safety Attention Priority")
# ---------------------------------------------------------------------
# NOT a validated accident-probability score. A transparent, documented
# heuristic ranking meant to surface locations combining historical
# volume, model-estimated risk, and peak-hour concentration — distinct
# from (and not a substitute for) the Random Forest's model risk score.
PRIORITY_WEIGHTS = {"risk": 0.5, "accident_volume": 0.3, "peak_hour_share": 0.2}


def compute_priority_score(hotspots: pd.DataFrame) -> pd.DataFrame:
    """
    "Road Safety Attention Priority" — a transparent ranking of hotspots,
    separate from the Random Forest MODEL RISK SCORE. A hotspot can have
    a moderate model risk score but still warrant attention because of
    high historical accident volume or a strong peak-hour concentration.

    Formula (weights chosen to foreground both model risk and real
    historical accident volume — documented here and shown in the UI,
    not a scientifically validated accident-probability measure):

        priority_score = 100 * (
            0.50 * (avg_risk_score / 100)                        # model risk, normalized
          + 0.30 * min-max normalized historical_accident_count   # historical volume
          + 0.20 * pct_peak_hour                                  # peak-hour concentration
        )

    All three inputs are real, already-computed hotspot fields (no
    fabricated data). Returns hotspots sorted by priority_score desc.
    """
    df = hotspots.copy()
    risk_norm = df["avg_risk_score"] / 100.0

    vol_min, vol_max = df["historical_accident_count"].min(), df["historical_accident_count"].max()
    vol_range = vol_max - vol_min
    vol_norm = (df["historical_accident_count"] - vol_min) / vol_range if vol_range > 0 else 0.0

    peak_norm = df["pct_peak_hour"] if "pct_peak_hour" in df.columns else 0.0

    df["priority_score"] = (
        PRIORITY_WEIGHTS["risk"] * risk_norm
        + PRIORITY_WEIGHTS["accident_volume"] * vol_norm
        + PRIORITY_WEIGHTS["peak_hour_share"] * peak_norm
    ) * 100
    df["priority_score"] = df["priority_score"].round(1)

    def _attention_label(score):
        if score >= 70:
            return "URGENT"
        if score >= 50:
            return "ELEVATED"
        if score >= 30:
            return "MODERATE"
        return "ROUTINE"

    df["attention_level"] = df["priority_score"].apply(_attention_label)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Safety Attention Alerts (historical-pattern alerts, NOT real-time)
# ---------------------------------------------------------------------
def generate_safety_alerts(hotspots: pd.DataFrame, top_n: int = 4) -> list[dict]:
    """
    Compact "Historical Pattern Alerts" derived from real hotspot fields
    — NOT real-time data (this system has no live traffic feed). Each
    alert names the specific hotspot and the specific data point behind
    it, so nothing here is invented.
    """
    alerts = []
    if len(hotspots) == 0:
        return alerts

    top_accidents = hotspots.sort_values("historical_accident_count", ascending=False).iloc[0]
    alerts.append({
        "type": "High Accident Concentration",
        "detail": f"{top_accidents['hotspot_id']} has the highest historical accident count "
                   f"({int(top_accidents['historical_accident_count'])} recorded accidents).",
    })

    if "pct_peak_hour" in hotspots.columns:
        peak_hs = hotspots.sort_values("pct_peak_hour", ascending=False).iloc[0]
        if peak_hs["pct_peak_hour"] >= 0.3:
            alerts.append({
                "type": "Peak-Hour Concentration",
                "detail": f"{peak_hs['hotspot_id']} has {peak_hs['pct_peak_hour']*100:.0f}% of its "
                           f"historical accidents occurring during peak hours.",
            })

    low_vis = hotspots[hotspots["dominant_visibility"] == "low"]
    if len(low_vis):
        row = low_vis.sort_values("avg_risk_score", ascending=False).iloc[0]
        alerts.append({
            "type": "Low-Visibility Hotspot",
            "detail": f"{row['hotspot_id']} predominantly experiences low-visibility conditions "
                       f"(risk score {row['avg_risk_score']}/100).",
        })

    high_traffic = hotspots[hotspots["dominant_traffic_density"] == "high"]
    if len(high_traffic):
        row = high_traffic.sort_values("avg_risk_score", ascending=False).iloc[0]
        alerts.append({
            "type": "High Traffic-Density Hotspot",
            "detail": f"{row['hotspot_id']} predominantly experiences high traffic density "
                       f"(risk score {row['avg_risk_score']}/100).",
        })

    critical = hotspots[hotspots["risk_level"] == "CRITICAL"]
    if len(critical):
        alerts.append({
            "type": "Critical Risk Level",
            "detail": f"{len(critical)} hotspot(s) are at CRITICAL model risk level "
                       f"(e.g. {critical.iloc[0]['hotspot_id']}).",
        })

    # de-duplicate by hotspot mentioned + type, preserve order, cap length
    seen = set()
    unique = []
    for a in alerts:
        if a["type"] not in seen:
            unique.append(a)
            seen.add(a["type"])
    return unique[:top_n]


# ---------------------------------------------------------------------
# System Insights — auto-generated, data-backed findings
# ---------------------------------------------------------------------
def generate_system_insights(df: pd.DataFrame, hotspots: pd.DataFrame) -> list[str]:
    """
    A handful of plain-language findings computed live from the actual
    dataset/hotspot table (never hard-coded). Every number here is
    recalculated each time this is called.
    """
    insights = []

    by_bucket = df.groupby("hour_bucket")["risk_score_100"].mean().sort_values(ascending=False)
    if len(by_bucket):
        insights.append(
            f"**{by_bucket.index[0]}** shows the highest average risk score "
            f"({by_bucket.iloc[0]:.0f}/100) across all recorded accidents."
        )

    if "traffic_density" in df.columns:
        by_traffic = df.groupby("traffic_density")["risk_score_100"].mean()
        if "high" in by_traffic.index and "low" in by_traffic.index:
            diff = by_traffic["high"] - by_traffic["low"]
            insights.append(
                f"High traffic-density conditions average a **{diff:.0f}-point higher** risk score "
                f"than low traffic-density conditions ({by_traffic['high']:.0f} vs {by_traffic['low']:.0f})."
            )

    if len(hotspots):
        top = hotspots.sort_values("historical_accident_count", ascending=False).iloc[0]
        insights.append(
            f"**{top['hotspot_id']}** has the highest historical accident concentration "
            f"({int(top['historical_accident_count'])} recorded accidents)."
        )

    if "visibility" in df.columns:
        by_vis = df.groupby("visibility")["risk_score_100"].mean()
        if "low" in by_vis.index and "high" in by_vis.index:
            diff = by_vis["low"] - by_vis["high"]
            insights.append(
                f"Low-visibility conditions average a **{diff:.0f}-point higher** risk score "
                f"than high-visibility conditions."
            )

    return insights


def recommendations_for(row: dict) -> list[str]:
    """
    Transparent rule-based recommendation engine.

    Accepts EITHER of two row shapes so it works for both a single
    Predict-Risk input and an aggregated hotspot row:

      Single prediction : traffic_density, visibility, weather,
                           road_type, traffic_signal (0/1), is_peak_hour (0/1)
      Hotspot aggregate  : dominant_traffic_density, dominant_visibility,
                           dominant_weather, dominant_road_type,
                           pct_traffic_signal (0-1 share), pct_peak_hour (0-1 share)
    """
    recs = []

    traffic_density = str(row.get("traffic_density", row.get("dominant_traffic_density", ""))).lower()
    visibility = str(row.get("visibility", row.get("dominant_visibility", ""))).lower()
    weather = str(row.get("weather", row.get("dominant_weather", ""))).lower()
    road_type = str(row.get("road_type", row.get("dominant_road_type", ""))).lower()

    # traffic_signal: single-row gives 0/1 presence; hotspot gives a share (0-1)
    if "traffic_signal" in row:
        signal_present = str(row.get("traffic_signal", 0)) in ("1", "1.0", "True", "true")
    else:
        signal_present = float(row.get("pct_traffic_signal", 1.0)) >= 0.5

    # is_peak_hour: single-row gives 0/1; hotspot gives a share (0-1) of peak-hour accidents
    if "is_peak_hour" in row:
        peak_hour_flag = str(row.get("is_peak_hour", 0)) in ("1", "1.0", "True", "true")
    else:
        peak_hour_flag = float(row.get("pct_peak_hour", 0.0)) >= 0.3

    if traffic_density == "high":
        recs.append("Increase traffic monitoring and enforcement during high-density periods.")

    if visibility == "low":
        recs.append("Improve street lighting and add reflective/warning signage to compensate for low visibility.")

    if weather in ("fog", "rain"):
        recs.append("Install weather-responsive warning signs and consider variable speed-limit signage for adverse weather.")

    if peak_hour_flag:
        recs.append("Deploy traffic personnel or automated enforcement during peak hours.")

    if not signal_present:
        recs.append("Evaluate this location for a traffic signal or a signed/controlled intersection.")

    if road_type == "highway":
        recs.append("Consider speed-control measures (rumble strips, speed cameras) appropriate for highway conditions.")

    if row.get("historical_accident_count", 0) and row["historical_accident_count"] > 0:
        recs.append("This location has a recorded accident history — recommend a targeted road-safety assessment.")

    if not recs:
        recs.append("No high-severity risk factors detected — maintain routine monitoring.")

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for r in recs:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique
