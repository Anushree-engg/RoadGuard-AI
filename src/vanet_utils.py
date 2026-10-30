"""
vanet_utils.py
---------------
A SIMULATED Vehicular Ad-hoc Network (VANET) layer, demonstrating how
vehicle- and roadside-unit-reported safety events could be integrated
with RoadGuard AI's existing hotspot data and trained risk model.

THIS IS A SOFTWARE SIMULATION, NOT REAL VANET HARDWARE OR REAL-TIME
VEHICLE DATA. Every vehicle, RSU, and event here is randomly generated
for demonstration purposes. The only REAL data used are:
  - actual hotspot locations/risk levels (models/hotspots.csv, via
    src/data_utils.py)
  - actual city coordinates (from the accident dataset)
  - the ACTUAL trained RandomForestRegressor, called (not retrained)
    to re-score risk under the simulated live conditions
  - the ACTUAL rule-based recommendation engine (recommendations_for)

Pipeline implemented:

    Simulated Vehicles -> V2V -> Roadside Unit (RSU) -> V2I ->
    RoadGuard AI (existing trained model) -> Hotspot Correlation ->
    Safety Attention Alert

Nothing here retrains, modifies, or fine-tunes the Random Forest model
or the K-Means hotspot clusters — this module only calls the existing
trained artifacts for on-demand inference, exactly like the Risk
Prediction and What-If pages already do.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data_utils import ALL_FEATURES, recommendations_for, risk_level

# ---------------------------------------------------------------------
# Simulation vocabulary
# ---------------------------------------------------------------------
# (event_type, weight, is_dangerous, v2v_message_template)
EVENT_TYPES = [
    ("Normal", 55, False, "All clear — normal driving conditions."),
    ("Sudden Braking", 12, True, "Sudden braking detected nearby."),
    ("Congestion", 12, True, "Heavy congestion building up ahead."),
    ("Accident Alert", 6, True, "Possible accident reported ahead — proceed with caution."),
    ("Poor Visibility", 8, True, "Poor visibility conditions reported."),
    ("Road Hazard", 7, True, "Road hazard reported on this stretch."),
]
EVENT_NAMES = [e[0] for e in EVENT_TYPES]
EVENT_WEIGHTS = [e[1] for e in EVENT_TYPES]
EVENT_DANGEROUS = {e[0]: e[2] for e in EVENT_TYPES}
EVENT_MESSAGE = {e[0]: e[3] for e in EVENT_TYPES}

TRAFFIC_LEVELS = ["low", "medium", "high"]
VISIBILITY_LEVELS = ["high", "medium", "low"]

RSU_ALERT_EVENT_THRESHOLD = 2   # >= this many dangerous events among connected vehicles -> RSU goes ACTIVE
RSU_RADIUS_KM = 10              # vehicles within this radius of an RSU are "connected"
HOTSPOT_CORRELATION_RADIUS_KM = 8  # RSU must be within this of a hotspot to correlate
V2V_RADIUS_KM = 4               # vehicles within this radius of each other can exchange V2V messages


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------
# 1. Simulated Vehicles
# ---------------------------------------------------------------------
def generate_vehicles(hotspots: pd.DataFrame, n_vehicles: int = 18, seed: int | None = None) -> pd.DataFrame:
    """
    Generate simulated vehicles. To make the hotspot-correlation demo
    meaningful, ~60% of vehicles are placed with small random jitter
    around REAL hotspot coordinates (weighted toward higher-risk
    hotspots), and the remainder are placed around random hotspot-area
    coordinates too (since we only have hotspot/city reference points
    in this dataset) — clearly a simulated placement, not real GPS data.

    Modeling assumption (stated here for transparency): vehicles placed
    near historically higher-risk hotspots are given a higher chance of
    reporting a "dangerous" event, reflecting the hotspot's historical
    risk — this is a simulation design choice, not a claim about real
    vehicles.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if len(hotspots) == 0:
        return pd.DataFrame()

    # weight hotspot selection by risk score so higher-risk hotspots
    # attract more simulated traffic (a deliberate, documented modeling
    # choice — see docstring)
    weights = (hotspots["avg_risk_score"].clip(lower=1)).to_numpy()
    weights = weights / weights.sum()

    rows = []
    now = datetime.now()
    for i in range(n_vehicles):
        hs = hotspots.iloc[np_rng.choice(len(hotspots), p=weights)]
        jitter_lat = np_rng.uniform(-0.015, 0.015)
        jitter_lon = np_rng.uniform(-0.015, 0.015)
        lat = hs["latitude"] + jitter_lat
        lon = hs["longitude"] + jitter_lon

        # bias event probability by this hotspot's historical risk
        risk_factor = min(1.0, hs["avg_risk_score"] / 100.0)
        dangerous_boost = 1.0 + risk_factor * 1.5
        adj_weights = [
            w * dangerous_boost if EVENT_DANGEROUS[name] else w
            for name, w in zip(EVENT_NAMES, EVENT_WEIGHTS)
        ]
        event_type = rng.choices(EVENT_NAMES, weights=adj_weights, k=1)[0]

        traffic_condition = rng.choices(TRAFFIC_LEVELS, weights=[2, 3, 3] if event_type != "Congestion" else [1, 2, 5], k=1)[0]
        visibility = rng.choices(VISIBILITY_LEVELS, weights=[3, 3, 2] if event_type != "Poor Visibility" else [1, 2, 6], k=1)[0]
        speed = rng.randint(45, 80) if event_type == "Normal" else rng.randint(5, 40)

        rows.append({
            "vehicle_id": f"V-{101 + i}",
            "latitude": lat,
            "longitude": lon,
            "speed_kmh": speed,
            "traffic_condition": traffic_condition,
            "visibility": visibility,
            "event_type": event_type,
            "is_dangerous": EVENT_DANGEROUS[event_type],
            "nearest_hotspot": hs["hotspot_id"],
            "nearest_hotspot_city": hs["city"],
            "timestamp": now - timedelta(seconds=rng.randint(0, 300)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. V2V — vehicle-to-vehicle safety messages
# ---------------------------------------------------------------------
def generate_v2v_messages(vehicles: pd.DataFrame, radius_km: float = V2V_RADIUS_KM, max_messages: int = 8) -> list[dict]:
    """
    For each vehicle reporting a dangerous event, find the nearest
    OTHER vehicle within radius_km and simulate a V2V broadcast from
    the reporting vehicle to it. Real V2V protocols broadcast to all
    neighbors; here we show one representative message per dangerous
    event for a readable demo log.
    """
    if len(vehicles) < 2:
        return []

    messages = []
    dangerous = vehicles[vehicles["is_dangerous"]]
    for _, v in dangerous.iterrows():
        others = vehicles[vehicles["vehicle_id"] != v["vehicle_id"]].copy()
        others["dist_km"] = haversine_km(v["latitude"], v["longitude"], others["latitude"], others["longitude"])
        nearby = others[others["dist_km"] <= radius_km].sort_values("dist_km")
        if len(nearby) == 0:
            continue
        target = nearby.iloc[0]
        messages.append({
            "from": v["vehicle_id"],
            "to": target["vehicle_id"],
            "event_type": v["event_type"],
            "message": EVENT_MESSAGE[v["event_type"]],
            "distance_km": round(target["dist_km"], 2),
            "timestamp": v["timestamp"],
        })

    messages.sort(key=lambda m: m["timestamp"], reverse=True)
    return messages[:max_messages]


# ---------------------------------------------------------------------
# 3 & 4. RSU simulation + V2I forwarding
# ---------------------------------------------------------------------
def generate_rsus(city_coord_map: dict, hotspots: pd.DataFrame) -> pd.DataFrame:
    """One simulated Roadside Unit per city, placed at the city's real
    mean accident coordinates (not a fabricated location)."""
    rows = []
    for city, coords in city_coord_map.items():
        if len(hotspots):
            dists = haversine_km(coords["latitude"], coords["longitude"],
                                  hotspots["latitude"].to_numpy(), hotspots["longitude"].to_numpy())
            nearest_idx = int(np.argmin(dists))
            nearest_hotspot = hotspots.iloc[nearest_idx]
            nearest_hotspot_dist = float(dists[nearest_idx])
        else:
            nearest_hotspot = None
            nearest_hotspot_dist = None

        rows.append({
            "rsu_id": f"RSU-{city}",
            "city": city,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "nearest_hotspot": nearest_hotspot["hotspot_id"] if nearest_hotspot is not None else "N/A",
            "nearest_hotspot_risk": nearest_hotspot["risk_level"] if nearest_hotspot is not None else "N/A",
            "nearest_hotspot_dist_km": round(nearest_hotspot_dist, 2) if nearest_hotspot_dist is not None else None,
        })
    return pd.DataFrame(rows)


def compute_rsu_status(vehicles: pd.DataFrame, rsus: pd.DataFrame, radius_km: float = RSU_RADIUS_KM,
                        alert_threshold: int = RSU_ALERT_EVENT_THRESHOLD) -> pd.DataFrame:
    """
    V2I step: each RSU 'collects' events from vehicles in its own city
    (Vehicle -> RSU) and aggregates them (RSU -> RoadGuard AI).

    NOTE ON DESIGN: vehicles are connected to the RSU serving their
    city (matched via nearest_hotspot_city), not a pure radius check
    against the RSU's coordinates. RSUs are placed at each city's mean
    accident coordinate, while vehicles are placed near specific
    hotspot CLUSTERS within that city — for geographically large
    cities those two points can be many km apart, so a fixed-radius
    check would under-count connections. City-based assignment is
    simpler and matches how build_event_log() already assigns
    vehicles to their RSU. `radius_km` is kept as a parameter for the
    RSU->hotspot proximity check elsewhere, not used for RSU->vehicle
    connectivity here.

    Returns rsus augmented with connected_vehicles, events_received,
    and alert_status.
    """
    rsus = rsus.copy()
    connected_counts, event_counts, statuses, dominant_events, dominant_traffic = [], [], [], [], []

    for _, rsu in rsus.iterrows():
        if len(vehicles) == 0:
            connected_counts.append(0); event_counts.append(0)
            statuses.append("NORMAL"); dominant_events.append("N/A"); dominant_traffic.append("N/A")
            continue
        connected = vehicles[vehicles["nearest_hotspot_city"] == rsu["city"]]
        dangerous = connected[connected["is_dangerous"]]

        connected_counts.append(len(connected))
        event_counts.append(len(dangerous))
        statuses.append("ACTIVE" if len(dangerous) >= alert_threshold else "NORMAL")
        dominant_events.append(dangerous["event_type"].mode().iloc[0] if len(dangerous) else "N/A")
        dominant_traffic.append(connected["traffic_condition"].mode().iloc[0] if len(connected) else "N/A")

    rsus["connected_vehicles"] = connected_counts
    rsus["events_received"] = event_counts
    rsus["alert_status"] = statuses
    rsus["dominant_event"] = dominant_events
    rsus["dominant_traffic"] = dominant_traffic
    return rsus


# ---------------------------------------------------------------------
# 5. RoadGuard AI integration + Hotspot Correlation + Safety Alerts
# ---------------------------------------------------------------------
def generate_safety_alerts(rsus: pd.DataFrame, vehicles: pd.DataFrame, hotspots: pd.DataFrame,
                            preprocessor=None, model=None,
                            max_hotspot_distance_km: float = HOTSPOT_CORRELATION_RADIUS_KM) -> list[dict]:
    """
    The core VANET -> RoadGuard AI integration step.

    For each ACTIVE RSU: take the dangerous events reported by vehicles
    connected to that RSU, find the REAL hotspot geographically nearest
    to those specific vehicles' mean position (not just the RSU's own
    city-centroid location — this keeps the correlation tied to where
    the danger was actually reported), and if that hotspot is within
    max_hotspot_distance_km, generate a Safety Attention Alert using:
      - that hotspot's REAL historical risk (from hotspots.csv)
      - optionally, the ACTUAL trained RandomForestRegressor re-scored
        on-demand with the simulated live conditions (NOT a retrain —
        one inference call, exactly like the Risk Prediction page)
      - the ACTUAL rule-based recommendation engine
    """
    alerts = []
    if len(rsus) == 0 or len(hotspots) == 0:
        return alerts

    active = rsus[rsus["alert_status"] == "ACTIVE"]
    for _, rsu in active.iterrows():
        connected = vehicles[vehicles["nearest_hotspot_city"] == rsu["city"]]
        dangerous = connected[connected["is_dangerous"]]
        if len(dangerous) == 0:
            continue

        # Correlate against the hotspot nearest to where the dangerous
        # events were actually reported, not the RSU's own location.
        mean_lat, mean_lon = dangerous["latitude"].mean(), dangerous["longitude"].mean()
        dists = haversine_km(mean_lat, mean_lon, hotspots["latitude"].to_numpy(), hotspots["longitude"].to_numpy())
        nearest_idx = int(np.argmin(dists))
        distance_km = float(dists[nearest_idx])
        if distance_km > max_hotspot_distance_km:
            continue
        hs = hotspots.iloc[nearest_idx]

        live_score, live_level = None, None
        if preprocessor is not None and model is not None:
            from src.model_utils import predict_risk_score
            live_traffic = connected["traffic_condition"].mode().iloc[0] if len(connected) else "medium"
            live_visibility = connected["visibility"].mode().iloc[0] if len(connected) else "medium"
            live_row = {
                "city": hs["city"], "state": hs["state"],
                "latitude": hs["latitude"], "longitude": hs["longitude"],
                "road_type": hs.get("dominant_road_type", "urban"),
                "lanes": 3,
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().strftime("%A"),
                "weather": hs.get("dominant_weather", "clear"),
                "visibility": live_visibility,
                "traffic_density": live_traffic,
                "temperature": 28,
                "traffic_signal": 1,
                "is_weekend": int(datetime.now().weekday() >= 5),
                "is_peak_hour": int(datetime.now().hour in [7, 8, 9, 17, 18, 19, 20]),
                "has_festival": "0",
            }
            try:
                input_df = pd.DataFrame([live_row])[ALL_FEATURES]
                live_score = float(predict_risk_score(input_df, preprocessor, model)[0])
                live_level, _ = risk_level(live_score)
            except Exception:
                live_score, live_level = None, None

        recs = recommendations_for({
            "traffic_density": connected["traffic_condition"].mode().iloc[0] if len(connected) else "medium",
            "visibility": connected["visibility"].mode().iloc[0] if len(connected) else "medium",
            "weather": hs.get("dominant_weather", ""),
            "road_type": hs.get("dominant_road_type", ""),
            "is_peak_hour": int(datetime.now().hour in [7, 8, 9, 17, 18, 19, 20]),
            "historical_accident_count": hs.get("historical_accident_count", 0),
        })

        alerts.append({
            "rsu_id": rsu["rsu_id"],
            "hotspot_id": hs["hotspot_id"],
            "dominant_event": dangerous["event_type"].mode().iloc[0],
            "events_received": len(dangerous),
            "distance_km": round(distance_km, 2),
            "historical_risk_level": hs["risk_level"],
            "historical_risk_score": hs["avg_risk_score"],
            "simulated_traffic": connected["traffic_condition"].mode().iloc[0] if len(connected) else "N/A",
            "live_score": live_score,
            "live_level": live_level,
            "recommendations": recs[:2],
        })

    return alerts


# ---------------------------------------------------------------------
# 8. Event log
# ---------------------------------------------------------------------
def build_event_log(vehicles: pd.DataFrame, rsus: pd.DataFrame) -> pd.DataFrame:
    """Time | Vehicle | RSU | Event | Nearby Hotspot | Status"""
    if len(vehicles) == 0:
        return pd.DataFrame(columns=["Time", "Vehicle", "RSU", "Event", "Nearby Hotspot", "Status"])

    rows = []
    for _, v in vehicles.iterrows():
        # find the nearest RSU (by city match is a good enough proxy since
        # RSUs are placed at city centroids and vehicles are generated
        # around hotspots within that same city)
        rsu_row = rsus[rsus["city"] == v["nearest_hotspot_city"]]
        rsu_id = rsu_row["rsu_id"].iloc[0] if len(rsu_row) else "N/A"
        status = "DANGEROUS" if v["is_dangerous"] else "Normal"
        rows.append({
            "Time": v["timestamp"].strftime("%H:%M:%S"),
            "Vehicle": v["vehicle_id"],
            "RSU": rsu_id,
            "Event": v["event_type"],
            "Nearby Hotspot": v["nearest_hotspot"],
            "Status": status,
        })
    log = pd.DataFrame(rows).sort_values("Time", ascending=False).reset_index(drop=True)
    return log
