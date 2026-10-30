"""
RoadGuard AI — Traffic Safety Intelligence Dashboard
=====================================================
Run with:
    streamlit run app.py

UI/UX NOTE:
This file is the presentation layer only. Every call into
src/data_utils.py and src/model_utils.py is unchanged from the
original prototype — same function names, same signatures, same
return values. Only layout, navigation, styling, and copy have been
reworked. See README.md for full setup / architecture notes.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import (
    ALL_FEATURES,
    DATA_PATH,
    PRIORITY_WEIGHTS,
    assign_hotspot_clusters,
    build_hotspots,
    compute_priority_score,
    generate_safety_alerts,
    generate_system_insights,
    load_hotspots,
    load_processed_data,
    recommendations_for,
    risk_level,
)
from src.model_utils import (
    contribution_by_group,
    humanize_feature_name,
    load_artifacts,
    model_files_exist,
    predict_risk_score,
    shap_explanation,
    shap_plain_english_summary,
    train_and_save,
)
from src import vanet_utils

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------
# Muted, professional risk palette (no neon). Kept as a single source of
# truth used by both the CSS below and every chart/badge in the app.
RISK_COLORS = {
    "LOW": "#1E8A4C",       # green
    "MEDIUM": "#B4790E",    # amber
    "HIGH": "#C2560C",      # orange
    "CRITICAL": "#B3261E",  # red
}
BRAND_PRIMARY = "#1B4B5A"     # deep teal-blue — traffic authority / command-center accent
BRAND_PRIMARY_DK = "#123544"
SIDEBAR_BG = "#122733"

CUSTOM_CSS = f"""
<style>
/* ---------- layout ---------- */
.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1360px;
}}
html, body, [class*="css"] {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

/* ---------- header ---------- */
.rg-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-size: 0.72rem;
    font-weight: 700;
    color: {BRAND_PRIMARY};
    margin-bottom: 2px;
}}
.rg-title {{
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 2px 0;
    color: #16232b;
}}
.rg-subtitle {{
    color: #5b6670;
    font-size: 0.95rem;
    margin: 0 0 1.1rem 0;
}}
.rg-section-label {{
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-size: 0.74rem;
    font-weight: 700;
    color: {BRAND_PRIMARY};
    border-bottom: 2px solid {BRAND_PRIMARY};
    display: inline-block;
    padding-bottom: 3px;
    margin: 1.4rem 0 0.7rem 0;
}}

/* ---------- risk badge ---------- */
.rg-badge {{
    display: inline-block;
    padding: 3px 13px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.3px;
}}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 1px 2px rgba(16,24,32,0.04);
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78rem !important;
    color: #5b6670 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    color: #16232b !important;
}}

/* ---------- generic bordered containers (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px !important;
}}

/* ---------- buttons (main content) ---------- */
.stButton>button {{
    border-radius: 8px;
}}
.stButton>button[kind="primary"] {{
    background-color: {BRAND_PRIMARY};
    border-color: {BRAND_PRIMARY};
}}
.stButton>button[kind="primary"]:hover {{
    background-color: {BRAND_PRIMARY_DK};
    border-color: {BRAND_PRIMARY_DK};
}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1.2rem;
}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
    color: #dbe6ec;
}}
.rg-sidebar-brand {{
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    color: #ffffff;
    line-height: 1.3;
}}
.rg-sidebar-sub {{
    font-size: 0.72rem;
    color: #86a0ae;
    margin-bottom: 0.6rem;
    letter-spacing: 0.2px;
}}
.rg-nav-section {{
    font-size: 0.66rem;
    letter-spacing: 1.2px;
    color: #6f8a99;
    font-weight: 700;
    text-transform: uppercase;
    margin: 1.1rem 0 0.3rem 4px;
}}
[data-testid="stSidebar"] hr {{
    border-color: #24485c;
    margin: 0.8rem 0;
}}
/* nav buttons: full-width, left-aligned, flat list appearance (no radio circles) */
[data-testid="stSidebar"] .stButton>button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background-color: transparent;
    border: 1px solid transparent;
    color: #d3e0e8;
    font-weight: 500;
    font-size: 0.88rem;
    border-radius: 7px;
    padding: 0.42rem 0.65rem;
    box-shadow: none;
}}
[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: #1c3d4f;
    color: #ffffff;
    border-color: #2c5468;
}}
[data-testid="stSidebar"] .stButton>button[kind="primary"] {{
    background-color: #2c6e8e;
    color: #ffffff;
    border-color: #2c6e8e;
    font-weight: 700;
}}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover {{
    background-color: #337c9e;
}}

/* ---------- dataframe / table polish ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid #e2e6ea;
    border-radius: 8px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def risk_badge_html(level: str, emoji: str = "") -> str:
    """Small colored pill for a risk level. `emoji` kept optional/unused
    by default to match the low-emoji design direction, but accepted for
    backward compatibility with callers that still pass one."""
    color = RISK_COLORS.get(level, "#666")
    return (
        f'<span class="rg-badge" style="background-color:{color}1a;'
        f'color:{color};border:1px solid {color};">{level}</span>'
    )


def section_label(text: str):
    st.markdown(f'<div class="rg-section-label">{text}</div>', unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f'<div class="rg-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="rg-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="rg-subtitle">{subtitle}</p>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached data / model loading (auto-trains model on first run)
# — UNCHANGED from the original app: same functions, same caching.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return load_processed_data(DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_model_artifacts():
    if not model_files_exist():
        df = get_data()
        with st.spinner("First run detected — training risk model, please wait..."):
            train_and_save(df)
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_hotspots(_preprocessor, _model):
    hotspots = load_hotspots()
    if hotspots is not None:
        return hotspots
    df = get_data()
    df = df.copy()
    df["predicted_risk"] = predict_risk_score(df, _preprocessor, _model)
    hotspots = build_hotspots(df, predicted_risk_col="predicted_risk")
    os.makedirs("models", exist_ok=True)
    hotspots.to_csv("models/hotspots.csv", index=False)
    return hotspots


@st.cache_data(show_spinner=False)
def get_cluster_assignments(df):
    """Per-accident-record hotspot_id assignment (K-MEANS geographic
    clustering) — used only by the Hotspot Intelligence 'Where x When'
    view to look up which historical records belong to a selected
    cluster. Cached since KMeans over 20k rows is the only nontrivial
    cost in this app outside of model training."""
    return assign_hotspot_clusters(df)


# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "last_input_row" not in st.session_state:
    st.session_state.last_input_row = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_shap_df" not in st.session_state:
    st.session_state.last_shap_df = None
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "command_center"
if "predict_reveal" not in st.session_state:
    st.session_state.predict_reveal = None  # None | "factors" | "actions"
if "vanet_seed" not in st.session_state:
    st.session_state.vanet_seed = 42


# ----------------------------------------------------------------------
# LOGIN SCREEN
# ----------------------------------------------------------------------
def login_screen():
    st.markdown('<div style="height:6vh"></div>', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown('<div class="rg-eyebrow">TRAFFIC SAFETY INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<p class="rg-title" style="font-size:2.3rem;">🚦 RoadGuard AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="rg-subtitle">AI-powered accident risk monitoring, hotspot detection '
            'and prevention support.</p>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("**Sign in**")
            with st.form("login_form"):
                st.text_input("Email", placeholder="analyst@roadguard.ai")
                st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button(
                    "Log In", use_container_width=True, type="primary"
                )
                if submitted:
                    st.session_state.logged_in = True
                    st.rerun()
            st.caption(
                "Demo login for this hackathon prototype — no real authentication is performed."
            )
        st.markdown("")
        with st.container(border=True):
            st.markdown("**About this system**")
            st.caption(
                "RoadGuard AI analyzes historical accident data to estimate accident risk, "
                "surface high-risk hotspots, explain why a location is risky, and recommend "
                "safety interventions — all computed from real data, with no fabricated numbers."
            )


# ----------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------
NAV_STRUCTURE = [
    ("MAIN", [
        ("command_center", "Command Center"),
        ("hotspot_intel", "Hotspot Intelligence"),
        ("risk_prediction", "Risk Prediction"),
        ("analytics", "Analytics"),
    ]),
    ("DECISION SUPPORT", [
        ("what_if", "What-If Simulator"),
        ("model_performance", "Model Performance"),
    ]),
    ("CONNECTED VEHICLES", [
        ("vanet", "VANET Safety Network"),
    ]),
]


def sidebar_nav():
    with st.sidebar:
        st.markdown('<div class="rg-sidebar-brand">🚦 ROADGUARD AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="rg-sidebar-sub">Traffic Safety Intelligence</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        for section_name, items in NAV_STRUCTURE:
            st.markdown(f'<div class="rg-nav-section">{section_name}</div>', unsafe_allow_html=True)
            for key, label in items:
                is_active = st.session_state.nav_page == key
                if st.button(
                    label,
                    key=f"nav_{key}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.nav_page = key
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Log out", key="nav_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.nav_page = "command_center"
            st.rerun()

    return st.session_state.nav_page


def goto(page_key: str):
    st.session_state.nav_page = page_key
    st.rerun()


# ----------------------------------------------------------------------
# Shared: Folium hotspot map
# — Map data/markers are unchanged (same hotspot fields, same
#   recommendations_for() calls). Only the tile provider changed: the
#   old "cartodbpositron" alias was showing "API key required" tile
#   watermarks (a known issue with that folium built-in alias on some
#   installs). Replaced with an explicit, no-key CARTO Positron tile
#   URL — same visual style, but doesn't depend on folium's alias
#   resolution.
# ----------------------------------------------------------------------
def render_folium_map(hotspots_df: pd.DataFrame, height: int = 460, key: str = "map"):
    import folium
    from streamlit_folium import st_folium

    if len(hotspots_df) == 0:
        st.info("No hotspots match the current filters.")
        return

    center_lat = hotspots_df["latitude"].mean()
    center_lon = hotspots_df["longitude"].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        name="Light",
        control=False,
    ).add_to(fmap)

    for _, row in hotspots_df.iterrows():
        color = RISK_COLORS.get(row["risk_level"], "#666")
        recs = recommendations_for(row.to_dict())
        rec_html = "".join(f"<li>{r}</li>" for r in recs[:4])
        popup_html = f"""
        <div style="font-family:'Segoe UI',Arial,sans-serif; width:240px;">
            <b>{row['hotspot_id']}</b><br>
            <span style="color:{color};font-weight:700;">{row['risk_level']}</span>
            &nbsp;·&nbsp;Risk Score: <b>{row['avg_risk_score']}/100</b><br>
            <b>Historical Accidents:</b> {row['historical_accident_count']}<br>
            <b>Peak Risk Period:</b> {row['high_risk_period']}<br>
            <b>Top Causes:</b> {row['top_causes']}<br>
            <b>Conditions:</b> {row['dominant_weather']}, {row['dominant_traffic_density']} traffic, {row['dominant_road_type']}<br>
            <b>Recommended:</b>
            <ul style="margin:4px 0 0 -18px;">{rec_html}</ul>
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7 + (row["avg_risk_score"] / 14),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.78,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['hotspot_id']} — {row['risk_level']}",
        ).add_to(fmap)

    st_folium(fmap, width=None, height=height, returned_objects=[], key=key)


def risk_legend():
    l1, l2, l3, l4 = st.columns(4)
    for col, (level, color) in zip((l1, l2, l3, l4), RISK_COLORS.items()):
        col.markdown(
            f'<span style="color:{color};font-weight:700;">●</span> {level}',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# Shared: VANET network map (vehicles, RSUs, hotspots) — a DIFFERENT
# map from the risk hotspot map on Hotspot Intelligence: different
# markers, different purpose (network topology, not risk severity).
# Kept visually distinct (different tile/marker styling) so it's never
# confused with the primary GIS map.
# ----------------------------------------------------------------------
def render_vanet_map(vehicles, rsus, hotspots, height=460, key="vanet_map"):
    import folium
    from streamlit_folium import st_folium

    if len(hotspots) == 0:
        st.info("No hotspot data available for the VANET map.")
        return

    center_lat, center_lon = hotspots["latitude"].mean(), hotspots["longitude"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        name="Light", control=False,
    ).add_to(fmap)

    # Reference hotspots — small gray markers, for correlation context only
    for _, hs in hotspots.iterrows():
        folium.CircleMarker(
            location=[hs["latitude"], hs["longitude"]],
            radius=4, color="#9aa5b1", fill=True, fill_color="#9aa5b1", fill_opacity=0.5,
            weight=1, tooltip=f"Hotspot: {hs['hotspot_id']} ({hs['risk_level']})",
        ).add_to(fmap)

    # RSUs — brand-colored square-ish markers
    for _, rsu in rsus.iterrows():
        rsu_color = RISK_COLORS["CRITICAL"] if rsu["alert_status"] == "ACTIVE" else BRAND_PRIMARY
        folium.CircleMarker(
            location=[rsu["latitude"], rsu["longitude"]],
            radius=10, color=rsu_color, fill=True, fill_color=rsu_color, fill_opacity=0.85,
            weight=2,
            tooltip=f"{rsu['rsu_id']} — {rsu['alert_status']} ({rsu['connected_vehicles']} vehicles)",
        ).add_to(fmap)

    # Vehicles — small markers, red if dangerous event, green if normal
    for _, v in vehicles.iterrows():
        v_color = RISK_COLORS["CRITICAL"] if v["is_dangerous"] else RISK_COLORS["LOW"]
        folium.CircleMarker(
            location=[v["latitude"], v["longitude"]],
            radius=5, color=v_color, fill=True, fill_color=v_color, fill_opacity=0.9,
            weight=1,
            tooltip=f"{v['vehicle_id']} — {v['event_type']}",
        ).add_to(fmap)

    st_folium(fmap, width=None, height=height, returned_objects=[], key=key)


# ----------------------------------------------------------------------
# Shared: recommendation priority cards
# ----------------------------------------------------------------------
def render_priority_cards(recs: list[str]):
    for i, r in enumerate(recs, start=1):
        with st.container(border=True):
            c1, c2 = st.columns([0.09, 0.91])
            c1.markdown(
                f'<div style="font-size:1.3rem;font-weight:800;color:{BRAND_PRIMARY};">{i:02d}</div>',
                unsafe_allow_html=True,
            )
            c2.markdown(r)


# ----------------------------------------------------------------------
# Shared: hotspot detail (card layout)
# — Same underlying fields + same recommendations_for() call as the
#   original render_hotspot_detail(); only the presentation changed.
# ----------------------------------------------------------------------
def render_hotspot_detail(row):
    st.markdown(f"### {row['hotspot_id']}")
    st.caption(
        f"{row['city']}, {row['state']} · AI-derived geographic risk cluster "
        "(K-Means clustering on coordinates) — not an official administrative zone."
    )
    b1, b2 = st.columns([1, 1])
    b1.markdown(risk_badge_html(row["risk_level"]), unsafe_allow_html=True)
    b1.caption("Model Risk Score (Random Forest)")
    if "attention_level" in row:
        b2.markdown(f'<span class="rg-badge" style="background-color:{BRAND_PRIMARY}1a;'
                    f'color:{BRAND_PRIMARY};border:1px solid {BRAND_PRIMARY};">{row["attention_level"]}</span>',
                    unsafe_allow_html=True)
        b2.caption("Attention Priority")
    st.markdown(f"##### {row['avg_risk_score']} / 100")
    if "priority_score" in row:
        st.caption(f"Road Safety Attention Priority: **{row['priority_score']}/100** (see formula above the table)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.caption("Historical Accidents")
            st.markdown(f"### {int(row['historical_accident_count'])}")
    with c2:
        with st.container(border=True):
            st.caption("Peak Risk Period")
            st.markdown(f"##### {row['high_risk_period']}")
    with c3:
        with st.container(border=True):
            st.caption("Peak Day")
            st.markdown(f"##### {row.get('peak_day', 'N/A')}")
    with c4:
        with st.container(border=True):
            st.caption("Dominant Conditions")
            st.markdown(
                f"##### {row['dominant_weather']} · {row['dominant_traffic_density']} traffic · {row['dominant_road_type']}"
            )

    d1, d2, d3 = st.columns(3)
    d1.metric("Dominant Visibility", str(row.get("dominant_visibility", "N/A")).title())
    d2.metric("Traffic Signal Present", f"{row.get('pct_traffic_signal', 0)*100:.0f}% of records")
    d3.metric("Peak-Hour Share", f"{row.get('pct_peak_hour', 0)*100:.0f}% of records")

    section_label("Why This Location Matters")
    st.markdown(
        f"- Historically dominant conditions here: **{row['dominant_weather']} weather**, "
        f"**{row['dominant_traffic_density']} traffic density**, on a **{row['dominant_road_type']} road** "
        f"(**{row['dominant_visibility']} visibility** typical).\n"
        f"- Most common recorded causes: **{row['top_causes']}**.\n"
        f"- Risk peaks during **{row['high_risk_period']}**, most often on **{row.get('peak_day', 'N/A')}**, "
        f"based on {int(row['historical_accident_count'])} historical accident records at this cluster."
    )

    section_label("Recommended Interventions")
    st.caption("Transparent rule-based logic (not machine-learned) — grounded in this hotspot's actual dominant conditions.")
    render_priority_cards(recommendations_for(row.to_dict()))


# ----------------------------------------------------------------------
# PAGE: COMMAND CENTER (was "Dashboard")
# Purpose: "What is happening overall? Do I need to pay attention to
# anything right now?" — an executive summary only. The detailed GIS
# map lives ONLY on Hotspot Intelligence; detailed historical charts
# live ONLY on Analytics. This page intentionally does not repeat them.
# ----------------------------------------------------------------------
def page_command_center(df, hotspots, metrics):
    page_header(
        "COMMAND CENTER",
        "RoadGuard AI",
        "Executive summary — accident risk monitoring, hotspot detection and prevention support.",
    )

    total_accidents = len(df)
    high_critical = hotspots[hotspots["risk_level"].isin(["HIGH", "CRITICAL"])].shape[0]
    peak_bucket = df.groupby("hour_bucket")["risk_score_100"].mean().sort_values(ascending=False).index[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Historical Accidents", f"{total_accidents:,}")
    k2.metric("AI-Derived Risk Hotspots", len(hotspots))
    k3.metric("High / Critical Hotspots", high_critical)
    k4.metric("Peak Risk Period", peak_bucket.split(" (")[0])

    st.markdown("")
    ranked = compute_priority_score(hotspots)

    left, right = st.columns([1.3, 1])
    with left:
        section_label("Priority Attention — Top 3 Locations")
        st.caption(
            "Ranked by **Road Safety Attention Priority** (see formula on Hotspot Intelligence) — "
            "not the same as model risk score alone. Full ranking and map: Hotspot Intelligence."
        )
        for i, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
            with st.container(border=True):
                c1, c2, c3 = st.columns([0.08, 0.62, 0.3])
                c1.markdown(f"**{i}**")
                c2.markdown(f"**{row['hotspot_id']}**")
                c2.caption(
                    f"{int(row['historical_accident_count'])} historical accidents · "
                    f"peak {row['high_risk_period'].split(' (')[0]}"
                )
                c3.markdown(risk_badge_html(row["risk_level"]), unsafe_allow_html=True)
                c3.caption(f"Priority {row['priority_score']}/100 · {row['attention_level']}")

    with right:
        section_label("Safety Attention Alerts")
        st.caption("Historical-pattern alerts derived from hotspot data — not real-time monitoring.")
        alerts = generate_safety_alerts(hotspots)
        if not alerts:
            st.info("No notable alerts under current thresholds.")
        for a in alerts:
            with st.container(border=True):
                st.markdown(f"**{a['type']}**")
                st.caption(a["detail"])

    st.markdown("")
    section_label("System Insights")
    st.caption("Auto-generated from the current dataset and hotspot table — recalculated each run, never hard-coded.")
    insights = generate_system_insights(df, hotspots)
    ic1, ic2 = st.columns(2)
    for i, text in enumerate(insights):
        (ic1 if i % 2 == 0 else ic2).markdown(f"- {text}")

    st.markdown("")
    section_label("Risk Distribution")
    c1, c2 = st.columns(2)
    with c1:
        by_hour = df.groupby("hour")["risk_score_100"].mean().reset_index()
        fig = px.line(by_hour, x="hour", y="risk_score_100", markers=True,
                      labels={"hour": "Hour", "risk_score_100": "Avg Risk"})
        fig.update_traces(line_color=BRAND_PRIMARY)
        fig.update_layout(title="Peak Risk Timeline", height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        counts = hotspots["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map=RISK_COLORS,
                     labels={"x": "Risk Level", "y": "Hotspots"})
        fig.update_layout(title="Hotspot Risk Level Distribution", showlegend=False, height=300,
                           margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "For the detailed GIS map and hotspot investigation, go to **Hotspot Intelligence**. "
        "For full historical breakdowns, go to **Analytics**."
    )


# ----------------------------------------------------------------------
# PAGE: HOTSPOT INTELLIGENCE (was "Hotspot Map")
# Purpose: "Where AND WHEN is accident risk concentrated?" — the ONE
# and ONLY page with the main GIS map. Also home of the Priority
# Ranking and Where x When analysis (kept here, not duplicated
# elsewhere, since both are inherently location-based).
# ----------------------------------------------------------------------
def page_hotspot_intelligence(df, hotspots):
    page_header(
        "GEOSPATIAL RISK ANALYSIS",
        "Hotspot Intelligence",
        "AI-derived geographic clusters (K-Means clustering on accident coordinates, per city) — "
        "the dataset has no official administrative zones, so these are coordinate-based clusters only.",
    )

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        city_filter = st.selectbox("City", ["All Cities"] + sorted(hotspots["city"].unique().tolist()))
    with f2:
        level_filter = st.selectbox("Risk Level", ["All Levels", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    with f3:
        st.caption(
            "Time-period filtering isn't available here: hotspots are aggregated across the "
            "full historical dataset. See **Where x When** below for time patterns per hotspot, "
            "or **Analytics → Time Analysis** for system-wide time-of-day patterns."
        )

    filtered = hotspots.copy()
    if city_filter != "All Cities":
        filtered = filtered[filtered["city"] == city_filter]
    if level_filter != "All Levels":
        filtered = filtered[filtered["risk_level"] == level_filter]

    section_label("Risk Hotspot Map")
    render_folium_map(filtered, height=480, key="hi_map")
    risk_legend()

    st.markdown("")
    section_label(f"Priority Ranking ({len(filtered)} of {len(hotspots)} hotspots)")
    st.caption(
        "**Road Safety Attention Priority** — NOT the model risk score. A transparent ranking: "
        f"{PRIORITY_WEIGHTS['risk']*100:.0f}% normalized model risk score + "
        f"{PRIORITY_WEIGHTS['accident_volume']*100:.0f}% normalized historical accident count + "
        f"{PRIORITY_WEIGHTS['peak_hour_share']*100:.0f}% peak-hour accident share. "
        "A hotspot can carry a moderate model risk score yet rank high "
        "in priority because of historical accident volume or peak-hour concentration. "
        "Normalized against the full hotspot set, so scores stay consistent across filters and pages."
    )
    # Priority score is computed once against the FULL (unfiltered) hotspot
    # set so its normalization (and thus every score) stays identical
    # everywhere it's shown — Command Center's Top 3 and this table always
    # agree on a given hotspot's priority_score. We filter the already-
    # ranked frame, rather than re-ranking the filtered subset.
    ranked_full = compute_priority_score(hotspots)
    ranked = ranked_full[ranked_full["hotspot_id"].isin(filtered["hotspot_id"])].reset_index(drop=True)
    if len(ranked):
        table = ranked.reset_index(drop=True).copy()
        table.index = table.index + 1
        st.dataframe(
            table[["hotspot_id", "historical_accident_count", "avg_risk_score", "priority_score",
                   "attention_level", "high_risk_period"]]
            .rename(columns={
                "hotspot_id": "Location", "historical_accident_count": "Historical Accidents",
                "avg_risk_score": "Model Risk Score", "priority_score": "Priority Score",
                "attention_level": "Attention Level", "high_risk_period": "Peak Risk Period",
            }),
            use_container_width=True,
        )
    else:
        st.info("No hotspots match the current filters.")
        return

    st.markdown("")
    section_label("Hotspot Detail")
    chosen = st.selectbox("Select a hotspot", ranked["hotspot_id"].tolist(), key="hi_select")
    row = ranked[ranked["hotspot_id"] == chosen].iloc[0]
    with st.container(border=True):
        render_hotspot_detail(row)

    st.markdown("")
    section_label("Where × When — Risk Pattern for This Hotspot")
    st.caption(
        "Historical/model-based analysis for the selected cluster only — not a forecast of exact future accidents."
    )
    cluster_ids = get_cluster_assignments(df)
    cluster_rows = df[cluster_ids == chosen]
    if len(cluster_rows) == 0:
        st.info("No individual accident records matched this cluster (unexpected — filters may be stale).")
    else:
        wc1, wc2 = st.columns([1, 2])
        with wc1:
            peak_hour_row = cluster_rows.groupby("hour")["risk_score_100"].mean().sort_values(ascending=False)
            peak_hour = int(peak_hour_row.index[0]) if len(peak_hour_row) else None
            peak_day = cluster_rows["day_of_week"].value_counts().index[0] if len(cluster_rows) else "N/A"
            with st.container(border=True):
                st.caption("Peak Hour (highest avg. model risk)")
                st.markdown(f"### {peak_hour:02d}:00" if peak_hour is not None else "N/A")
            with st.container(border=True):
                st.caption("Peak Day (by accident count)")
                st.markdown(f"### {peak_day}")
        with wc2:
            by_hour = cluster_rows.groupby("hour").size().reset_index(name="accidents")
            fig = px.bar(by_hour, x="hour", y="accidents",
                         labels={"hour": "Hour of Day", "accidents": "Accident Count"})
            fig.update_traces(marker_color=BRAND_PRIMARY)
            fig.update_layout(title=f"Accident Count by Hour — {chosen}", height=280, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# PAGE: RISK PREDICTION (merges Predict Risk + Explain Prediction +
# Intervention Planner's "from prediction" tab into one workflow)
# ----------------------------------------------------------------------
def build_input_row(city_state_map, key_prefix=""):
    section_label("Location")
    l1, l2 = st.columns(2)
    with l1:
        city = st.selectbox("City", sorted(city_state_map.keys()), key=f"{key_prefix}city")
    with l2:
        # NOTE: state is derived fresh from city_state_map on every rerun
        # (not read back from a widget's own session_state), and shown
        # as plain markdown rather than a `key`-ed text_input — a
        # disabled text_input with a fixed `key` only honors `value=`
        # on its FIRST render in Streamlit, so it would otherwise keep
        # showing whichever state was selected first (e.g. "Karnataka"
        # for the default city) even after the user changes City. This
        # was a real display bug in the previous version; it did not
        # affect the actual prediction (which always used the freshly
        # computed `state` variable below), but it was confusing.
        state = city_state_map[city]
        st.markdown(f"**State**  \n{state}")

    section_label("Time")
    t1, t2, t3 = st.columns(3)
    with t1:
        hour = st.slider("Hour of Day", 0, 23, 18, key=f"{key_prefix}hour")
    with t2:
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            key=f"{key_prefix}dow",
        )
    with t3:
        is_weekend = st.checkbox("Weekend", value=False, key=f"{key_prefix}weekend")
        is_peak_hour = st.checkbox("Peak Hour", value=(hour in [7, 8, 9, 17, 18, 19, 20]), key=f"{key_prefix}peak")
        has_festival = st.checkbox("Festival / Holiday", value=False, key=f"{key_prefix}festival")

    tr1, rd1 = st.columns(2)
    with tr1:
        section_label("Traffic")
        traffic_density = st.selectbox("Traffic Density", ["low", "medium", "high"], key=f"{key_prefix}traffic_density")
        traffic_signal = st.checkbox("Traffic Signal Present", value=True, key=f"{key_prefix}signal")
    with rd1:
        section_label("Road")
        road_type = st.selectbox("Road Type", ["urban", "highway", "rural"], key=f"{key_prefix}road_type")
        lanes = st.slider("Number of Lanes", 1, 5, 3, key=f"{key_prefix}lanes")

    section_label("Environment")
    e1, e2, e3 = st.columns(3)
    with e1:
        weather = st.selectbox("Weather", ["clear", "rain", "fog"], key=f"{key_prefix}weather")
    with e2:
        visibility = st.selectbox("Visibility", ["high", "medium", "low"], key=f"{key_prefix}visibility")
    with e3:
        temperature = st.slider("Temperature (°C)", 10, 45, 28, key=f"{key_prefix}temperature")

    return {
        "city": city,
        "state": state,
        "latitude": None,
        "longitude": None,
        "road_type": road_type,
        "lanes": lanes,
        "hour": hour,
        "day_of_week": day_of_week,
        "weather": weather,
        "visibility": visibility,
        "traffic_density": traffic_density,
        "temperature": temperature,
        "traffic_signal": int(traffic_signal),
        "is_weekend": int(is_weekend),
        "is_peak_hour": int(is_peak_hour),
        "has_festival": str(int(has_festival)),
    }


def render_shap_section(model, feature_names, shap_df):
    """Renders the full SHAP breakdown from an ALREADY-COMPUTED shap_df
    (computed once at prediction time and cached in session_state) —
    avoids recomputing SHAP every time the user toggles this panel."""
    if shap_df is None:
        st.warning(
            "SHAP is not installed (or the explanation failed), so a detailed feature-level "
            "breakdown can't be shown right now. Install it with `pip install shap` and rerun. "
            "Showing the model's overall (global) feature importance instead:"
        )
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({
                "label": [humanize_feature_name(f) for f in feature_names],
                "importance": importances,
            }).sort_values("importance", ascending=False).head(10)
            fig = px.bar(imp_df, x="importance", y="label", orientation="h")
            fig.update_traces(marker_color=BRAND_PRIMARY)
            fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"}, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        return

    st.caption(
        "SHAP (SHapley Additive exPlanations) explains how each factor influenced THIS specific "
        "Random Forest prediction — it does not prove that any factor causes higher/lower risk in general."
    )

    top = shap_df.head(8).copy()
    top["label"] = top["feature"].apply(humanize_feature_name)
    top["direction"] = np.where(top["shap_value"] >= 0, "Increases risk", "Decreases risk")
    fig = px.bar(
        top.sort_values("shap_value"),
        x="shap_value", y="label", orientation="h", color="direction",
        color_discrete_map={"Increases risk": RISK_COLORS["CRITICAL"], "Decreases risk": RISK_COLORS["LOW"]},
    )
    fig.update_layout(height=380, xaxis_title="Impact on predicted risk score", yaxis_title="",
                       legend_title="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    lg1, lg2 = st.columns(2)
    lg1.markdown(f'<span style="color:{RISK_COLORS["CRITICAL"]};font-weight:700;">■</span> Increases predicted risk (positive SHAP contribution)', unsafe_allow_html=True)
    lg2.markdown(f'<span style="color:{RISK_COLORS["LOW"]};font-weight:700;">■</span> Decreases predicted risk (negative SHAP contribution)', unsafe_allow_html=True)

    section_label("In Plain Language")
    for s in shap_plain_english_summary(shap_df, top_n=3):
        st.markdown(f"- {s}")

    section_label("Contribution by Category")
    st.caption("Weather / Road / Traffic / Infrastructure — normalized across all model features, not just the top 8 above.")
    group_df = contribution_by_group(shap_df)
    if len(group_df):
        gc = st.columns(len(group_df))
        for i, (_, grow) in enumerate(group_df.iterrows()):
            gc[i].metric(grow["group"], f"{grow['pct']}%")


def page_risk_prediction(df, preprocessor, model, feature_names):
    page_header(
        "DECISION SUPPORT",
        "Risk Prediction",
        "Random Forest model — estimate accident risk for a location and time.",
    )

    city_state_map = df.drop_duplicates("city").set_index("city")["state"].to_dict()
    city_coord_map = df.groupby("city")[["latitude", "longitude"]].mean().to_dict("index")

    with st.container(border=True):
        row = build_input_row(city_state_map, key_prefix="predict_")

    coords = city_coord_map.get(row["city"], {"latitude": 20.0, "longitude": 78.0})
    row["latitude"] = coords["latitude"]
    row["longitude"] = coords["longitude"]

    st.markdown("")
    predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([row])[ALL_FEATURES]
        score = float(predict_risk_score(input_df, preprocessor, model)[0])
        level, emoji = risk_level(score)
        st.session_state.last_input_row = row
        st.session_state.last_prediction = {"score": score, "level": level, "emoji": emoji}
        st.session_state.predict_reveal = "actions"
        # Compute SHAP once here (full feature set) and cache it — both
        # the inline "Top Contributing Factors" teaser below and the
        # full "View Risk Factors" panel reuse this same result instead
        # of recomputing SHAP on every button click.
        st.session_state.last_shap_df = shap_explanation(
            input_df, preprocessor, model, feature_names, max_display=len(feature_names)
        )

    if st.session_state.last_input_row is not None and st.session_state.last_prediction is not None:
        row = st.session_state.last_input_row
        pred = st.session_state.last_prediction
        shap_df = st.session_state.get("last_shap_df")

        st.markdown("")
        result_col, meta_col = st.columns([1, 2])
        with result_col:
            with st.container(border=True):
                st.caption("PREDICTED ACCIDENT RISK")
                st.markdown(f"# {pred['score']:.0f} / 100")
                st.markdown(risk_badge_html(pred["level"]), unsafe_allow_html=True)
        with meta_col:
            with st.container(border=True):
                st.caption("Scenario Summary")
                st.markdown(
                    f"**{row['city']}, {row['state']}** · {row['road_type']} road · "
                    f"{row['weather']} weather · hour {row['hour']} ({row['day_of_week']})"
                )
                interpretation = {
                    "LOW": "Conditions indicate a low likelihood of high-severity risk factors combining at this time and location.",
                    "MEDIUM": "Some risk factors are present. Routine monitoring is reasonable, with attention during identified high-risk conditions.",
                    "HIGH": "Multiple risk factors are combining at this time and location. Proactive intervention is recommended.",
                    "CRITICAL": "A significant combination of risk factors is present. This scenario warrants immediate attention.",
                }
                st.caption(interpretation.get(pred["level"], ""))
                if shap_df is not None:
                    top_factors = ", ".join(humanize_feature_name(f) for f in shap_df.head(3)["feature"])
                    st.caption(f"Top contributing factors: **{top_factors}**  \u00b7  full breakdown below.")

        st.markdown("")
        b1, b2, b3 = st.columns(3)
        if b1.button("View Risk Factors", use_container_width=True,
                      type="primary" if st.session_state.predict_reveal == "factors" else "secondary"):
            st.session_state.predict_reveal = "factors"
            st.rerun()
        if b2.button("View Recommended Actions", use_container_width=True,
                      type="primary" if st.session_state.predict_reveal == "actions" else "secondary"):
            st.session_state.predict_reveal = "actions"
            st.rerun()
        if b3.button("Run What-If Simulation", use_container_width=True):
            goto("what_if")

        st.markdown("")
        if st.session_state.predict_reveal == "factors":
            section_label("Why Is the Risk High? (SHAP Explanation)")
            render_shap_section(model, feature_names, shap_df)
        elif st.session_state.predict_reveal == "actions":
            section_label("Recommended Actions")
            st.caption("Transparent rule-based logic (not machine-learned) — grounded in the scenario above.")
            render_priority_cards(recommendations_for(row))
    else:
        st.info("Fill in the conditions above and click **Predict Risk** to get started.")


# ----------------------------------------------------------------------
# PAGE: WHAT-IF SIMULATOR
# — Identical prediction calls to the original expander version; only
#   promoted to its own page and given a clearer before/after layout.
# ----------------------------------------------------------------------
def page_what_if(preprocessor, model):
    page_header(
        "DECISION SUPPORT",
        "What-If Simulator",
        "Model-based scenario simulation — compare current conditions against a modified scenario.",
    )

    if st.session_state.last_input_row is None:
        st.warning("No prediction yet. Go to **Risk Prediction**, run a prediction, then return here.")
        if st.button("Go to Risk Prediction"):
            goto("risk_prediction")
        return

    base_row = st.session_state.last_input_row
    base_score = st.session_state.last_prediction["score"]
    base_level = st.session_state.last_prediction["level"]

    st.caption(
        "Adjust the conditions below and compare against your last prediction. "
        "This estimates how the trained model responds to changed conditions — "
        "it does not guarantee a specific reduction in real-world accidents."
    )

    sim_row = dict(base_row)
    with st.container(border=True):
        section_label("Simulated Conditions")
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_row["traffic_density"] = st.selectbox(
                "Traffic Density", ["low", "medium", "high"],
                index=["low", "medium", "high"].index(base_row["traffic_density"]),
                key="sim_traffic_density")
        with s2:
            sim_row["visibility"] = st.selectbox(
                "Visibility", ["high", "medium", "low"],
                index=["high", "medium", "low"].index(base_row["visibility"]),
                key="sim_visibility")
        with s3:
            sim_row["weather"] = st.selectbox(
                "Weather", ["clear", "rain", "fog"],
                index=["clear", "rain", "fog"].index(base_row["weather"]),
                key="sim_weather")
        s4, s5 = st.columns(2)
        with s4:
            sim_row["traffic_signal"] = int(st.checkbox(
                "Traffic Signal Present", value=bool(base_row["traffic_signal"]), key="sim_signal"))
        with s5:
            sim_row["is_peak_hour"] = int(st.checkbox(
                "Peak Hour", value=bool(base_row["is_peak_hour"]), key="sim_peak"))

        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_sim:
        sim_df = pd.DataFrame([sim_row])[ALL_FEATURES]
        sim_score = float(predict_risk_score(sim_df, preprocessor, model)[0])
        sim_level, _ = risk_level(sim_score)
        st.session_state.sim_result = {"score": sim_score, "level": sim_level, "row": dict(sim_row)}

    st.markdown("")
    result = st.session_state.get("sim_result")

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True):
            st.caption("CURRENT SCENARIO")
            st.markdown(
                f"Traffic: **{base_row['traffic_density'].title()}**  \n"
                f"Visibility: **{base_row['visibility'].title()}**  \n"
                f"Weather: **{base_row['weather'].title()}**"
            )
            st.markdown(f"### {base_score:.0f} / 100")
            st.markdown(risk_badge_html(base_level), unsafe_allow_html=True)
    with cc2:
        with st.container(border=True):
            st.caption("SIMULATED SCENARIO")
            if result is None:
                st.markdown(
                    f"Traffic: **{sim_row['traffic_density'].title()}**  \n"
                    f"Visibility: **{sim_row['visibility'].title()}**  \n"
                    f"Weather: **{sim_row['weather'].title()}**"
                )
                st.caption("Click **Run Simulation** to see the result.")
            else:
                r = result["row"]
                st.markdown(
                    f"Traffic: **{r['traffic_density'].title()}**  \n"
                    f"Visibility: **{r['visibility'].title()}**  \n"
                    f"Weather: **{r['weather'].title()}**"
                )
                st.markdown(f"### {result['score']:.0f} / 100")
                st.markdown(risk_badge_html(result["level"]), unsafe_allow_html=True)

    if result is not None:
        delta = result["score"] - base_score
        direction = "decrease" if delta < 0 else ("increase" if delta > 0 else "no change")

        changed_fields = [
            ("Traffic Density", "traffic_density"), ("Visibility", "visibility"),
            ("Weather", "weather"), ("Traffic Signal Present", "traffic_signal"),
            ("Peak Hour", "is_peak_hour"),
        ]
        changes = []
        for label, field in changed_fields:
            old_v, new_v = base_row[field], result["row"][field]
            if str(old_v) != str(new_v):
                if field in ("traffic_signal", "is_peak_hour"):
                    old_v = "Yes" if str(old_v) in ("1", "True") else "No"
                    new_v = "Yes" if str(new_v) in ("1", "True") else "No"
                else:
                    old_v, new_v = str(old_v).title(), str(new_v).title()
                changes.append(f"{label}: {old_v} → {new_v}")

        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Risk", f"{base_score:.0f}/100")
        m2.metric("Simulated Risk", f"{result['score']:.0f}/100")
        m3.metric("Change", f"{delta:+.1f} pts", delta_color="inverse")

        level_change = (
            f"{base_level} → {result['level']}" if base_level != result["level"] else f"No change ({base_level})"
        )
        st.markdown(f"**Risk Level Change:** {level_change}")

        if changes:
            st.markdown(f"**Changed Factors:** {' · '.join(changes)}")
        else:
            st.caption("No conditions were changed from the current scenario.")

        st.caption(
            f"Model-based scenario simulation (Random Forest): under the simulated conditions, "
            f"the trained model estimates a {abs(delta):.1f}-point {direction} in risk score "
            "(0-100 scale). This is not causal evidence and does not guarantee a specific "
            "real-world reduction in accidents."
        )


# ----------------------------------------------------------------------
# PAGE: ANALYTICS
# — Same charts/queries as the original page_analytics(); regrouped
#   into four labeled sections with consistent chart sizing.
# ----------------------------------------------------------------------
CHART_H = 320


def page_analytics(df, hotspots):
    page_header(
        "SYSTEM-WIDE INSIGHT",
        "Analytics",
        "Historical patterns and estimated-risk trends across the full dataset.",
    )

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    section_label("Time Analysis")
    st.caption("Estimated risk from historical data by time — not a prediction of exact future accident counts.")
    t1, t2 = st.columns(2)
    with t1:
        by_hour = df.groupby("hour")["risk_score_100"].mean().reset_index()
        fig = px.line(by_hour, x="hour", y="risk_score_100", markers=True,
                      labels={"hour": "Hour of Day", "risk_score_100": "Avg Risk Score"})
        fig.update_traces(line_color=BRAND_PRIMARY)
        fig.update_layout(title="Average Risk by Hour", height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        by_hour_count = df.groupby("hour").size().reset_index(name="accidents")
        fig = px.bar(by_hour_count, x="hour", y="accidents",
                     labels={"hour": "Hour of Day", "accidents": "Accident Count"})
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(title="Accidents by Hour", height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    t3, t4 = st.columns(2)
    with t3:
        by_dow = df.groupby("day_of_week")["risk_score_100"].mean().reindex(dow_order).reset_index()
        fig = px.bar(by_dow, x="day_of_week", y="risk_score_100",
                     labels={"day_of_week": "Day of Week", "risk_score_100": "Avg Risk Score"})
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(title="Average Risk by Day of Week", height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        heat = df.pivot_table(index="day_of_week", columns="hour", values="risk_score_100", aggfunc="mean")
        heat = heat.reindex(dow_order)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="OrRd", labels={"color": "Avg Risk"})
        fig.update_layout(title="Risk Heatmap: Day × Hour", height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    section_label("Location Analysis")
    l1, l2 = st.columns(2)
    with l1:
        by_state = df["state"].value_counts().reset_index()
        by_state.columns = ["state", "accidents"]
        fig = px.bar(by_state, x="state", y="accidents", title="Accidents by State")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with l2:
        city_counts = df["city"].value_counts().reset_index()
        city_counts.columns = ["city", "accidents"]
        fig = px.bar(city_counts, x="city", y="accidents", title="Accidents by City")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 10 High-Risk Locations**")
    top10 = hotspots.head(10).reset_index(drop=True)
    top10.index = top10.index + 1
    st.dataframe(
        top10[["hotspot_id", "avg_risk_score", "risk_level", "historical_accident_count", "high_risk_period"]]
        .rename(columns={
            "hotspot_id": "Location", "avg_risk_score": "Risk Score", "risk_level": "Risk Level",
            "historical_accident_count": "Historical Accidents", "high_risk_period": "Peak Risk Period",
        }),
        use_container_width=True,
    )

    section_label("Condition Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        by_weather = df.groupby("weather")["risk_score_100"].mean().reset_index()
        fig = px.bar(by_weather, x="weather", y="risk_score_100", title="Avg Risk by Weather")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_traffic = df.groupby("traffic_density")["risk_score_100"].mean().reset_index()
        fig = px.bar(by_traffic, x="traffic_density", y="risk_score_100", title="Avg Risk by Traffic Density")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        by_road = df.groupby("road_type")["risk_score_100"].mean().reset_index()
        fig = px.bar(by_road, x="road_type", y="risk_score_100", title="Avg Risk by Road Type")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    section_label("Historical Severity")
    st.caption(
        "Describes historical outcomes only — never used as model inputs "
        "(using them would be data leakage, since they aren't known before an accident)."
    )
    h1, h2 = st.columns(2)
    with h1:
        sev = df["accident_severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.pie(sev, names="severity", values="count", title="Accident Severity Distribution")
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with h2:
        causes = df["cause"].value_counts().reset_index()
        causes.columns = ["cause", "count"]
        fig = px.bar(causes, x="cause", y="count", title="Top Accident Causes")
        fig.update_traces(marker_color=BRAND_PRIMARY)
        fig.update_layout(height=CHART_H, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE (was the sidebar "Model Evaluation" expander)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# PAGE: VANET SAFETY NETWORK
# Purpose: demonstrate a SIMULATED Vehicular Ad-hoc Network layer —
# Simulated Vehicles -> V2V -> RSU -> V2I -> RoadGuard AI (existing
# trained model) -> Hotspot Correlation -> Safety Attention Alert.
#
# Everything vehicle/RSU/event related is randomly generated for this
# demo. The REAL data used are: actual hotspot locations/risk levels,
# actual city coordinates, the ACTUAL trained RandomForestRegressor
# (called on-demand, never retrained), and the ACTUAL rule-based
# recommendation engine.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_vanet_scenario(_hotspots, _city_coord_map, seed):
    """Cached on `seed` — regenerating only happens when the user clicks
    'Simulate New Events' (which bumps the seed), not on every rerun."""
    vehicles = vanet_utils.generate_vehicles(_hotspots, n_vehicles=18, seed=seed)
    rsus = vanet_utils.generate_rsus(_city_coord_map, _hotspots)
    rsus = vanet_utils.compute_rsu_status(vehicles, rsus)
    return vehicles, rsus


def page_vanet(df, hotspots, preprocessor, model):
    page_header(
        "CONNECTED VEHICLES",
        "VANET Safety Network",
        "Simulated VANET communication layer demonstrating how vehicle and roadside-unit "
        "safety information could be integrated with RoadGuard AI.",
    )
    st.info(
        "⚠️ **This is a software simulation, not real-time vehicle hardware or live traffic data.** "
        "Vehicles, roadside units (RSUs), and events below are randomly generated for demonstration. "
        "The hotspot locations/risk levels, the Random Forest risk model, and the recommendation "
        "engine used to correlate and respond to these simulated events are all real and unchanged "
        "— this page calls them for live inference, it does not retrain or modify them."
    )

    st.caption(
        "**Pipeline:** Simulated Vehicles → V2V → Roadside Unit (RSU) → V2I → "
        "RoadGuard AI (trained model, on-demand inference) → Hotspot Correlation → Safety Attention Alert"
    )

    city_coord_map = df.groupby("city")[["latitude", "longitude"]].mean().to_dict("index")

    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("Simulate New Events", type="primary", use_container_width=True):
            st.session_state.vanet_seed += 1
            st.rerun()
    with top_l:
        st.caption(f"Scenario #{st.session_state.vanet_seed - 41} — click 'Simulate New Events' to generate a new random scenario.")

    vehicles, rsus = get_vanet_scenario(hotspots, city_coord_map, st.session_state.vanet_seed)
    msgs = vanet_utils.generate_v2v_messages(vehicles)
    alerts = vanet_utils.generate_safety_alerts(rsus, vehicles, hotspots, preprocessor, model)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Vehicles", len(vehicles))
    k2.metric("Active RSUs", int((rsus["alert_status"] == "ACTIVE").sum()), help=f"of {len(rsus)} total RSUs (one per city)")
    k3.metric("Safety Events", int(vehicles["is_dangerous"].sum()), help="Non-'Normal' events reported by simulated vehicles")
    k4.metric("Hotspot Alerts", len(alerts), help="Safety Attention Alerts correlated with a real historical hotspot")

    st.markdown("")
    section_label("VANET Network Map")
    st.caption("Gray = reference hotspots · Blue/Red large = RSUs (red = ACTIVE) · Small green/red = vehicles (red = dangerous event)")
    render_vanet_map(vehicles, rsus, hotspots, height=440)

    st.markdown("")
    section_label("Safety Attention Alerts")
    if not alerts:
        st.info("No hotspot-correlated safety alerts in this scenario. Click **Simulate New Events** to generate a new one.")
    for a in alerts:
        with st.container(border=True):
            st.markdown(f"#### ⚠️ Safety Attention Alert — {a['hotspot_id']}")
            st.caption(f"{a['events_received']} vehicle(s) near {a['rsu_id']} reported **{a['dominant_event']}** "
                       f"({a['distance_km']} km from this hotspot).")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Historical Hotspot Risk")
                st.markdown(risk_badge_html(a["historical_risk_level"]), unsafe_allow_html=True)
                st.caption(f"{a['historical_risk_score']}/100 (historical average)")
            with c2:
                st.caption("Simulated Traffic Now")
                st.markdown(f"##### {str(a['simulated_traffic']).title()}")
            with c3:
                st.caption("RoadGuard AI Live Re-Assessment")
                if a["live_score"] is not None:
                    st.markdown(risk_badge_html(a["live_level"]), unsafe_allow_html=True)
                    st.caption(f"{a['live_score']:.0f}/100 (Random Forest, live simulated conditions)")
                else:
                    st.caption("Unavailable")
            st.markdown("**Safety Recommendation** (rule-based engine):")
            for r in a["recommendations"]:
                st.markdown(f"- {r}")

    st.markdown("")
    col_v2v, col_rsu = st.columns([1, 1])
    with col_v2v:
        section_label("V2V Messages")
        st.caption("Vehicle-to-vehicle safety broadcasts (most recent first).")
        if not msgs:
            st.info("No V2V messages in this scenario.")
        for m in msgs:
            with st.container(border=True):
                st.markdown(f"**{m['from']} → {m['to']}**")
                st.caption(f"\"{m['message']}\" · {m['distance_km']} km apart")
    with col_rsu:
        section_label("RSU Status")
        st.caption("One simulated Roadside Unit per city (V2I aggregation point).")
        rsu_table = rsus[["rsu_id", "connected_vehicles", "events_received", "alert_status", "nearest_hotspot"]].copy()
        rsu_table.columns = ["RSU", "Connected Vehicles", "Events Received", "Status", "Nearest Hotspot"]
        st.dataframe(rsu_table, use_container_width=True, hide_index=True)

    st.markdown("")
    section_label("VANET Event Log")
    log = vanet_utils.build_event_log(vehicles, rsus)
    st.dataframe(log, use_container_width=True, hide_index=True)


def page_model_performance(metrics):
    page_header(
        "DECISION SUPPORT",
        "Model Performance",
        "Random Forest regression model — how it was trained and evaluated.",
    )

    if not metrics:
        st.warning("Metrics unavailable.")
        return

    st.info(
        "Model performance reflects prediction of the dataset's **risk_score_100** target and "
        "should not be interpreted as accident prediction accuracy. This is a regression model "
        "estimating a continuous 0–100 risk score — not a calibrated accident-probability classifier."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Model R² (held-out)",
        metrics.get("R2", "N/A"),
        help="R² measures how well the model predicts the dataset's risk score on held-out test data. "
             "It is not accident prediction accuracy.",
    )
    m2.metric("MAE (points)", metrics.get("MAE", "N/A"),
              help="Mean Absolute Error — average deviation between predicted and actual risk score, on the 0-100 scale.")
    m3.metric("RMSE (points)", metrics.get("RMSE", "N/A"),
              help="Root Mean Squared Error — penalizes larger errors more than MAE, same 0-100 scale.")

    n_train = metrics.get("n_train")
    n_test = metrics.get("n_test")
    st.markdown("")
    with st.container(border=True):
        section_label("Training Details")
        lines = [
            f"- **Model type:** {metrics.get('model_type', 'RandomForestRegressor (scikit-learn)')}",
            f"- **Task:** {metrics.get('task_type', 'N/A')}",
            f"- **Target:** {metrics.get('target', 'N/A')}",
            f"- **Training samples:** {n_train:,}" if n_train else "- **Training samples:** N/A",
            f"- **Testing samples:** {n_test:,}" if n_test else "- **Testing samples:** N/A",
        ]
        if n_train and n_test:
            total = n_train + n_test
            lines.append(
                f"- **Split:** {n_train/total*100:.0f}% training / {n_test/total*100:.0f}% testing "
                f"({total:,} total records)"
            )
        st.markdown("\n".join(lines))

    with st.container(border=True):
        section_label("What the Model Does and Doesn't Predict")
        st.markdown(
            "- **Predicts:** a 0–100 risk score derived from conditions known *before* a "
            "potential accident (time, weather, visibility, traffic density, road type, location).\n"
            "- **Does not predict:** accident counts, casualties, or severity outcomes — those "
            "post-accident fields are deliberately excluded from training to avoid data leakage "
            "(see README.md for the full leakage analysis).\n"
            "- **R² of "
            f"{metrics.get('R2', 'N/A')}** means the model explains that share of variance in the "
            "dataset's risk score on held-out data — it is a measure of curve-fit quality, not a "
            "real-world accident-prediction accuracy percentage."
        )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not st.session_state.logged_in:
        login_screen()
        return

    try:
        df = get_data()
        preprocessor, model, feature_names, metrics = get_model_artifacts()
        hotspots = get_hotspots(preprocessor, model)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. Make sure `data/accidents.csv` exists.")
        return
    except Exception as e:
        st.error(f"Something went wrong loading data/model: {e}")
        return

    page = sidebar_nav()

    if page == "command_center":
        page_command_center(df, hotspots, metrics)
    elif page == "hotspot_intel":
        page_hotspot_intelligence(df, hotspots)
    elif page == "risk_prediction":
        page_risk_prediction(df, preprocessor, model, feature_names)
    elif page == "analytics":
        page_analytics(df, hotspots)
    elif page == "what_if":
        page_what_if(preprocessor, model)
    elif page == "model_performance":
        page_model_performance(metrics)
    elif page == "vanet":
        page_vanet(df, hotspots, preprocessor, model)


if __name__ == "__main__":
    main()
