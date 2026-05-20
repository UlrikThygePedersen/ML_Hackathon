import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter


@st.cache_data
def load_from_kaggle() -> pd.DataFrame:
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "davidcariboo/player-scores",
        "players.csv",
    )


@st.cache_data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
    out["age"] = ((pd.Timestamp.now() - out["date_of_birth"]).dt.days / 365.25).round(1)
    out["value_retention"] = (
        out["market_value_in_eur"] / out["highest_market_value_in_eur"]
    )
    out["log_market_value"] = np.log10(out["market_value_in_eur"].replace(0, np.nan))
    return out


st.title("EDA / Analysis")

# --- Data loading ---
col_load, col_upload = st.columns([1, 2])

with col_load:
    if st.button("Load from Kaggle", type="primary"):
        with st.spinner("Loading players.csv from Kaggle…"):
            st.session_state["df"] = load_from_kaggle()

with col_upload:
    uploaded = st.file_uploader(
        "Or upload a CSV file", type="csv", label_visibility="collapsed"
    )
    if uploaded:
        st.session_state["df"] = pd.read_csv(uploaded)

raw = st.session_state.get("df")

if raw is None:
    st.info("Click **Load from Kaggle** or upload a CSV file to get started.")
    st.stop()

df = prepare(raw)

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")

    positions = sorted(df["position"].dropna().unique().tolist())
    sel_positions = st.multiselect("Position", positions, default=positions)

    top_leagues = (
        df["current_club_domestic_competition_id"]
        .value_counts()
        .head(30)
        .index.tolist()
    )
    sel_leagues = st.multiselect("League", top_leagues, default=top_leagues)

    age_min = int(df["age"].dropna().min())
    age_max = int(df["age"].dropna().max())
    sel_age = st.slider("Age range", age_min, age_max, (age_min, age_max))

    feet = sorted(df["foot"].dropna().unique().tolist())
    sel_foot = st.multiselect("Foot", feet, default=feet)

    top_nations = df["country_of_citizenship"].value_counts().head(20).index.tolist()
    sel_nations = st.multiselect(
        "Nationality (top 20)", top_nations, default=top_nations
    )

mask = (
    df["position"].isin(sel_positions)
    & (
        df["current_club_domestic_competition_id"].isin(sel_leagues)
        | df["current_club_domestic_competition_id"].isna()
    )
    & df["age"].between(sel_age[0], sel_age[1], inclusive="both")
    & (df["foot"].isin(sel_foot) | df["foot"].isna())
    & (
        df["country_of_citizenship"].isin(sel_nations)
        | df["country_of_citizenship"].isna()
    )
)
df = df[mask]

st.caption(f"{len(df):,} players · {df.shape[1]} columns")
st.divider()

# ── 1. Dataset health ────────────────────────────────────────────────────────

st.header("Dataset health")

# 1a. Missing values
st.subheader("Missing values")
useful_cols = [
    "age",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
    "country_of_citizenship",
    "international_caps",
    "international_goals",
    "current_club_domestic_competition_id",
    "market_value_in_eur",
    "highest_market_value_in_eur",
]
missing = (
    df[useful_cols]
    .isnull()
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .rename(columns={"index": "column", 0: "pct_missing"})
    .sort_values("pct_missing", ascending=True)
)
fig = px.bar(
    missing,
    x="pct_missing",
    y="column",
    orientation="h",
    labels={"pct_missing": "% missing", "column": ""},
    color="pct_missing",
    color_continuous_scale="Reds",
    height=350,
)
fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# 1b. Player count by position
st.subheader("Player count by position")
pos_counts = df["position"].value_counts().reset_index()
pos_counts.columns = ["position", "count"]
fig = px.bar(
    pos_counts,
    x="position",
    y="count",
    color="position",
    labels={"count": "Players"},
    height=300,
)
fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

st.divider()

# ── 2. Target variable ───────────────────────────────────────────────────────

st.header("Target variable — Market value")

col1, col2 = st.columns(2)

# 2a. Market value distribution (log scale)
with col1:
    st.subheader("Distribution (log scale)")
    val_df = df["market_value_in_eur"].dropna()
    fig = px.histogram(
        val_df,
        x=val_df,
        nbins=60,
        log_x=True,
        labels={"x": "Market value (€)", "count": "Players"},
        height=350,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# 2b. Current vs highest value
with col2:
    st.subheader("Current vs peak value")
    scatter_df = df[
        ["market_value_in_eur", "highest_market_value_in_eur", "position"]
    ].dropna()
    fig = px.scatter(
        scatter_df,
        x="highest_market_value_in_eur",
        y="market_value_in_eur",
        color="position",
        log_x=True,
        log_y=True,
        labels={
            "highest_market_value_in_eur": "Peak value (€)",
            "market_value_in_eur": "Current value (€)",
        },
        opacity=0.4,
        height=350,
    )
    # diagonal reference line
    max_val = (
        scatter_df[["market_value_in_eur", "highest_market_value_in_eur"]].max().max()
    )
    fig.add_trace(
        go.Scatter(
            x=[1e4, max_val],
            y=[1e4, max_val],
            mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            showlegend=False,
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ── 3. Feature analysis ──────────────────────────────────────────────────────

st.header("Feature analysis")

# 3a. Age vs market value
st.subheader("Age vs market value")
age_df = df[["age", "market_value_in_eur", "position"]].dropna()
age_df = age_df[age_df["age"].between(15, 45)]
fig = px.scatter(
    age_df,
    x="age",
    y="market_value_in_eur",
    color="position",
    log_y=True,
    opacity=0.35,
    trendline="lowess",
    labels={"market_value_in_eur": "Market value (€)", "age": "Age"},
    height=400,
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)

# 3b. Position vs market value (boxplot)
with col1:
    st.subheader("Market value by position")
    box_df = df[["position", "market_value_in_eur"]].dropna()
    fig = px.box(
        box_df,
        x="position",
        y="market_value_in_eur",
        color="position",
        log_y=True,
        labels={"market_value_in_eur": "Market value (€)", "position": ""},
        height=380,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# 3c. Foot preference vs market value
with col2:
    st.subheader("Foot preference vs market value")
    foot_df = (
        df[["foot", "market_value_in_eur"]]
        .dropna()
        .groupby("foot")["market_value_in_eur"]
        .median()
        .reset_index()
        .sort_values("market_value_in_eur", ascending=False)
    )
    fig = px.bar(
        foot_df,
        x="foot",
        y="market_value_in_eur",
        color="foot",
        labels={"market_value_in_eur": "Median market value (€)", "foot": ""},
        height=380,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)

# 3d. League vs average market value (top 15)
with col1:
    st.subheader("Market value by league (top 15)")
    league_df = (
        df[["current_club_domestic_competition_id", "market_value_in_eur"]]
        .dropna()
        .groupby("current_club_domestic_competition_id")["market_value_in_eur"]
        .median()
        .reset_index()
        .sort_values("market_value_in_eur", ascending=False)
        .head(15)
    )
    fig = px.bar(
        league_df,
        x="market_value_in_eur",
        y="current_club_domestic_competition_id",
        orientation="h",
        labels={
            "market_value_in_eur": "Median market value (€)",
            "current_club_domestic_competition_id": "",
        },
        height=420,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# 3e. Nationality vs average market value (top 20)
with col2:
    st.subheader("Market value by nationality (top 20)")
    nat_df = (
        df[["country_of_citizenship", "market_value_in_eur"]]
        .dropna()
        .groupby("country_of_citizenship")["market_value_in_eur"]
        .median()
        .reset_index()
        .sort_values("market_value_in_eur", ascending=False)
        .head(20)
    )
    fig = px.bar(
        nat_df,
        x="market_value_in_eur",
        y="country_of_citizenship",
        orientation="h",
        labels={
            "market_value_in_eur": "Median market value (€)",
            "country_of_citizenship": "",
        },
        height=420,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)

# 3f. Height vs market value
with col1:
    st.subheader("Height vs market value")
    height_df = df[["height_in_cm", "market_value_in_eur", "position"]].dropna()
    fig = px.scatter(
        height_df,
        x="height_in_cm",
        y="market_value_in_eur",
        color="position",
        log_y=True,
        opacity=0.4,
        trendline="lowess",
        labels={
            "height_in_cm": "Height (cm)",
            "market_value_in_eur": "Market value (€)",
        },
        height=380,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# 3g. International caps vs market value
with col2:
    st.subheader("International caps vs market value")
    caps_df = df[["international_caps", "market_value_in_eur", "position"]].dropna()
    caps_df = caps_df[caps_df["international_caps"] >= 10]
    fig = px.scatter(
        caps_df,
        x="international_caps",
        y="market_value_in_eur",
        color="position",
        log_y=True,
        opacity=0.5,
        trendline="lowess",
        labels={
            "international_caps": "International caps",
            "market_value_in_eur": "Market value (€)",
        },
        height=380,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
