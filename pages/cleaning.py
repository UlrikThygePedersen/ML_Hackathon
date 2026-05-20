import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

USEFUL_COLS = [
    "age", "position", "sub_position", "foot", "height_in_cm",
    "country_of_citizenship", "international_caps", "international_goals",
    "current_club_domestic_competition_id", "market_value_in_eur",
    "highest_market_value_in_eur",
]

DEFAULT_STRATEGIES = {
    "market_value_in_eur":                  "Drop rows",
    "highest_market_value_in_eur":          "Drop rows",
    "age":                                  "Median",
    "height_in_cm":                         "Median by position",
    "foot":                                 "Mode",
    "sub_position":                         "Mode by position",
    "international_caps":                   "Fill with 0",
    "international_goals":                  "Fill with 0",
    "country_of_citizenship":               "Mode",
    "current_club_domestic_competition_id": 'Fill with "Unknown"',
}

STRATEGIES = [
    "Drop rows",
    "Mean",
    "Median",
    "Median by position",
    "Mode",
    "Mode by position",
    "Fill with 0",
    'Fill with "Unknown"',
]


def apply_strategy(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    if strategy == "Drop rows":
        return df.dropna(subset=[col])
    if strategy == "Mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "Median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "Median by position":
        df[col] = df[col].fillna(
            df.groupby("position")[col].transform("median")
        ).fillna(df[col].median())
    elif strategy == "Mode":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif strategy == "Mode by position":
        df[col] = df[col].fillna(
            df.groupby("position")[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
        ).fillna(df[col].mode()[0])
    elif strategy == "Fill with 0":
        df[col] = df[col].fillna(0)
    elif strategy == 'Fill with "Unknown"':
        df[col] = df[col].fillna("Unknown")
    return df


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("Data Cleaning / Imputation")

raw = st.session_state.get("df")
if raw is None:
    st.warning("No dataset loaded. Load data on the EDA page first.")
    st.stop()

# prepare age if not already done
if "age" not in raw.columns:
    raw = raw.copy()
    raw["date_of_birth"] = pd.to_datetime(raw["date_of_birth"], errors="coerce")
    raw["age"] = ((pd.Timestamp.now() - raw["date_of_birth"]).dt.days / 365.25).round(1)

# ── Section 1: Column selector ───────────────────────────────────────────────
st.header("1 · Select features")
st.caption("Untick columns that should not be passed to the model.")

all_cols = [c for c in raw.columns if c not in ("date_of_birth",)]
default_ticked = [c for c in USEFUL_COLS if c in all_cols]

col_grid = st.columns(3)
selected_cols = []
for i, col in enumerate(all_cols):
    with col_grid[i % 3]:
        if st.checkbox(col, value=(col in default_ticked), key=f"col_{col}"):
            selected_cols.append(col)

if not selected_cols:
    st.warning("Select at least one column.")
    st.stop()

# ── Section 2: Imputation strategies ────────────────────────────────────────
st.divider()
st.header("2 · Imputation strategies")
st.caption("Only columns with missing values are shown.")

cols_with_nulls = [c for c in selected_cols if raw[c].isnull().any()]
strategies_chosen = {}

if not cols_with_nulls:
    st.success("No missing values in the selected columns.")
else:
    header = st.columns([2, 1, 2])
    header[0].markdown("**Column**")
    header[1].markdown("**Missing**")
    header[2].markdown("**Strategy**")

    for col in cols_with_nulls:
        n_missing = raw[col].isnull().sum()
        pct = n_missing / len(raw) * 100
        row = st.columns([2, 1, 2])
        row[0].write(col)
        row[1].write(f"{n_missing:,} ({pct:.1f}%)")
        default = DEFAULT_STRATEGIES.get(col, "Median" if raw[col].dtype != "object" else "Mode")
        strategies_chosen[col] = row[2].selectbox(
            col, STRATEGIES,
            index=STRATEGIES.index(default),
            label_visibility="collapsed",
            key=f"strat_{col}",
        )

# ── Section 3: Apply ─────────────────────────────────────────────────────────
st.divider()
st.header("3 · Apply & preview")

if st.button("Apply cleaning", type="primary"):
    df_clean = raw[selected_cols].copy()

    for col, strategy in strategies_chosen.items():
        if col in df_clean.columns:
            df_clean = apply_strategy(df_clean, col, strategy)

    st.session_state["df_clean"] = df_clean

before_nulls = raw[selected_cols].isnull().sum().sum()
df_clean = st.session_state.get("df_clean")

if df_clean is not None:
    after_nulls = df_clean.isnull().sum().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows before", f"{len(raw):,}")
    col2.metric("Rows after", f"{len(df_clean):,}", delta=f"{len(df_clean) - len(raw):,}")
    col3.metric("Missing values remaining", after_nulls)

    st.subheader("Missing values — before vs after")
    before = raw[selected_cols].isnull().mean().mul(100).rename("before")
    after = df_clean.reindex(columns=selected_cols).isnull().mean().mul(100).rename("after")
    compare = pd.concat([before, after], axis=1).reset_index().rename(columns={"index": "column"})
    compare = compare[compare["before"] > 0].sort_values("before", ascending=True)

    fig = px.bar(
        compare.melt(id_vars="column", var_name="stage", value_name="pct_missing"),
        x="pct_missing", y="column", color="stage", barmode="group",
        orientation="h",
        labels={"pct_missing": "% missing", "column": ""},
        color_discrete_map={"before": "#EF553B", "after": "#00CC96"},
        height=max(250, len(compare) * 35),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cleaned data preview")
    st.dataframe(df_clean.head(50), use_container_width=True)
