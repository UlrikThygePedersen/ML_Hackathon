import streamlit as st
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


@st.cache_data
def load_from_kaggle() -> pd.DataFrame:
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "davidcariboo/player-scores",
        "players.csv",
    )


st.title("EDA / Analysis")

col_load, col_upload = st.columns([1, 2])

with col_load:
    if st.button("Load from Kaggle", type="primary"):
        with st.spinner("Loading players.csv from Kaggle…"):
            st.session_state["df"] = load_from_kaggle()

with col_upload:
    uploaded = st.file_uploader("Or upload a CSV file", type="csv", label_visibility="collapsed")
    if uploaded:
        st.session_state["df"] = pd.read_csv(uploaded)

df = st.session_state.get("df")

if df is None:
    st.info("Click **Load from Kaggle** or upload a CSV file to get started.")
    st.stop()

# --- EDA ---
st.subheader("Data preview")
st.dataframe(df.head(100), use_container_width=True)

st.subheader("Shape")
st.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

st.subheader("Descriptive statistics")
st.dataframe(df.describe(), use_container_width=True)

st.subheader("Missing values")
missing = df.isnull().sum().rename("missing").to_frame()
missing["pct"] = (missing["missing"] / len(df) * 100).round(2)
st.dataframe(missing[missing["missing"] > 0], use_container_width=True)

st.subheader("Distribution")
numeric_cols = df.select_dtypes("number").columns.tolist()
if numeric_cols:
    col = st.selectbox("Column", numeric_cols)
    st.bar_chart(df[col].value_counts().sort_index(), height=300)
else:
    st.warning("No numeric columns found.")
