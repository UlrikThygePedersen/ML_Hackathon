import streamlit as st
import pandas as pd

st.title("EDA / Analysis")

uploaded = st.file_uploader("Upload a CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = None

if df is not None:
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
else:
    st.info("Upload a CSV file above to get started.")
