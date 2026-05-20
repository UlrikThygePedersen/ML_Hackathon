import streamlit as st

st.title("Data Cleaning / Imputation")

if st.session_state.get("df") is None:
    st.warning("No dataset loaded. Load data on the EDA page first.")
    st.stop()

st.info("Cleaning logic coming soon.")
