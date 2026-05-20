import streamlit as st

st.set_page_config(layout="wide")

pg = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon="🏠"),
    st.Page("pages/eda.py", title="EDA / Analysis", icon="📊"),
    st.Page("pages/training.py", title="ML Training", icon="🤖"),
    st.Page("pages/inference.py", title="ML Inference", icon="🔮"),
    st.Page("pages/registry.py", title="Model Registry", icon="📋"),
])
pg.run()
