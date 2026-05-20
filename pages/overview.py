import streamlit as st

st.title("ML Hackathon — Overview")
st.markdown(
    """
    Welcome to the ML Hackathon dashboard. Use the sidebar to navigate between sections.

    | Page | Purpose |
    |------|---------|
    | **EDA / Analysis** | Explore and visualise the dataset |
    | **ML Training** | Configure and run model training |
    | **ML Inference** | Run predictions on new data |
    | **Model Registry** | Browse trained models and training stats |
    """
)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Dataset rows", "—")
col2.metric("Models trained", "—")
col3.metric("Best accuracy", "—")

st.info("Load data on the EDA page and train a model to populate the metrics above.")
