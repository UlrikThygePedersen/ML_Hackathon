import streamlit as st
import pandas as pd

st.title("ML Inference")

run = st.session_state.get("latest_run")

if run is None:
    st.warning("No trained model found. Train a model on the ML Training page first.")
    st.stop()

st.info(
    f"Using **{run['model']}** trained at {run['timestamp']} "
    f"(accuracy {run['accuracy']:.2%})"
)

st.subheader("Input features")
input_values = {}
for feat in run["feature_names"]:
    input_values[feat] = st.number_input(feat, value=0.0, format="%g")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([input_values])
    prediction = run["clf"].predict(input_df)[0]
    st.success(f"Prediction: **{prediction}**")

    if hasattr(run["clf"], "predict_proba"):
        proba = run["clf"].predict_proba(input_df)[0]
        classes = run["clf"].classes_
        proba_df = pd.DataFrame({"class": classes, "probability": proba}).set_index(
            "class"
        )
        st.bar_chart(proba_df, height=250)

st.divider()
st.subheader("Batch inference")
uploaded = st.file_uploader("Upload a CSV for batch predictions", type="csv")
if uploaded:
    batch_df = pd.read_csv(uploaded)
    missing = [f for f in run["feature_names"] if f not in batch_df.columns]
    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
    else:
        preds = run["clf"].predict(batch_df[run["feature_names"]])
        result_df = batch_df.copy()
        result_df["prediction"] = preds
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download predictions",
            result_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
