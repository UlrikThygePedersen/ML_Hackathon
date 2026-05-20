import streamlit as st
import pandas as pd

st.title("Model Registry / Training Stats")

runs = st.session_state.get("runs", [])

if not runs:
    st.info("No training runs yet. Train a model on the ML Training page.")
    st.stop()

records = [
    {
        "timestamp": r["timestamp"],
        "model": r["model"],
        "target": r["target"],
        "features": len(r["features"]),
        "test_size": r["test_size"],
        "accuracy": r["accuracy"],
    }
    for r in runs
]
registry_df = pd.DataFrame(records)

st.subheader("All runs")
st.dataframe(registry_df, width="stretch")

st.subheader("Accuracy over runs")
st.line_chart(registry_df["accuracy"], height=250)

best = registry_df.loc[registry_df["accuracy"].idxmax()]
st.subheader("Best run")
st.json(best.to_dict())

selected_idx = st.selectbox(
    "Inspect run",
    range(len(runs)),
    format_func=lambda i: f"{runs[i]['timestamp']} — {runs[i]['model']} ({runs[i]['accuracy']:.2%})",
)
run = runs[selected_idx]
st.write("**Features used:**", run["features"])
if st.button("Set as active model for inference"):
    st.session_state["latest_run"] = run
    st.success("Active model updated. Go to ML Inference to use it.")
