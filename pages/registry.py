import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("Model Registry / Training Stats")

runs = st.session_state.get("runs", [])
if not runs:
    st.info("No training runs yet. Train a model on the ML Training page.")
    st.stop()

active_run = st.session_state.get("latest_run")

# ── 1. All runs overview ──────────────────────────────────────────────────────

st.header("1 · All runs")

records = [
    {
        "run": i + 1,
        "timestamp": r["timestamp"],
        "model": r["model"],
        "n_features": len(r["features"]),
        "r2": r["metrics"]["r2"],
        "rmse_eur": r["metrics"]["rmse_eur"],
        "mae_eur": r["metrics"]["mae_eur"],
        "n_estimators": r["hyperparams"]["n_estimators"],
        "max_depth": r["hyperparams"]["max_depth"],
        "learning_rate": r["hyperparams"]["learning_rate"],
        "active": "✅" if r is active_run else "",
    }
    for i, r in enumerate(runs)
]
overview_df = pd.DataFrame(records)
st.dataframe(overview_df, width="stretch")

st.subheader("R² over runs")
fig = px.line(
    overview_df, x="run", y="r2", markers=True,
    labels={"run": "Run", "r2": "R²"},
    height=250,
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# ── 2. Run inspector ──────────────────────────────────────────────────────────

st.divider()
st.header("2 · Run inspector")

selected_idx = st.selectbox(
    "Select run",
    range(len(runs)),
    format_func=lambda i: (
        f"Run {i+1} · {runs[i]['timestamp']} · "
        f"R² {runs[i]['metrics']['r2']:.3f} · "
        f"MAE €{runs[i]['metrics']['mae_eur']:,.0f}"
        + (" ✅ active" if runs[i] is active_run else "")
    ),
)
run = runs[selected_idx]

if run is active_run:
    st.success("This is the currently active model used for inference.")
else:
    if st.button("Set as active model for inference", type="primary"):
        st.session_state["latest_run"] = run
        st.rerun()

# ── Metrics ───────────────────────────────────────────────────────────────────

st.subheader("Validation metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{run['metrics']['r2']:.3f}")
c2.metric("RMSE (log)", f"{run['metrics']['rmse_log']:.3f}")
c3.metric("RMSE (€)", f"€{run['metrics']['rmse_eur']:,.0f}")
c4.metric("MAE (€)", f"€{run['metrics']['mae_eur']:,.0f}")

# ── Hyperparameters ───────────────────────────────────────────────────────────

st.subheader("Hyperparameters")
hp = run["hyperparams"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("n_estimators", hp["n_estimators"])
c2.metric("max_depth", hp["max_depth"])
c3.metric("learning_rate", hp["learning_rate"])
c4.metric("subsample", hp["subsample"])
c5.metric("colsample_bytree", hp["colsample_bytree"])
c6.metric("min_child_weight", hp["min_child_weight"])

# ── Validation charts ─────────────────────────────────────────────────────────

st.subheader("Validation set analysis")

if "val_actual" in run and "val_predicted" in run:
    actual = np.array(run["val_actual"])
    predicted = np.array(run["val_predicted"])
    residuals = actual - predicted

    pred_df = pd.DataFrame({"actual": actual, "predicted": predicted, "residual": residuals})

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predicted vs actual**")
        fig = px.scatter(
            pred_df, x="actual", y="predicted",
            log_x=True, log_y=True, opacity=0.5,
            labels={"actual": "Actual (€)", "predicted": "Predicted (€)"},
            height=380,
        )
        min_val = pred_df[["actual", "predicted"]].min().min()
        max_val = pred_df[["actual", "predicted"]].max().max()
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(dash="dash", color="grey", width=1),
            showlegend=False,
        ))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("**Residuals distribution**")
        fig = px.histogram(
            pred_df, x="residual", nbins=60,
            labels={"residual": "Actual − Predicted (€)"},
            height=380,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
else:
    st.info("Retrain the model to see validation charts.")

# ── Feature importance ────────────────────────────────────────────────────────

st.subheader("Feature importance (top 20)")
importance_df = (
    pd.DataFrame({
        "feature": run["feature_names_in"],
        "importance": run["clf"].feature_importances_,
    })
    .sort_values("importance", ascending=True)
    .tail(20)
)
fig = px.bar(
    importance_df, x="importance", y="feature", orientation="h",
    labels={"importance": "Importance", "feature": ""},
    height=max(300, len(importance_df) * 22),
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# ── Features & preprocessing ──────────────────────────────────────────────────

st.subheader("Features & preprocessing")
prep_records = [
    {"feature": col, "preprocessing": strategy}
    for col, strategy in run["feature_strategies"].items()
]
st.dataframe(pd.DataFrame(prep_records), width="stretch")
