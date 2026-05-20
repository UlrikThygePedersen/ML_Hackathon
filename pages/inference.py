import numpy as np
import pandas as pd
import streamlit as st


def preprocess_input(
    input_dict: dict,
    fitted_preprocessors: dict,
    feature_names_in: list,
) -> pd.DataFrame:
    result = {}

    for col, (strategy_type, fitted_obj) in fitted_preprocessors.items():
        if col not in input_dict:
            continue
        val = input_dict[col]

        if strategy_type == "drop":
            pass

        elif strategy_type == "onehot":
            for dummy_col in fitted_obj:
                expected = dummy_col[len(col) + 1:]
                result[dummy_col] = 1.0 if str(val) == expected else 0.0

        elif strategy_type == "label":
            try:
                result[col] = float(fitted_obj.transform([str(val)])[0])
            except ValueError:
                result[col] = 0.0

        elif strategy_type == "target":
            result[col] = float(fitted_obj.get(val, fitted_obj.mean()))

        elif strategy_type == "log":
            result[col] = float(np.log1p(max(0.0, float(val))))

        elif strategy_type == "standard":
            result[col] = float(fitted_obj.transform([[float(val)]])[0][0])

        elif strategy_type == "minmax":
            result[col] = float(fitted_obj.transform([[float(val)]])[0][0])

        elif strategy_type == "keep":
            result[col] = val

    return pd.DataFrame([result]).reindex(columns=feature_names_in, fill_value=0)


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("ML Inference")

run = st.session_state.get("latest_run")
if run is None:
    st.warning("No trained model found. Train a model on the ML Training page first.")
    st.stop()

st.info(
    f"**{run['model']}** · trained {run['timestamp']} · "
    f"R² {run['metrics']['r2']:.3f} · "
    f"MAE €{run['metrics']['mae_eur']:,.0f}"
)

# Source data for populating selectbox options
source_df = st.session_state.get("df_clean") if "df_clean" in st.session_state else st.session_state.get("df")

st.subheader("Input features")
st.caption("Fill in the player attributes to get a predicted market value.")

input_values = {}
cols = st.columns(3)

for i, col in enumerate(run["features"]):
    strategy_type, _ = run["fitted_preprocessors"].get(col, ("keep", None))

    if strategy_type == "drop":
        continue

    is_cat = strategy_type in ("onehot", "label", "target")

    with cols[i % 3]:
        if is_cat and source_df is not None and col in source_df.columns:
            options = sorted(source_df[col].dropna().unique().tolist())
            input_values[col] = st.selectbox(col, options, key=f"inf_{col}")
        elif is_cat:
            input_values[col] = st.text_input(col, value="", key=f"inf_{col}")
        else:
            col_data = source_df[col] if source_df is not None and col in (source_df.columns if source_df is not None else []) else None
            default = float(col_data.median()) if col_data is not None else 0.0
            input_values[col] = st.number_input(col, value=default, format="%g", key=f"inf_{col}")

st.divider()
if st.button("Predict market value", type="primary"):
    X = preprocess_input(
        input_values,
        run["fitted_preprocessors"],
        run["feature_names_in"],
    )

    raw_pred = run["clf"].predict(X)[0]

    if run.get("target_transform") == "Log transform":
        predicted_value = np.expm1(raw_pred)
    else:
        predicted_value = raw_pred

    st.success(f"Predicted market value: **€{predicted_value:,.0f}**")
