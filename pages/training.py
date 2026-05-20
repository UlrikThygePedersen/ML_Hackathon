import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

NUMERIC_OPTIONS = ["Keep as-is", "Log transform", "Standard scale", "Min-max scale", "Drop"]
CATEGORICAL_OPTIONS = ["One-hot encode", "Label encode", "Target encode", "Drop"]
TARGET_OPTIONS = ["Log transform", "Keep as-is"]


def suggest_preprocessing(series: pd.Series) -> str:
    if series.dtype == "object" or hasattr(series, "cat"):
        return "One-hot encode" if series.nunique() < 10 else "Label encode"
    return "Log transform" if abs(series.dropna().skew()) > 1 else "Keep as-is"


def apply_preprocessing(
    df: pd.DataFrame,
    feature_strategies: dict,
    target_col: str,
    target_strategy: str,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    X = df[list(feature_strategies.keys())].copy()
    y = df[target_col].copy()

    if target_strategy == "Log transform":
        y = np.log1p(y)

    fitted = {}
    for col in list(feature_strategies.keys()):
        strategy = feature_strategies[col]
        if col not in X.columns:
            continue

        if strategy == "Drop":
            X = X.drop(columns=[col])
            fitted[col] = ("drop", None)

        elif strategy == "One-hot encode":
            dummies = pd.get_dummies(X[col].astype(str), prefix=col, drop_first=False)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            fitted[col] = ("onehot", dummies.columns.tolist())

        elif strategy == "Label encode":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            fitted[col] = ("label", le)

        elif strategy == "Target encode":
            means = df.groupby(col)[target_col].mean()
            X[col] = X[col].map(means).fillna(df[target_col].mean())
            fitted[col] = ("target", means)

        elif strategy == "Log transform":
            X[col] = np.log1p(X[col].clip(lower=0))
            fitted[col] = ("log", None)

        elif strategy == "Standard scale":
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[[col]]).ravel()
            fitted[col] = ("standard", scaler)

        elif strategy == "Min-max scale":
            scaler = MinMaxScaler()
            X[col] = scaler.fit_transform(X[[col]]).ravel()
            fitted[col] = ("minmax", scaler)

        elif strategy == "Keep as-is":
            fitted[col] = ("keep", None)

    return X, y, fitted


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("ML Training")

df = st.session_state.get("df_clean") if "df_clean" in st.session_state else st.session_state.get("df")

if df is None:
    st.warning("No dataset loaded. Load data on the EDA page first.")
    st.stop()

if "df_clean" in st.session_state:
    st.success("Using cleaned dataset from the Data Cleaning page.")
else:
    st.warning("No cleaned dataset found — using raw data. Consider running Data Cleaning first.")

# ── 1. Data summary ───────────────────────────────────────────────────────────

st.header("1 · Data summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Numeric columns", df.select_dtypes("number").shape[1])

# ── 2. Feature & target setup ─────────────────────────────────────────────────

st.divider()
st.header("2 · Feature & target setup")

all_cols = df.columns.tolist()
default_target = "market_value_in_eur" if "market_value_in_eur" in all_cols else all_cols[-1]
target = st.selectbox("Target column", all_cols, index=all_cols.index(default_target))

feature_cols = st.multiselect(
    "Feature columns",
    [c for c in all_cols if c != target],
    default=[c for c in all_cols if c != target],
)

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

# ── 3. Preprocessing ──────────────────────────────────────────────────────────

st.divider()
st.header("3 · Preprocessing")

st.subheader("Target")
target_strategy = st.selectbox(
    f"`{target}`",
    TARGET_OPTIONS,
    index=0,
    help="Log transform is strongly recommended — market value is log-normally distributed.",
)

st.subheader("Features")
st.caption("Suggestions are based on column type and skewness. You can override any.")

hdr = st.columns([2, 1, 1, 2])
hdr[0].markdown("**Column**")
hdr[1].markdown("**Type**")
hdr[2].markdown("**Unique**")
hdr[3].markdown("**Preprocessing**")

feature_strategies = {}
for col in feature_cols:
    series = df[col]
    is_cat = series.dtype == "object" or hasattr(series, "cat")
    col_type = "categorical" if is_cat else "numeric"
    options = CATEGORICAL_OPTIONS if is_cat else NUMERIC_OPTIONS
    suggestion = suggest_preprocessing(series)

    row = st.columns([2, 1, 1, 2])
    row[0].write(col)
    row[1].write(col_type)
    row[2].write(str(series.nunique()))
    feature_strategies[col] = row[3].selectbox(
        col,
        options,
        index=options.index(suggestion) if suggestion in options else 0,
        label_visibility="collapsed",
        key=f"prep_{col}",
    )

# ── 4. Hyperparameters ────────────────────────────────────────────────────────

st.divider()
st.header("4 · Hyperparameters")

test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05)

c1, c2, c3 = st.columns(3)
n_estimators = c1.slider("n_estimators", 100, 1000, 300, step=50)
max_depth = c2.slider("max_depth", 3, 10, 6)
learning_rate = c3.slider("learning_rate", 0.01, 0.30, 0.10, step=0.01)

with st.expander("Advanced hyperparameters"):
    c1, c2, c3 = st.columns(3)
    subsample = c1.slider("subsample", 0.5, 1.0, 0.8, step=0.05)
    colsample_bytree = c2.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05)
    min_child_weight = c3.slider("min_child_weight", 1, 10, 3)

# ── 5. Train ──────────────────────────────────────────────────────────────────

st.divider()
if st.button("Train model", type="primary"):
    work_df = df[feature_cols + [target]].dropna(subset=[target])
    X, y, fitted = apply_preprocessing(work_df, feature_strategies, target, target_strategy)

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=42,
        verbosity=0,
    )

    with st.spinner("Training XGBoost…"):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if target_strategy == "Log transform":
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_test_orig = y_test
        y_pred_orig = pd.Series(y_pred, index=y_test.index)

    r2 = r2_score(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_eur = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_eur = mean_absolute_error(y_test_orig, y_pred_orig)

    run = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": "XGBoost",
        "target": target,
        "target_transform": target_strategy,
        "features": feature_cols,
        "feature_strategies": feature_strategies,
        "fitted_preprocessors": fitted,
        "feature_names_in": X.columns.tolist(),
        "hyperparams": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
        },
        "metrics": {
            "r2": round(r2, 4),
            "rmse_log": round(rmse_log, 4),
            "rmse_eur": round(rmse_eur, 0),
            "mae_eur": round(mae_eur, 0),
        },
        "clf": model,
        "val_actual": y_test_orig.values.tolist(),
        "val_predicted": y_pred_orig.tolist() if hasattr(y_pred_orig, "tolist") else list(y_pred_orig),
    }
    st.session_state.setdefault("runs", []).append(run)
    st.session_state["latest_run"] = run

    # ── Results ──────────────────────────────────────────────────────────────

    st.header("5 · Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("RMSE (log)", f"{rmse_log:.3f}")
    c3.metric("RMSE (€)", f"€{rmse_eur:,.0f}")
    c4.metric("MAE (€)", f"€{mae_eur:,.0f}")

    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig_imp = px.bar(
        importance_df, x="importance", y="feature", orientation="h",
        labels={"importance": "Importance", "feature": ""},
        height=max(300, len(importance_df) * 22),
    )
    fig_imp.update_layout(margin=dict(l=0, r=0, t=10, b=0))

    pred_df = pd.DataFrame({"actual": y_test_orig.values, "predicted": y_pred_orig})
    fig_pred = px.scatter(
        pred_df, x="actual", y="predicted",
        log_x=True, log_y=True, opacity=0.5,
        labels={"actual": "Actual (€)", "predicted": "Predicted (€)"},
        height=400,
    )
    min_val = pred_df[["actual", "predicted"]].min().min()
    max_val = pred_df[["actual", "predicted"]].max().max()
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(dash="dash", color="grey", width=1),
        showlegend=False,
    ))
    fig_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature importance (top 20)")
        st.plotly_chart(fig_imp, width="stretch")
    with c2:
        st.subheader("Predicted vs actual")
        st.plotly_chart(fig_pred, width="stretch")

    st.success(f"Model saved to registry — R² = {r2:.3f}")
