import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import datetime

st.title("ML Training")

df = st.session_state.get("df")

if df is None:
    st.warning("No dataset loaded. Upload a CSV on the EDA page first.")
    st.stop()

st.subheader("Feature selection")
all_cols = df.columns.tolist()
target = st.selectbox("Target column", all_cols, index=len(all_cols) - 1)
feature_cols = st.multiselect(
    "Feature columns",
    [c for c in all_cols if c != target],
    default=[c for c in all_cols if c != target],
)

st.subheader("Model configuration")
model_name = st.selectbox(
    "Algorithm",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"],
)
test_size = st.slider("Test split", 0.1, 0.5, 0.2, step=0.05)

if st.button("Train", type="primary"):
    if not feature_cols:
        st.error("Select at least one feature column.")
        st.stop()

    X = df[feature_cols].select_dtypes("number")
    y = df[target]

    if X.empty:
        st.error("No numeric feature columns selected.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    clf = models[model_name]

    with st.spinner("Training…"):
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    run = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "features": feature_cols,
        "target": target,
        "test_size": test_size,
        "accuracy": round(acc, 4),
        "clf": clf,
        "feature_names": X.columns.tolist(),
    }
    st.session_state.setdefault("runs", []).append(run)
    st.session_state["latest_run"] = run

    st.success(f"Accuracy: **{acc:.2%}**")
    st.text(classification_report(y_test, y_pred))
