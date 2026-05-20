import base64
import os
import streamlit as st


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.markdown(
    """
    <div style="
        background: #1a3a5c;
        border-radius: 14px;
        padding: 36px 40px;
        margin-bottom: 8px;
    ">
        <h1 style="color:#ffffff; margin:0 0 6px 0; font-size:2.2rem;">ML Hackathon — Overview</h1>
        <p style="color:#a8c4e0; margin:0; font-size:1.05rem; max-width:720px;">
            An end-to-end machine learning pipeline for <strong style="color:#7ec8e3;">player performance prediction</strong>.
            Explore, clean, and model a dataset of player statistics, then run live predictions —
            all from a single interactive dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ## Workflow

    The project follows a standard end-to-end ML pipeline, split across dedicated pages:

    | Page | Purpose |
    |------|---------|
    | **EDA / Analysis** | Explore distributions, correlations, and key statistics in the raw dataset |
    | **Data Cleaning** | Handle missing values, outliers, and feature engineering |
    | **ML Training** | Configure hyperparameters and train classification or regression models |
    | **ML Inference** | Run predictions on new or held-out data using a trained model |
    | **Model Registry** | Browse all trained model runs, compare metrics, and select a champion model |

    """
)


st.divider()

# Contributors
st.subheader("Contributors")

CONTRIBUTORS = [
    {
        "handle": "ULPN",
        "name": "Ulrike Thyge Pedersen",
        "img": "assets/ULPN.png",
        "github": "https://github.com/UlrikThygePedersen",
    },
    {
        "handle": "JBXT",
        "name": "Jan Bendix Portius",
        "img": "assets/JBXT.png",
        "github": "https://github.com/Bartimaeus25",
    },
]

# CSS: circular image crop + card styling
st.markdown(
    """
    <style>
    .contributor-card {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 18px 22px;
        border: 1px solid #4a4a4a;
        border-radius: 12px;
        background: #2a2a2a;
    }
    .contributor-card img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        flex-shrink: 0;
        border: 2px solid #555;
    }
    .contributor-placeholder {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: #555;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        flex-shrink: 0;
    }
    .contributor-info .handle {
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff;
        margin: 0 0 3px 0;
    }
    .contributor-info .fullname {
        color: #cccccc;
        font-size: 0.88rem;
        margin: 0 0 10px 0;
    }
    .contributor-info a {
        font-size: 0.83rem;
        color: #79b8ff;
        text-decoration: none;
        font-weight: 500;
    }
    .contributor-info a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

CARD_TEMPLATE = """
<div class="contributor-card">
    {img_tag}
    <div class="contributor-info">
        <p class="handle">{handle}</p>
        <p class="fullname">{name}</p>
        <a href="{github}" target="_blank">GitHub &#x2197;</a>
    </div>
</div>
"""

cols = st.columns(len(CONTRIBUTORS), gap="medium")
for col, c in zip(cols, CONTRIBUTORS):
    img_path = c["img"]
    if os.path.exists(img_path):
        b64 = _img_to_b64(img_path)
        img_tag = "<img src='data:image/png;base64," + b64 + "' />"
    else:
        img_tag = "<div class='contributor-placeholder'>👤</div>"
    html = CARD_TEMPLATE.format(
        img_tag=img_tag,
        handle=c["handle"],
        name=c["name"],
        github=c["github"],
    )
    with col:
        st.markdown(html, unsafe_allow_html=True)
