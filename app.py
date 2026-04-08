from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from fraud_logic import generate_synthetic_transactions
from xai_component import build_feature_importance_chart, generate_natural_language_explanation


st.set_page_config(
    page_title="Explainable AI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #161b2f, #0b0d14 50%);
        color: #f5f7ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121a33, #0c1226 80%);
        border-right: 1px solid #293250;
    }
    .metric-card {
        border: 1px solid #2a3354;
        background: rgba(20, 25, 44, 0.85);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
    }
    .metric-title {
        font-size: 0.95rem;
        color: #95a0c4;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #eaf0ff;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: #273158;
        color: #afc7ff;
        font-size: 0.8rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return generate_synthetic_transactions(50)


if "selected_tx_idx" not in st.session_state:
    st.session_state.selected_tx_idx = None

if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0


def render_metrics(data: pd.DataFrame):
    total_scanned = len(data)
    high_risk = int((data["Risk_Score"] >= 70).sum())
    total_savings = float(data.loc[data["Risk_Score"] >= 70, "Amount"].sum() * 0.18)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📦 Total Scanned</div>
                <div class="metric-value">{total_scanned}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🚨 High Risk Detected</div>
                <div class="metric-value">{high_risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💰 Total Savings</div>
                <div class="metric-value">${total_savings:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard(data: pd.DataFrame):
    st.markdown("## 🛡️ Explainable AI Fraud Detection Dashboard")
    st.markdown('<span class="badge">Live Fraud Intelligence</span>', unsafe_allow_html=True)
    render_metrics(data)
    st.write("")

    plot_df = data.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Time"]).dt.date
    trend_df = plot_df.groupby("Date", as_index=False)["Risk_Score"].mean()
    trend_fig = px.line(
        trend_df,
        x="Date",
        y="Risk_Score",
        markers=True,
        title="📈 Fraud Trends",
        template="plotly_dark",
    )
    trend_fig.update_traces(line_color="#00c2ff", line_width=3, marker_size=8)
    trend_fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))

    bins = pd.cut(
        data["Risk_Score"],
        bins=[-1, 39, 69, 100],
        labels=["Low", "Medium", "High"],
    )
    pie_df = bins.value_counts().rename_axis("Risk_Level").reset_index(name="Count")
    pie_fig = px.pie(
        pie_df,
        values="Count",
        names="Risk_Level",
        title="🧭 Risk Distribution",
        hole=0.45,
        color="Risk_Level",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#ff4b4b"},
        template="plotly_dark",
    )
    pie_fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))

    left, right = st.columns(2)
    left.plotly_chart(trend_fig, use_container_width=True)
    right.plotly_chart(pie_fig, use_container_width=True)


def render_alerts(data: pd.DataFrame):
    st.markdown("## 🚨 Live Alerts")
    st.caption("Select a transaction and click Investigate for explainable AI reasoning.")

    working_df = data.copy()
    working_df["Amount"] = working_df["Amount"].map(lambda x: f"${x:,.2f}")
    display_cols = ["Time", "User", "Amount", "Location", "Device", "IP_Score", "Risk_Score"]
    st.dataframe(working_df[display_cols], use_container_width=True, hide_index=True)

    idx_options = list(range(len(data)))
    labels = [
        f"#{i+1} | {data.loc[i, 'User']} | Risk {data.loc[i, 'Risk_Score']} | ${data.loc[i, 'Amount']:,.0f}"
        for i in idx_options
    ]
    idx_map = {labels[i]: idx_options[i] for i in idx_options}

    selected_label = st.selectbox("🔎 Choose a transaction to inspect", labels)
    chosen_idx = idx_map[selected_label]

    if st.button("🧪 Investigate", type="primary"):
        st.session_state.selected_tx_idx = chosen_idx

    if st.session_state.selected_tx_idx is not None:
        tx = data.loc[st.session_state.selected_tx_idx].to_dict()
        explanation = generate_natural_language_explanation(tx)
        st.success(f"**Natural Language Explanation:** {explanation}")
        st.plotly_chart(build_feature_importance_chart(tx), use_container_width=True)


def render_simulator():
    st.markdown("## 🧠 Fraud Simulator")
    st.caption("Interactive sandbox to test risk scoring behavior.")

    amount = st.slider("💵 Transaction Amount", 10, 15000, 1200, step=10)
    location_mismatch = st.toggle("🌍 Location mismatch vs user home", value=False)
    odd_hour = st.toggle("🌙 Unusual time (1 AM - 5 AM)", value=False)
    ip_score = st.slider("🛰️ IP Risk Score", 0, 100, 45)

    score = 0
    if amount > 5000:
        score += 45
    if location_mismatch:
        score += 30
    if odd_hour:
        score += 15
    if ip_score >= 80:
        score += 20
    elif ip_score >= 60:
        score += 10
    score = min(score, 100)

    level = "Low"
    icon = "🟢"
    if score >= 70:
        level, icon = "High", "🔴"
    elif score >= 40:
        level, icon = "Medium", "🟠"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Simulation Outcome</div>
            <div class="metric-value">{icon} {level} Risk ({score}/100)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## 🌌 AI Fraud Console")
    page = st.radio("Navigate", ["Dashboard", "Live Alerts", "Simulator"], label_visibility="collapsed")
    if st.button("🔄 Refresh Data"):
        st.session_state.refresh_count += 1
        get_data.clear()
    st.caption("Built for hackathon demos ⚡")


df_transactions = get_data()

if page == "Dashboard":
    render_dashboard(df_transactions)
elif page == "Live Alerts":
    render_alerts(df_transactions)
else:
    render_simulator()
