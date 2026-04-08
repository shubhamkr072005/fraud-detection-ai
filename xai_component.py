from __future__ import annotations

import plotly.graph_objects as go


def generate_natural_language_explanation(transaction: dict) -> str:
    amount = float(transaction.get("Amount", 0))
    avg_amount = float(transaction.get("User_Avg_Amount", 1))
    location = str(transaction.get("Location", "Unknown"))
    home = str(transaction.get("User_Home", "Unknown"))
    ip_score = int(transaction.get("IP_Score", 0))
    tx_time = str(transaction.get("Time", ""))

    hour = 12
    try:
        hour = int(tx_time.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        hour = 12

    reasons = []
    if avg_amount > 0 and amount > avg_amount * 2:
        increase = int(((amount - avg_amount) / avg_amount) * 100)
        reasons.append(f"amount is {increase}% higher than this user's usual spend")
    if location != home:
        reasons.append("transaction location does not match the user's home profile")
    if 1 <= hour <= 5:
        reasons.append("transaction occurred during unusual hours (1 AM - 5 AM)")
    if ip_score >= 80:
        reasons.append("originated from a high-risk IP")
    elif ip_score >= 60:
        reasons.append("originated from a medium-risk IP")

    if not reasons:
        return "This transaction appears normal with low anomaly signals across amount, location, and timing."

    joined_reasons = "; ".join(reasons)
    return f"Flagged because {joined_reasons}."


def build_feature_importance_chart(transaction: dict):
    amount = float(transaction.get("Amount", 0))
    avg_amount = float(transaction.get("User_Avg_Amount", 1))
    location = str(transaction.get("Location", "Unknown"))
    home = str(transaction.get("User_Home", "Unknown"))
    ip_score = int(transaction.get("IP_Score", 0))
    tx_time = str(transaction.get("Time", ""))

    hour = 12
    try:
        hour = int(tx_time.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        hour = 12

    amount_importance = min(45, max(0, ((amount / max(avg_amount, 1)) - 1) * 12))
    location_importance = 30 if location != home else 5
    time_importance = 15 if 1 <= hour <= 5 else 4
    ip_importance = min(20, ip_score / 5)

    features = ["Amount Spike", "Location Mismatch", "Unusual Time", "IP Risk"]
    values = [amount_importance, location_importance, time_importance, ip_importance]

    fig = go.Figure(
        go.Bar(
            x=features,
            y=values,
            marker_color=["#ff4b4b", "#f39c12", "#9b59b6", "#00c2ff"],
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="🧠 Feature Importance (Explainable AI)",
        template="plotly_dark",
        yaxis_title="Contribution to Risk",
        xaxis_title="Signals",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
