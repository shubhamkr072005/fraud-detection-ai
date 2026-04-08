from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd


USER_PROFILES = {
    "Ava": {"home_location": "New York", "usual_device": "iPhone", "avg_amount": 340},
    "Liam": {"home_location": "London", "usual_device": "MacBook", "avg_amount": 520},
    "Noah": {"home_location": "Dubai", "usual_device": "Android", "avg_amount": 610},
    "Emma": {"home_location": "San Francisco", "usual_device": "Windows PC", "avg_amount": 430},
    "Sophia": {"home_location": "Singapore", "usual_device": "iPad", "avg_amount": 710},
    "Mason": {"home_location": "Berlin", "usual_device": "Linux Laptop", "avg_amount": 480},
}

ALL_LOCATIONS = [
    "New York",
    "London",
    "Dubai",
    "San Francisco",
    "Singapore",
    "Berlin",
    "Moscow",
    "Lagos",
    "Bogota",
    "Istanbul",
]

DEVICES = ["iPhone", "Android", "MacBook", "Windows PC", "iPad", "Linux Laptop"]


def calculate_risk_score(
    amount: float, location: str, user_home: str, transaction_hour: int, ip_score: int
) -> int:
    """
    Rule-based ML-style risk scoring:
    a) Amount > 5000
    b) Location differs from user home
    c) Unusual time (1 AM to 5 AM)
    """
    score = 0

    if amount > 5000:
        score += 45
    if location != user_home:
        score += 30
    if 1 <= transaction_hour <= 5:
        score += 15

    # Add IP-based signal for realism.
    if ip_score >= 80:
        score += 20
    elif ip_score >= 60:
        score += 10

    return min(score, 100)


def _sample_amount(user_avg: float) -> float:
    # Mostly normal spend, sometimes extreme anomalies.
    if random.random() < 0.18:
        return round(random.uniform(5500, 15000), 2)
    return round(random.uniform(user_avg * 0.4, user_avg * 3.2), 2)


def generate_synthetic_transactions(n_rows: int = 50) -> pd.DataFrame:
    """Generate a realistic synthetic transaction dataset."""
    now = datetime.now()
    rows = []
    user_names = list(USER_PROFILES.keys())

    for _ in range(n_rows):
        user = random.choice(user_names)
        profile = USER_PROFILES[user]

        time_offset_minutes = random.randint(0, 24 * 60 * 7)
        tx_time = now - timedelta(minutes=time_offset_minutes)
        amount = _sample_amount(profile["avg_amount"])

        # 75% at home location, 25% potentially suspicious location.
        if random.random() < 0.75:
            location = profile["home_location"]
        else:
            location = random.choice([loc for loc in ALL_LOCATIONS if loc != profile["home_location"]])

        # Device is usually consistent for user.
        if random.random() < 0.7:
            device = profile["usual_device"]
        else:
            device = random.choice(DEVICES)

        ip_score = random.randint(5, 98)
        risk = calculate_risk_score(
            amount=amount,
            location=location,
            user_home=profile["home_location"],
            transaction_hour=tx_time.hour,
            ip_score=ip_score,
        )

        rows.append(
            {
                "Time": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "User": user,
                "Amount": amount,
                "Location": location,
                "Device": device,
                "IP_Score": ip_score,
                "Risk_Score": risk,
                "User_Home": profile["home_location"],
                "User_Avg_Amount": profile["avg_amount"],
            }
        )

    df = pd.DataFrame(rows).sort_values("Time", ascending=False).reset_index(drop=True)
    return df
