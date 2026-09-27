from __future__ import annotations

import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import FraudDetector

st.set_page_config(
    page_title="SafePay | Transaction Check",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Overall page */
.stApp { background: #f7f8fa; }
.block-container { max-width: 920px; padding-top: 2.2rem; padding-bottom: 3.5rem; }

/* Typography */
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body, .stMarkdown, .stText, label, p { font-size: 1.04rem !important; }

/* Header */
.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem; }
.brand { display:flex; align-items:center; gap:.75rem; }
.brand-icon { width:48px; height:48px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:#ffffff; border:1px solid #e7e9ee; font-size:1.55rem; box-shadow:0 2px 10px rgba(25,35,50,.05); }
.brand-name { font-size:1.35rem; font-weight:750; color:#17202b; }
.brand-sub { color:#6d7682; font-size:.92rem; margin-top:.1rem; }

/* Intro card */
.intro { background:#ffffff; border:1px solid #e5e8ee; border-radius:20px; padding:1.55rem 1.6rem; margin-bottom:1rem; box-shadow:0 5px 24px rgba(28,39,55,.05); }
.intro h1 { margin:0 0 .45rem 0; font-size:2rem; line-height:1.18; color:#141b24; }
.intro p { margin:0; color:#66717f; font-size:1.07rem !important; line-height:1.55; }

/* Cards */
.card { background:#fff; border:1px solid #e5e8ee; border-radius:18px; padding:1.25rem 1.3rem; margin:1rem 0; box-shadow:0 4px 18px rgba(28,39,55,.04); }
.card-title { font-size:1.22rem; font-weight:750; color:#17202b; margin-bottom:.2rem; }
.card-help { color:#6f7885; font-size:.95rem; margin-bottom:.8rem; }

/* Inputs */
div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label, div[data-testid="stDateInput"] label, div[data-testid="stTimeInput"] label, div[data-testid="stSelectbox"] label { font-size:1rem !important; font-weight:650 !important; color:#303946 !important; }
input, textarea, [data-baseweb="select"] > div { font-size:1.04rem !important; }

/* Primary button */
.stButton > button, div[data-testid="stFormSubmitButton"] > button { min-height:3.2rem; font-size:1.05rem !important; font-weight:750 !important; border-radius:12px !important; }

/* Result */
.result-card { background:#fff; border:1px solid #e5e8ee; border-radius:20px; padding:1.45rem 1.5rem; margin-top:1.1rem; box-shadow:0 5px 24px rgba(28,39,55,.05); }
.result-title { font-size:1.15rem; font-weight:750; color:#596473; text-transform:uppercase; letter-spacing:.05em; }
.risk-title { font-size:2rem; font-weight:800; margin:.35rem 0 .45rem; color:#151c25; }
.risk-copy { color:#606b78; font-size:1.05rem; line-height:1.55; }
.risk-low { border-left:7px solid #2e9a64; }
.risk-medium { border-left:7px solid #e3a51a; }
.risk-high { border-left:7px solid #d95555; }

/* Friendly score row */
.score-label { color:#66717f; font-size:.92rem; margin-bottom:.25rem; }
.score-value { font-size:1.55rem; font-weight:800; color:#17202b; }

/* Hide technical Streamlit chrome */
footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

@media (max-width: 700px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .intro h1 { font-size:1.65rem; }
  .risk-title { font-size:1.65rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🔒</div>
    <div>
      <div class="brand-name">SafePay</div>
      <div class="brand-sub">Transaction security check</div>
    </div>
  </div>
</div>

<div class="intro">
  <h1>Check a transaction before it goes through</h1>
  <p>Enter the basic payment details below. Our fraud-detection model checks the transaction for unusual activity and gives you a simple risk result.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not Path("models/autoencoder.keras").exists() or not Path("models/lstm.keras").exists():
    st.warning("The trained models are not available yet.")
    st.code(
        "python src/train_autoencoder.py --data data/demo_transactions.csv\n"
        "python src/train_lstm.py --data data/demo_transactions.csv"
    )
    st.stop()


@st.cache_resource
def load_detector():
    return FraudDetector()


try:
    detector = load_detector()
except Exception as exc:
    st.error(f"We couldn't load the fraud-detection models: {exc}")
    st.stop()

st.markdown(
    """
<div class="card">
  <div class="card-title">Payment details</div>
  <div class="card-help">Use the same information you would normally see on a payment confirmation.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("transaction_form"):
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount (₹)", min_value=0.0, value=1500.0, step=100.0)
        transaction_date = st.date_input("Payment date", value=date.today())
        transaction_type = st.selectbox(
            "Payment method",
            ["UPI", "Card", "Online", "ATM", "Bank Transfer"],
        )
    with c2:
        previous_amount = st.number_input(
            "Previous payment amount (₹)", min_value=0.0, value=1200.0, step=100.0
        )
        transaction_time = st.time_input("Payment time (24-hour)", value=time(14, 30), step=300)
        transactions_today = st.number_input(
            "Payments made today", min_value=1, max_value=200, value=4, step=1
        )

    st.markdown(
        """
<div class="card" style="margin-top:1rem; box-shadow:none; background:#fafbfc;">
  <div class="card-title">A couple of quick security checks</div>
  <div class="card-help">These help us spot changes in normal payment behaviour.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    s1, s2 = st.columns(2)
    with s1:
        new_device = st.toggle("This is a new device", value=False)
    with s2:
        new_location = st.toggle("This is a new location", value=False)

    st.write("")
    submitted = st.form_submit_button("Check transaction", use_container_width=True)

if submitted:
    tx = pd.DataFrame(
        [
            {
                "Amount": amount,
                "PreviousAmount": previous_amount,
                "TransactionsToday": transactions_today,
                "NewDevice": int(new_device),
                "NewLocation": int(new_location),
                "TransactionType": transaction_type,
                "TransactionDate": transaction_date.strftime("%Y-%m-%d"),
                "TransactionTime": transaction_time.strftime("%H:%M"),
                "Class": 0,
            }
        ]
    )

    history = st.session_state.get("transaction_history", [])
    history.append(tx.iloc[0].to_dict())
    history = history[-detector.sequence_length :]
    st.session_state["transaction_history"] = history

    history_df = pd.DataFrame(history)
    result = detector.predict(history_df)

    combined = result["combined_risk_score"]
    ae_score = result["autoencoder_anomaly_score"]
    lstm_score = result["lstm_fraud_probability"]

    if result["prediction"] == "FRAUD":
        risk_class = "risk-high"
        title = "This payment looks unusual"
        icon = "🚨"
        message = "We found several signals that are associated with fraudulent activity. In a real payment app, this transaction would be worth verifying before proceeding."
    elif result["prediction"] == "REVIEW":
        risk_class = "risk-medium"
        title = "This payment needs a quick check"
        icon = "🟠"
        message = "Some activity looks different from the normal pattern. It may be genuine, but an extra verification step would be sensible."
    else:
        risk_class = "risk-low"
        title = "This payment looks normal"
        icon = "✅"
        message = "The transaction is broadly consistent with the patterns learned by the fraud-detection model."

    st.markdown(
        f"""
<div class="result-card {risk_class}">
  <div class="result-title">Transaction result</div>
  <div class="risk-title">{icon} {title}</div>
  <div class="risk-copy">{message}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'<div class="score-label">Overall risk</div><div class="score-value">{combined:.0%}</div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="score-label">Unusual activity</div><div class="score-value">{ae_score:.0%}</div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="score-label">Recent pattern</div><div class="score-value">{lstm_score:.0%}</div>', unsafe_allow_html=True)

    with st.expander("How did we decide?"):
        st.write(
            "The system checks two things: how unusual the transaction looks on its own, "
            "and whether it fits the recent pattern of transactions. Those signals are combined "
            "into the overall risk score."
        )
        st.caption(
            "Model details: Autoencoder reconstruction/anomaly score + LSTM sequence risk."
        )

    with st.expander("Transaction summary"):
        s1, s2 = st.columns(2)
        s1.write(f"**Amount:** ₹{amount:,.2f}")
        s1.write(f"**Payment method:** {transaction_type}")
        s1.write(f"**Date:** {transaction_date.strftime('%d/%m/%Y')}")
        s2.write(f"**Time:** {transaction_time.strftime('%H:%M')}")
        s2.write(f"**New device:** {'Yes' if new_device else 'No'}")
        s2.write(f"**New location:** {'Yes' if new_location else 'No'}")

st.divider()
st.caption("Academic prototype · Not a real banking authorization or financial-security service.")
