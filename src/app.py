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
    page_title="SafePay | Financial Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
    --bg: #0b1220;
    --panel: #111a2b;
    --panel-2: #162238;
    --border: #263754;
    --text: #f4f7fb;
    --muted: #aebbd0;
    --accent: #5b8cff;
    --accent-hover: #759dff;
    --good: #37c987;
    --warn: #f2b94b;
    --bad: #f06b6b;
}

.stApp {
    background: linear-gradient(180deg, #09111e 0%, #0d1524 100%);
    color: var(--text);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    width: 100%;
    max-width: 1500px;
    padding: 2rem 3.5rem 4rem 3.5rem;
}

html, body, [class*="css"], .stMarkdown, .stText, label, p, span, div {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body, .stMarkdown, .stText, p, label {
    font-size: 1.02rem;
}

h1, h2, h3 {
    color: var(--text) !important;
}

/* top brand */
.brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.brand-left { display:flex; align-items:center; gap:.9rem; }
.brand-icon {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: #15233a;
    border: 1px solid var(--border);
    font-size: 1.7rem;
}
.brand-name { font-size: 1.65rem; font-weight: 800; color: var(--text); }
.brand-sub { font-size: 1rem; color: var(--muted); margin-top:.15rem; }

.hero {
    background: linear-gradient(135deg, #13213a 0%, #101a2d 100%);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 2rem 2.1rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 18px 50px rgba(0,0,0,.18);
}
.hero h1 { margin:0 0 .45rem 0; font-size: 2.25rem; line-height:1.15; }
.hero p { margin:0; color: var(--muted); font-size:1.13rem !important; line-height:1.55; }

.section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.45rem 1.55rem 1.6rem 1.55rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 12px 36px rgba(0,0,0,.14);
}
.section-title { font-size:1.35rem; font-weight:800; color:var(--text); }
.section-help { color:var(--muted); font-size:1rem; margin:.25rem 0 1rem 0; }

/* Streamlit input styling */
div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stToggle"] label {
    color: #d8e1ee !important;
    font-size: 1.03rem !important;
    font-weight: 700 !important;
}

input, textarea, [data-baseweb="select"] > div {
    font-size: 1.05rem !important;
    background: #0d1728 !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stNumberInput"] input {
    min-height: 2.9rem;
}

/* toggle */
[data-testid="stToggle"] { padding-top:.4rem; }

/* Submit button */
div[data-testid="stFormSubmitButton"] > button {
    width:100%;
    min-height: 3.5rem;
    border-radius: 14px !important;
    border: 1px solid #6f99ff !important;
    background: var(--accent) !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 22px rgba(91,140,255,.25);
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--accent-hover) !important;
    color: white !important;
}

/* result panel */
.review-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    overflow:hidden;
    margin-top: .5rem;
    box-shadow: 0 18px 45px rgba(0,0,0,.18);
}
.review-head {
    padding: 1.55rem 1.65rem 1.35rem 1.65rem;
    border-bottom:1px solid var(--border);
}
.review-kicker {
    color: #9fb2cc;
    font-size: .92rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.review-title { font-size: 2rem; font-weight: 850; margin:.35rem 0 .45rem; color:var(--text); }
.review-copy { color: #c2cede; font-size:1.08rem; line-height:1.6; margin:0; }
.review-grid { display:grid; grid-template-columns: 1.1fr 1fr; gap:0; }
.review-section { padding:1.4rem 1.65rem; }
.review-section + .review-section { border-left:1px solid var(--border); }
.review-section h3 { font-size:1.15rem; margin:0 0 .9rem; }
.review-item { display:flex; justify-content:space-between; gap:1rem; padding:.62rem 0; border-bottom:1px solid rgba(38,55,84,.65); }
.review-item:last-child { border-bottom:0; }
.review-label { color:#9fb0c8; }
.review-value { color:#f1f5fb; font-weight:700; text-align:right; }
.explain {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 14px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: #dbe4f0;
    line-height:1.6;
}

.risk-low { border-top:5px solid var(--good); }
.risk-medium { border-top:5px solid var(--warn); }
.risk-high { border-top:5px solid var(--bad); }

.score-card {
    background: #0f192b;
    border:1px solid var(--border);
    border-radius:16px;
    padding:1rem 1.05rem;
}
.score-label { color:#9fb0c8; font-size:.92rem; }
.score-value { color:#f5f8fc; font-size:1.7rem; font-weight:850; margin-top:.15rem; }

[data-testid="stMetric"] {
    background:#0f192b;
    border:1px solid var(--border);
    padding:1rem;
    border-radius:16px;
}

[data-testid="stExpander"] {
    background:#0f192b;
    border:1px solid var(--border);
    border-radius:14px;
}

.stAlert { border-radius:14px; }

footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

@media (max-width: 900px) {
    .block-container { padding-left:1rem; padding-right:1rem; }
    .review-grid { grid-template-columns: 1fr; }
    .review-section + .review-section { border-left:0; border-top:1px solid var(--border); }
    .hero h1 { font-size:1.8rem; }
    .review-title { font-size:1.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="brand-row">
  <div class="brand-left">
    <div class="brand-icon">🔒</div>
    <div>
      <div class="brand-name">SafePay</div>
      <div class="brand-sub">Financial transaction security check</div>
    </div>
  </div>
</div>

<div class="hero">
  <h1>Check a payment before it goes through</h1>
  <p>Enter a few details about the payment. The system checks for unusual transaction behaviour and gives you a clear risk assessment.</p>
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
<div class="section-card">
  <div class="section-title">Payment details</div>
  <div class="section-help">Use the basic information you would normally see in a payment confirmation.</div>
""",
    unsafe_allow_html=True,
)

with st.form("transaction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount (₹)", min_value=0.0, value=1500.0, step=100.0)
        transaction_type = st.selectbox("Payment method", ["UPI", "Card", "Online", "ATM", "Bank Transfer"])
    with c2:
        previous_amount = st.number_input("Previous payment amount (₹)", min_value=0.0, value=1200.0, step=100.0)
        transaction_date = st.date_input("Payment date", value=date.today())
    with c3:
        transactions_today = st.number_input("Payments made today", min_value=1, max_value=200, value=4, step=1)
        transaction_time = st.time_input("Payment time (24-hour)", value=time(14, 30), step=300)

    st.markdown(
        "<div class='section-help' style='margin-top:1rem;'>Security checks help the model spot changes in normal payment behaviour.</div>",
        unsafe_allow_html=True,
    )
    s1, s2 = st.columns(2)
    with s1:
        new_device = st.toggle("This is a new device", value=False)
    with s2:
        new_location = st.toggle("This is a new location", value=False)

    st.write("")
    submitted = st.form_submit_button("🔍  Check transaction")

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    tx = pd.DataFrame([
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
    ])

    history = st.session_state.get("transaction_history", [])
    history.append(tx.iloc[0].to_dict())
    history = history[-detector.sequence_length:]
    st.session_state["transaction_history"] = history

    result = detector.predict(pd.DataFrame(history))
    combined = float(result["combined_risk_score"])
    ae_score = float(result["autoencoder_anomaly_score"])
    lstm_score = float(result["lstm_fraud_probability"])

    if result["prediction"] == "FRAUD":
        risk_class = "risk-high"
        title = "This payment looks unusual"
        icon = "🚨"
        message = "The model found several signals that are associated with unusual or potentially fraudulent activity. In a real payment service, this would be a good point to verify the payment before continuing."
        next_step = "Recommended: verify the payment and device details before completing it."
    elif result["prediction"] == "REVIEW":
        risk_class = "risk-medium"
        title = "This payment needs a quick check"
        icon = "🟠"
        message = "Some behaviour is different from the patterns learned by the model. This does not mean the payment is fraudulent, but an extra verification step would be sensible."
        next_step = "Recommended: confirm the amount, device and payment location."
    else:
        risk_class = "risk-low"
        title = "This payment looks normal"
        icon = "✅"
        message = "The transaction is broadly consistent with the behaviour learned by the fraud-detection models. No strong anomaly was detected by the prototype."
        next_step = "Recommended: no additional action based on this prototype's assessment."

    st.markdown(
        f"""
<div class="review-panel {risk_class}">
  <div class="review-head">
    <div class="review-kicker">Transaction review</div>
    <div class="review-title">{icon} {title}</div>
    <p class="review-copy">{message}</p>
    <div class="explain"><strong>What this means:</strong> {next_step}</div>
  </div>
  <div class="review-grid">
    <div class="review-section">
      <h3>Payment details</h3>
      <div class="review-item"><span class="review-label">Amount</span><span class="review-value">₹{amount:,.2f}</span></div>
      <div class="review-item"><span class="review-label">Payment method</span><span class="review-value">{transaction_type}</span></div>
      <div class="review-item"><span class="review-label">Date</span><span class="review-value">{transaction_date.strftime('%d/%m/%Y')}</span></div>
      <div class="review-item"><span class="review-label">Time</span><span class="review-value">{transaction_time.strftime('%H:%M')}</span></div>
      <div class="review-item"><span class="review-label">Payments today</span><span class="review-value">{transactions_today}</span></div>
    </div>
    <div class="review-section">
      <h3>Security signals</h3>
      <div class="review-item"><span class="review-label">New device</span><span class="review-value">{'Yes' if new_device else 'No'}</span></div>
      <div class="review-item"><span class="review-label">New location</span><span class="review-value">{'Yes' if new_location else 'No'}</span></div>
      <div class="review-item"><span class="review-label">Overall risk</span><span class="review-value">{combined:.0%}</span></div>
      <div class="review-item"><span class="review-label">Unusual activity</span><span class="review-value">{ae_score:.0%}</span></div>
      <div class="review-item"><span class="review-label">Recent pattern risk</span><span class="review-value">{lstm_score:.0%}</span></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Why did the system flag this?")
    st.write(
        "The fraud detector uses two views of the payment. The Autoencoder checks how unusual the transaction looks compared with learned normal behaviour. The LSTM looks at the recent transaction sequence for unusual patterns. Their signals are combined into the overall risk score."
    )

    q1, q2 = st.columns(2)
    with q1:
        st.metric("Overall risk", f"{combined:.0%}")
    with q2:
        st.metric("Recent pattern risk", f"{lstm_score:.0%}")

    with st.expander("View technical model details"):
        st.write(f"Autoencoder anomaly score: {ae_score:.2%}")
        st.write(f"LSTM fraud probability: {lstm_score:.2%}")
        st.write(f"Combined risk score: {combined:.2%}")
        st.caption("Prototype model: Autoencoder reconstruction/anomaly signal + LSTM sequence-risk signal.")

    with st.expander("View the exact transaction record"):
        st.dataframe(tx.drop(columns=["Class"]), use_container_width=True, hide_index=True)

st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
st.caption("Academic prototype · This result is an AI risk assessment, not a real banking authorization or guarantee of fraud.")
