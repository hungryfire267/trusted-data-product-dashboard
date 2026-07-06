# Trusted Data Product Dashboard

An interactive Streamlit dashboard that demonstrates how data product health, quality actions, and analytics readiness can be brought together into a clear decision workflow.

## What It Shows

This demo answers a practical business question:

> Which data product should be trusted, scaled, monitored, or fixed first?

The dashboard uses simulated enterprise analytics data to show:

- data product adoption and confidence trends
- curated dataset contract health
- missing-data and validation checks
- quality issue prioritisation
- product evidence health
- analytics use-case readiness
- a trained Random Forest classifier that predicts whether each product is `Ready`, `Watch`, or `Fix First`

## Why It Matters

The dashboard is designed to move beyond basic reporting. It makes the next action explicit:

- what can be trusted now
- what needs remediation
- where data quality risk is concentrated
- which analytics opportunities are mature enough to progress

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Note

This project uses simulated data only. It does not contain confidential, production, customer, or employer data.
