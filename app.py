from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Trusted Data Product Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "ink": "#17202A",
    "muted": "#5B6673",
    "line": "#D8DEE6",
    "paper": "#FFFFFF",
    "wash": "#F6F8FA",
    "teal": "#047A78",
    "blue": "#2764D8",
    "amber": "#B46A00",
    "red": "#B42318",
    "green": "#2E7D32",
    "violet": "#6852B8",
}


@dataclass(frozen=True)
class FilterState:
    divisions: list[str]
    products: list[str]
    priority: list[str]
    date_range: tuple[pd.Timestamp, pd.Timestamp]


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --ink: {PALETTE["ink"]};
                --muted: {PALETTE["muted"]};
                --line: {PALETTE["line"]};
                --wash: {PALETTE["wash"]};
                --teal: {PALETTE["teal"]};
                --blue: {PALETTE["blue"]};
                --amber: {PALETTE["amber"]};
                --red: {PALETTE["red"]};
                --green: {PALETTE["green"]};
            }}

            .stApp {{
                background: #F4F7F9;
                color: var(--ink);
            }}

            [data-testid="stSidebar"] {{
                background: #FFFFFF;
                border-right: 1px solid var(--line);
            }}

            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1420px;
            }}

            h1, h2, h3 {{
                letter-spacing: 0;
                color: var(--ink);
            }}

            h1 {{
                font-size: 2rem;
                margin-bottom: 0.25rem;
            }}

            .subtitle {{
                color: var(--muted);
                font-size: 1rem;
                margin-bottom: 1.2rem;
            }}

            .metric-card {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem 1rem 0.85rem;
                min-height: 132px;
                box-shadow: 0 1px 2px rgba(23, 32, 42, 0.04);
            }}

            .metric-label {{
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: 0.02em;
            }}

            .metric-value {{
                color: var(--ink);
                font-size: 2rem;
                line-height: 1.15;
                font-weight: 750;
                margin-top: 0.35rem;
            }}

            .metric-note {{
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: 0.55rem;
            }}

            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0.35rem 0 1rem;
            }}

            .pill {{
                display: inline-flex;
                align-items: center;
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                color: var(--ink);
                background: #FFFFFF;
                font-size: 0.83rem;
                font-weight: 620;
            }}

            .section-card {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 1px 2px rgba(23, 32, 42, 0.035);
            }}

            .story-card {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-left: 5px solid var(--blue);
                border-radius: 8px;
                padding: 1rem 1.1rem;
                margin: 0.75rem 0 1rem;
                box-shadow: 0 1px 2px rgba(23, 32, 42, 0.035);
            }}

            .story-title {{
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 760;
                margin-bottom: 0.35rem;
            }}

            .story-body {{
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.45;
            }}

            .brief-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 0.65rem 0 1rem;
            }}

            .brief-cell {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-left: 4px solid var(--teal);
                border-radius: 8px;
                padding: 0.85rem;
                min-height: 104px;
            }}

            .brief-kicker {{
                color: var(--muted);
                font-size: 0.75rem;
                font-weight: 750;
                text-transform: uppercase;
                letter-spacing: 0.03em;
                margin-bottom: 0.25rem;
            }}

            .brief-text {{
                color: var(--ink);
                font-size: 0.95rem;
                line-height: 1.35;
            }}

            .value-strip {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 0.75rem;
                margin: 0.75rem 0 1rem;
            }}

            .value-item {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem;
            }}

            .action-table {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem 1rem;
                margin-bottom: 1rem;
            }}

            .signal-row {{
                display: grid;
                grid-template-columns: 1.05fr 0.7fr 0.7fr 0.75fr 0.85fr 1fr;
                gap: 0.6rem;
                align-items: center;
                border-bottom: 1px solid var(--line);
                padding: 0.5rem 0;
                font-size: 0.92rem;
            }}

            .signal-row:last-child {{
                border-bottom: none;
            }}

            .signal-head {{
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 750;
                text-transform: uppercase;
            }}

            .action-row {{
                display: grid;
                grid-template-columns: 1.1fr 1.4fr 1.2fr 1.2fr;
                gap: 0.6rem;
                align-items: center;
                border-bottom: 1px solid var(--line);
                padding: 0.5rem 0;
                font-size: 0.92rem;
            }}

            .action-row:last-child {{
                border-bottom: none;
            }}

            .status-good {{
                color: var(--green);
                font-weight: 750;
            }}

            .status-watch {{
                color: var(--amber);
                font-weight: 750;
            }}

            .status-risk {{
                color: var(--red);
                font-weight: 750;
            }}

            .small-muted {{
                color: var(--muted);
                font-size: 0.88rem;
            }}

            div[data-testid="stMetric"] {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem;
            }}

            div[data-testid="stMetricValue"] {{
                font-size: 1.65rem;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.35rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px 8px 0 0;
                padding: 0.55rem 0.85rem;
            }}

            .stDataFrame {{
                border: 1px solid var(--line);
                border-radius: 8px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def build_dataset(seed: int = 18) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-10-01", "2026-07-01", freq="W-MON")
    divisions = ["Retail", "Enterprise", "Operations", "Finance", "Risk"]
    products = ["Customer 360", "Claims", "Workforce", "Revenue", "Data Quality"]
    priorities = ["Adoption", "Performance", "Quality", "Predictive", "Self-service"]

    rows: list[dict[str, object]] = []
    for date in dates:
        week_factor = (date - dates.min()).days / max((dates.max() - dates.min()).days, 1)
        for division in divisions:
            base = {
                "Retail": 72,
                "Enterprise": 64,
                "Operations": 58,
                "Finance": 69,
                "Risk": 61,
            }[division]
            for product in products:
                product_lift = {
                    "Customer 360": 7,
                    "Claims": -2,
                    "Workforce": -5,
                    "Revenue": 4,
                    "Data Quality": 1,
                }[product]
                product_profile = {
                    "Customer 360": {"quality": 94, "spec": 88, "missing": 6.5, "validation": 95},
                    "Claims": {"quality": 82, "spec": 71, "missing": 16.5, "validation": 84},
                    "Workforce": {"quality": 88, "spec": 78, "missing": 11.5, "validation": 89},
                    "Revenue": {"quality": 91, "spec": 83, "missing": 8.8, "validation": 92},
                    "Data Quality": {"quality": 96, "spec": 80, "missing": 7.2, "validation": 94},
                }[product]
                division_adjustment = {
                    "Retail": 1.5,
                    "Enterprise": 0.2,
                    "Operations": -1.8,
                    "Finance": 0.8,
                    "Risk": -2.4,
                }[division]
                priority = rng.choice(priorities, p=[0.27, 0.18, 0.22, 0.16, 0.17])
                adoption = np.clip(base + product_lift + week_factor * 14 + rng.normal(0, 5), 20, 98)
                quality = np.clip(
                    product_profile["quality"] + division_adjustment + week_factor * 2.2 + rng.normal(0, 2.8),
                    60,
                    99.4,
                )
                cycle_time = np.clip(9.5 - week_factor * 3.8 + rng.normal(0, 1.1) + (1.5 if division == "Operations" else 0), 1.8, 15)
                active_users = int(np.clip(adoption * rng.normal(18, 3), 250, 2600))
                decisions = int(np.clip(active_users * rng.normal(0.19, 0.035), 30, 560))
                spec_coverage = np.clip(
                    product_profile["spec"] + division_adjustment * 0.7 + week_factor * 4.5 + rng.normal(0, 4.0),
                    45,
                    98,
                )
                missingness = np.clip(
                    product_profile["missing"] - week_factor * 2.4 - division_adjustment * 0.25 + rng.normal(0, 1.7),
                    1,
                    28,
                )
                validation_pass = np.clip(
                    product_profile["validation"] - missingness * 0.25 + division_adjustment * 0.4 + rng.normal(0, 1.8),
                    70,
                    99,
                )
                insight_confidence = np.clip(
                    (quality * 0.45) + (adoption * 0.25) + (spec_coverage * 0.30) + rng.normal(0, 3),
                    35,
                    98,
                )
                rows.append(
                    {
                        "week": date,
                        "division": division,
                        "product": product,
                        "priority": priority,
                        "dashboard_adoption": round(float(adoption), 1),
                        "data_quality": round(float(quality), 1),
                        "spec_coverage": round(float(spec_coverage), 1),
                        "missing_rate": round(float(missingness), 1),
                        "validation_pass_rate": round(float(validation_pass), 1),
                        "insight_confidence": round(float(insight_confidence), 1),
                        "cycle_time_days": round(float(cycle_time), 1),
                        "active_users": active_users,
                        "decisions_supported": decisions,
                        "records_processed_m": round(float(rng.normal(12, 2.5) + week_factor * 6), 1),
                        "defects_open": int(np.clip(rng.normal(18, 8) - week_factor * 6 + (9 if product == "Claims" else 0), 1, 42)),
                    }
                )

    df = pd.DataFrame(rows)

    issue_rows = [
        ("Claims", "Risk", "Duplicate source keys in claims staging", "High", "Data Quality", 74, "SME review"),
        ("Workforce", "Operations", "Late roster feed delaying weekly view", "Medium", "Performance", 62, "Technical fix"),
        ("Revenue", "Finance", "Missing regional code mapping", "High", "Quality", 81, "Requirements"),
        ("Customer 360", "Retail", "Low usage in branch manager cohort", "Medium", "Adoption", 55, "Enablement"),
        ("Data Quality", "Enterprise", "New null-rate anomaly in customer segment", "High", "Predictive", 88, "Monitoring"),
        ("Claims", "Enterprise", "Scenario model awaiting validation set", "Medium", "Predictive", 43, "Testing"),
    ]
    issues = pd.DataFrame(
        issue_rows,
        columns=["product", "division", "issue", "severity", "priority", "impact_score", "next_action"],
    )

    use_cases = pd.DataFrame(
        [
            ("Arrears early-warning model", "Predictive", "Feature discovery", 48, "Customer risk attributes profiled against trend and quality checks"),
            ("Executive adoption drivers", "Trend analysis", "In build", 68, "Power BI-style prototype ready for stakeholder walkthrough"),
            ("Missing-data impact monitor", "Quality monitoring", "Testing", 77, "Imputation benchmark logic applied to incomplete operational feeds"),
            ("Self-service confidence index", "Insights", "Backlog", 22, "Problem statement drafted with Digital Data Product Manager"),
            ("Revenue variance scenarios", "Scenario modelling", "Discovery", 45, "Finance assumptions being translated into technical specs"),
        ],
        columns=["use_case", "method", "stage", "readiness", "latest_update"],
    )

    contracts = pd.DataFrame(
        [
            ("Customer 360", "Customer profile mart", "Daily 07:00", "Branch manager, CX lead", 96, 92, "Ready"),
            ("Claims", "Claims lifecycle fact", "Hourly", "Risk analyst, Claims SME", 81, 76, "Needs key fix"),
            ("Workforce", "Roster utilisation mart", "Weekly Mon 09:00", "Operations planner", 87, 70, "Freshness watch"),
            ("Revenue", "Revenue variance mart", "Daily 08:30", "Finance business partner", 91, 84, "Ready"),
            ("Data Quality", "Quality rules register", "Daily 06:30", "Data steward", 94, 89, "Ready"),
        ],
        columns=[
            "product",
            "curated_dataset",
            "refresh_sla",
            "primary_users",
            "lineage_score",
            "acceptance_coverage",
            "release_status",
        ],
    )

    return df, issues, use_cases, contracts


def fmt_percent(value: float) -> str:
    return f"{value:.1f}%"


def fmt_delta(value: float, suffix: str = " pts") -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}{suffix}"


def pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100


def filter_data(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    start, end = state.date_range
    mask = (
        df["division"].isin(state.divisions)
        & df["product"].isin(state.products)
        & df["priority"].isin(state.priority)
        & df["week"].between(start, end)
    )
    return df.loc[mask].copy()


@st.cache_data
def train_data_product_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, float]:
    feature_cols = [
        "dashboard_adoption",
        "data_quality",
        "spec_coverage",
        "missing_rate",
        "validation_pass_rate",
        "insight_confidence",
        "cycle_time_days",
        "defects_open",
    ]
    model_df = df.copy()
    risk_score = (
        (100 - model_df["insight_confidence"]) * 0.25
        + (100 - model_df["data_quality"]) * 0.20
        + (100 - model_df["validation_pass_rate"]) * 0.20
        + model_df["missing_rate"] * 0.20
        + model_df["defects_open"] * 0.15
    )
    model_df["risk_class"] = np.select(
        [risk_score >= 22, risk_score >= 15],
        ["Fix First", "Watch"],
        default="Ready",
    )

    x = model_df[feature_cols]
    y = model_df["risk_class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance["Feature"] = feature_importance["feature"].map(
        {
            "dashboard_adoption": "Adoption",
            "data_quality": "Quality",
            "spec_coverage": "Spec Coverage",
            "missing_rate": "Missing Rate",
            "validation_pass_rate": "Validation Pass",
            "insight_confidence": "Evidence Confidence",
            "cycle_time_days": "Cycle Time",
            "defects_open": "Open Defects",
        }
    )
    feature_importance = feature_importance[["Feature", "Importance"]]
    return model, feature_importance, model_df, float(accuracy)


def render_metric_card(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_story_context(filtered: pd.DataFrame) -> dict[str, object]:
    by_product = (
        filtered.groupby("product", as_index=False)
        .agg(
            adoption=("dashboard_adoption", "mean"),
            quality=("data_quality", "mean"),
            confidence=("insight_confidence", "mean"),
            missing_rate=("missing_rate", "mean"),
            validation_pass=("validation_pass_rate", "mean"),
            defects=("defects_open", "sum"),
        )
    )
    weakest = by_product.sort_values(["confidence", "validation_pass"]).iloc[0]
    strongest = by_product.sort_values(["confidence", "adoption"], ascending=False).iloc[0]
    biggest_quality_gap = by_product.sort_values(["missing_rate", "defects"], ascending=False).iloc[0]

    return {
        "weakest_product": str(weakest["product"]),
        "strongest_product": str(strongest["product"]),
        "quality_focus": str(biggest_quality_gap["product"]),
        "weakest_confidence": float(weakest["confidence"]),
        "strongest_confidence": float(strongest["confidence"]),
        "missing_rate": float(biggest_quality_gap["missing_rate"]),
        "validation_pass": float(biggest_quality_gap["validation_pass"]),
        "defects": int(biggest_quality_gap["defects"]),
    }


def render_story_summary(filtered: pd.DataFrame) -> None:
    story = build_story_context(filtered)
    confidence_gap = story["strongest_confidence"] - story["weakest_confidence"]
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-title">Executive takeaway</div>
            <div class="story-body">
                Prioritise <b>{story["quality_focus"]}</b> first because it carries the clearest trust risk:
                {story["missing_rate"]:.1f}% missingness, {story["validation_pass"]:.1f}% validation pass rate,
                and {story["defects"]:,} open defects. Use <b>{story["strongest_product"]}</b> as the benchmark pattern.
            </div>
        </div>
        <div class="value-strip">
            <div class="value-item">
                <div class="brief-kicker">Decision changed</div>
                <div class="brief-text">Move effort from broad dashboard rollout to targeted remediation of {story["quality_focus"]}.</div>
            </div>
            <div class="value-item">
                <div class="brief-kicker">Risk reduced</div>
                <div class="brief-text">Avoid using a lower-trust product for forecasting or executive reporting before validation improves.</div>
            </div>
            <div class="value-item">
                <div class="brief-kicker">Value created</div>
                <div class="brief-text">Close a {confidence_gap:.1f} point confidence gap by fixing quality blockers before scaling adoption.</div>
            </div>
        </div>
        <div class="action-table">
            <div class="action-row signal-head">
                <div>Priority</div><div>Action</div><div>Measure</div><div>Outcome</div>
            </div>
            <div class="action-row">
                <div>{story["quality_focus"]}</div>
                <div>Stabilise missingness and validation rules</div>
                <div>Validation pass above 90%</div>
                <div>Ready for trusted reporting</div>
            </div>
            <div class="action-row">
                <div>{story["strongest_product"]}</div>
                <div>Replicate adoption pattern</div>
                <div>Confidence maintained while usage grows</div>
                <div>Scale self-service safely</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(filtered: pd.DataFrame, all_data: pd.DataFrame) -> None:
    latest_week = filtered["week"].max()
    prior_week = latest_week - pd.Timedelta(days=28)
    current = filtered.loc[filtered["week"] == latest_week]
    previous = filtered.loc[filtered["week"].between(prior_week - pd.Timedelta(days=7), prior_week)]

    adoption = current["dashboard_adoption"].mean()
    adoption_prev = previous["dashboard_adoption"].mean()
    quality = current["data_quality"].mean()
    confidence = current["insight_confidence"].mean()
    cycle = current["cycle_time_days"].mean()
    decisions = current["decisions_supported"].sum()
    users = current["active_users"].sum()

    st.markdown("# Trusted Data Product Dashboard")
    st.markdown(
        '<div class="subtitle">A practical dashboard for data health, quality actions, and analytics readiness.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="pill-row">
            <span class="pill">Reporting period: {filtered["week"].min():%d %b %Y} to {latest_week:%d %b %Y}</span>
            <span class="pill">Rows in view: {len(filtered):,} of {len(all_data):,}</span>
            <span class="pill">Focus: trusted data, clear actions, measurable outcomes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Self-service adoption",
            fmt_percent(adoption),
            f"{fmt_delta(adoption - adoption_prev)} vs prior month",
            PALETTE["teal"],
        )
    with c2:
        render_metric_card(
            "Records validated",
            f"{current['records_processed_m'].sum():.1f}M",
            "Warehouse-style checks across curated analytical datasets",
            PALETTE["blue"],
        )
    with c3:
        render_metric_card(
            "Evidence confidence",
            fmt_percent(confidence),
            f"Quality at {quality:.1f}% with requirement coverage included",
            PALETTE["green"] if confidence >= 88 else PALETTE["amber"],
        )
    with c4:
        render_metric_card(
            "Analysis cycle time",
            f"{cycle:.1f} days",
            "Average brief-to-insight delivery speed",
            PALETTE["violet"],
        )


def render_sidebar(df: pd.DataFrame) -> FilterState:
    st.sidebar.title("Controls")
    st.sidebar.caption("Start with the decision, then narrow the evidence set.")

    stakeholder_decision = st.sidebar.selectbox(
        "Decision moment",
        [
            "Where should we intervene next?",
            "Can this dashboard be trusted?",
            "Which use case is ready to test?",
            "What should the next sprint deliver?",
        ],
    )
    st.sidebar.caption(f"Framing: {stakeholder_decision}")

    divisions = st.sidebar.multiselect(
        "Division",
        sorted(df["division"].unique()),
        default=sorted(df["division"].unique()),
    )
    products = st.sidebar.multiselect(
        "Data product",
        sorted(df["product"].unique()),
        default=sorted(df["product"].unique()),
    )
    priority = st.sidebar.multiselect(
        "Business priority",
        sorted(df["priority"].unique()),
        default=sorted(df["priority"].unique()),
    )

    min_date = df["week"].min().date()
    max_date = df["week"].max().date()
    selected = st.sidebar.date_input(
        "Reporting window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(selected) != 2:
        selected = (min_date, max_date)

    st.sidebar.divider()
    st.sidebar.markdown("**Checklist**")
    st.sidebar.checkbox("Problem statement agreed", value=True)
    st.sidebar.checkbox("Dataset contract checked", value=True)
    st.sidebar.checkbox("Trend driver reviewed", value=True)
    st.sidebar.checkbox("Pilot ready", value=False)

    return FilterState(
        divisions=divisions or sorted(df["division"].unique()),
        products=products or sorted(df["product"].unique()),
        priority=priority or sorted(df["priority"].unique()),
        date_range=(pd.Timestamp(selected[0]), pd.Timestamp(selected[1])),
    )


def render_trends(filtered: pd.DataFrame) -> None:
    trend = (
        filtered.groupby("week", as_index=False)
        .agg(
            dashboard_adoption=("dashboard_adoption", "mean"),
            data_quality=("data_quality", "mean"),
            insight_confidence=("insight_confidence", "mean"),
            cycle_time_days=("cycle_time_days", "mean"),
            defects_open=("defects_open", "sum"),
        )
        .set_index("week")
    )
    story = build_story_context(filtered)

    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: #2764D8;">
            <div class="story-title">Performance question</div>
            <div class="story-body">
                Is usage growing faster than trust? If <b>{story["quality_focus"]}</b> gains adoption before defects and validation improve,
                the business may scale a product that users cannot fully rely on.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Evidence Confidence Trend")
        st.line_chart(
            trend[["dashboard_adoption", "data_quality", "insight_confidence"]],
            color=[PALETTE["teal"], PALETTE["green"], PALETTE["blue"]],
            height=320,
        )
    with right:
        st.subheader("Defect Pressure")
        st.bar_chart(trend[["defects_open"]], color=PALETTE["amber"], height=320)


def render_driver_matrix(filtered: pd.DataFrame) -> None:
    by_division = (
        filtered.groupby("division", as_index=False)
        .agg(
            adoption=("dashboard_adoption", "mean"),
            quality=("data_quality", "mean"),
            confidence=("insight_confidence", "mean"),
            spec_coverage=("spec_coverage", "mean"),
            cycle_time=("cycle_time_days", "mean"),
            decisions=("decisions_supported", "sum"),
            defects=("defects_open", "sum"),
        )
        .sort_values("decisions", ascending=False)
    )
    by_division["quality_status"] = np.select(
        [by_division["quality"] >= 90, by_division["quality"] >= 84],
        ["Healthy", "Watch"],
        default="Risk",
    )

    st.subheader("Division Driver Matrix")
    st.dataframe(
        by_division,
        width="stretch",
        hide_index=True,
        column_config={
            "division": st.column_config.TextColumn("Division"),
            "adoption": st.column_config.ProgressColumn(
                "Adoption",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "quality": st.column_config.ProgressColumn(
                "Quality",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "confidence": st.column_config.ProgressColumn(
                "Evidence confidence",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "spec_coverage": st.column_config.ProgressColumn(
                "Spec coverage",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "cycle_time": st.column_config.NumberColumn("Cycle time", format="%.1f days"),
            "decisions": st.column_config.NumberColumn("Decisions", format="%d"),
            "defects": st.column_config.NumberColumn("Open defects", format="%d"),
            "quality_status": st.column_config.TextColumn("Status"),
        },
    )


def render_problem_canvas(filtered: pd.DataFrame) -> None:
    top_division = filtered.groupby("division")["decisions_supported"].sum().idxmax()
    weakest_product = filtered.groupby("product")["insight_confidence"].mean().idxmin()
    fastest_gain = filtered.groupby("product")["dashboard_adoption"].mean().idxmax()
    story = build_story_context(filtered)

    st.subheader("Decision Canvas")
    st.markdown(
        f"""
        <div class="brief-grid">
            <div class="brief-cell">
                <div class="brief-kicker">Decision question</div>
                <div class="brief-text">Which data product should the group prioritise to improve decision confidence this month?</div>
            </div>
            <div class="brief-cell" style="border-left-color: {PALETTE["blue"]};">
                <div class="brief-kicker">Evidence needed</div>
                <div class="brief-text">Usage, quality, lineage, acceptance coverage, and open issues by week, division, and product.</div>
            </div>
            <div class="brief-cell" style="border-left-color: {PALETTE["amber"]};">
                <div class="brief-kicker">Recommended action</div>
                <div class="brief-text">Focus on {weakest_product} for confidence lift; use {fastest_gain} as the adoption pattern to replicate in {top_division}.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_decomposition(filtered: pd.DataFrame) -> None:
    signals = (
        filtered.groupby("product", as_index=False)
        .agg(
            adoption=("dashboard_adoption", "mean"),
            quality=("data_quality", "mean"),
            coverage=("spec_coverage", "mean"),
            confidence=("insight_confidence", "mean"),
        )
        .sort_values("confidence", ascending=False)
    )
    signals["decision"] = np.where(
        signals["confidence"] >= 88,
        "Scale",
        np.where(signals["quality"] < 86, "Fix data", "Validate requirements"),
    )

    rows = [
        f"""
        <div class="signal-row">
            <div>{row.product}</div>
            <div>{row.adoption:.0f}%</div>
            <div>{row.quality:.0f}%</div>
            <div>{row.coverage:.0f}%</div>
            <div>{row.confidence:.0f}%</div>
            <div><b>{row.decision}</b></div>
        </div>
        """
        for row in signals.itertuples()
    ]
    st.subheader("Driver Decomposition")
    st.markdown(
        """
        <div class="section-card">
            <div class="signal-row signal-head">
                <div>Data product</div><div>Adoption</div><div>Quality</div><div>Spec</div><div>Confidence</div><div>Decision</div>
            </div>
        """
        + "\n".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_contracts(contracts: pd.DataFrame, active_products: Iterable[str]) -> None:
    scoped = contracts[contracts["product"].isin(active_products)].copy()
    if scoped.empty:
        scoped = contracts.copy()

    st.subheader("Curated Dataset Contracts")
    st.dataframe(
        scoped,
        width="stretch",
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Data Product"),
            "curated_dataset": st.column_config.TextColumn("Curated Dataset", width="medium"),
            "refresh_sla": st.column_config.TextColumn("Refresh SLA"),
            "release_status": st.column_config.TextColumn("Release Status"),
            "lineage_score": st.column_config.ProgressColumn("Lineage", min_value=0, max_value=100, format="%d%%"),
            "acceptance_coverage": st.column_config.ProgressColumn("Acceptance", min_value=0, max_value=100, format="%d%%"),
            "primary_users": st.column_config.TextColumn("Primary Users", width="medium"),
        },
    )


def render_validation_lab(filtered: pd.DataFrame) -> None:
    validation = (
        filtered.groupby("product", as_index=False)
        .agg(
            missing_rate=("missing_rate", "mean"),
            validation_pass_rate=("validation_pass_rate", "mean"),
            quality=("data_quality", "mean"),
            records_processed_m=("records_processed_m", "sum"),
        )
        .sort_values("validation_pass_rate")
    )
    validation["recommended_treatment"] = np.select(
        [
            validation["missing_rate"] >= 14,
            validation["validation_pass_rate"] < 87,
            validation["quality"] >= 92,
        ],
        [
            "Benchmark imputation before modelling",
            "Reconcile source-to-mart rules",
            "Ready for downstream analysis",
        ],
        default="Monitor with weekly anomaly checks",
    )
    highest_missing = validation.sort_values("missing_rate", ascending=False).iloc[0]

    st.subheader("Validation and Missing-Data Lab")
    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: {PALETTE["amber"]};">
            <div class="story-title">Data health insight</div>
            <div class="story-body">
                <b>{highest_missing["product"]}</b> has the highest missing-rate signal at {highest_missing["missing_rate"]:.1f}%.
                The value for the user is knowing when to pause, fix the data, and avoid false confidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        validation,
        width="stretch",
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Data Product"),
            "missing_rate": st.column_config.ProgressColumn("Missing Rate", min_value=0, max_value=30, format="%.1f%%"),
            "validation_pass_rate": st.column_config.ProgressColumn("Validation Pass", min_value=0, max_value=100, format="%.1f%%"),
            "quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f%%"),
            "records_processed_m": st.column_config.NumberColumn("Records Processed", format="%.1fM"),
            "recommended_treatment": st.column_config.TextColumn("Recommended Treatment", width="large"),
        },
    )

    st.markdown(
        """
        <div class="section-card">
            <b>Action guide</b><br>
            Products with high missingness or lower validation pass rates should be stabilised before they are used for reporting, forecasting, or stakeholder decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_issues(issues: pd.DataFrame, active_products: Iterable[str], active_divisions: Iterable[str]) -> None:
    scoped = issues[
        issues["product"].isin(active_products) & issues["division"].isin(active_divisions)
    ].copy()
    if scoped.empty:
        scoped = issues.copy()

    st.subheader("Data Quality and Delivery Queue")
    top_issue = scoped.sort_values("impact_score", ascending=False).iloc[0]
    high_count = int(scoped["severity"].eq("High").sum())
    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: {PALETTE["red"]};">
            <div class="story-title">Queue insight</div>
            <div class="story-body">
                The highest-impact blocker is <b>{top_issue["issue"]}</b> in <b>{top_issue["product"]}</b>
                with an impact score of {top_issue["impact_score"]}. There are <b>{high_count}</b> high-severity items in scope,
                so the queue should be worked by impact rather than arrival order.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        scoped.sort_values(["severity", "impact_score"], ascending=[True, False]),
        width="stretch",
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Data Product"),
            "division": st.column_config.TextColumn("Division"),
            "severity": st.column_config.TextColumn("Severity"),
            "priority": st.column_config.TextColumn("Priority"),
            "impact_score": st.column_config.ProgressColumn("Impact", min_value=0, max_value=100, format="%d"),
            "issue": st.column_config.TextColumn("Issue", width="large"),
            "next_action": st.column_config.TextColumn("Next Action"),
        },
    )


def render_product_evidence_health(filtered: pd.DataFrame) -> None:
    quality_trend = (
        filtered.groupby(["product"], as_index=False)
        .agg(
            data_quality=("data_quality", "mean"),
            spec_coverage=("spec_coverage", "mean"),
            insight_confidence=("insight_confidence", "mean"),
            defects_open=("defects_open", "sum"),
        )
        .sort_values("data_quality")
    )
    weakest = quality_trend.sort_values(["insight_confidence", "data_quality"]).iloc[0]
    strongest = quality_trend.sort_values(["insight_confidence", "data_quality"], ascending=False).iloc[0]

    st.subheader("Product Evidence Health")
    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: {PALETTE["blue"]};">
            <div class="story-title">Evidence health insight</div>
            <div class="story-body">
                <b>{weakest["product"]}</b> is the weakest evidence product at {weakest["insight_confidence"]:.1f}% confidence
                and {weakest["defects_open"]:,} open defects. <b>{strongest["product"]}</b> is the cleanest benchmark at
                {strongest["insight_confidence"]:.1f}% confidence. Use this view to decide what can be trusted now versus what needs remediation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        quality_trend,
        width="stretch",
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Data Product"),
            "data_quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f%%"),
            "spec_coverage": st.column_config.ProgressColumn("Spec Coverage", min_value=0, max_value=100, format="%.1f%%"),
            "insight_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
            "defects_open": st.column_config.NumberColumn("Open Defects", format="%d"),
        },
    )


def render_use_cases(use_cases: pd.DataFrame) -> None:
    st.subheader("Analytics Pipeline")
    visible_cases = use_cases.drop(columns=["method"])
    top_case = visible_cases.sort_values("readiness", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: {PALETTE["violet"]};">
            <div class="story-title">Pipeline decision</div>
            <div class="story-body">
                <b>{top_case["use_case"]}</b> is the most ready item in the pipeline at {top_case["readiness"]}%.
                The pipeline turns "interesting ideas" into a prioritised list of what can safely move forward.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        visible_cases.sort_values("readiness", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "use_case": st.column_config.TextColumn("Use Case", width="medium"),
            "stage": st.column_config.TextColumn("Stage"),
            "readiness": st.column_config.ProgressColumn("Readiness", min_value=0, max_value=100, format="%d%%"),
            "latest_update": st.column_config.TextColumn("Latest Update", width="large"),
        },
    )


def render_model_scorecard(filtered: pd.DataFrame, use_cases: pd.DataFrame) -> None:
    latest = filtered.loc[filtered["week"] == filtered["week"].max()]
    model_inputs = (
        latest.groupby("product", as_index=False)
        .agg(
            quality=("data_quality", "mean"),
            coverage=("spec_coverage", "mean"),
            adoption=("dashboard_adoption", "mean"),
            defects=("defects_open", "sum"),
        )
    )
    model_inputs["testability"] = np.clip(
        model_inputs["quality"] * 0.38
        + model_inputs["coverage"] * 0.42
        + model_inputs["adoption"] * 0.20
        - model_inputs["defects"] * 0.35,
        0,
        100,
    ).round(1)
    model_inputs["recommended_test"] = np.where(
        model_inputs["testability"] >= 78,
        "Pilot model",
        np.where(model_inputs["quality"] < 86, "Stabilise data", "Tighten acceptance criteria"),
    )

    st.subheader("Predictive Analytics Testability")
    st.markdown("**Quality vs Coverage by Product**")
    chart_inputs = model_inputs.rename(
        columns={
            "product": "Product",
            "quality": "Quality",
            "coverage": "Coverage",
            "testability": "Testability",
        }
    )
    st.scatter_chart(
        chart_inputs,
        x="Coverage",
        y="Quality",
        size="Testability",
        color="Product",
        height=320,
    )
    st.dataframe(
        model_inputs.sort_values("testability", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Data Product"),
            "quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.1f%%"),
            "coverage": st.column_config.ProgressColumn("Spec Coverage", min_value=0, max_value=100, format="%.1f%%"),
            "adoption": st.column_config.ProgressColumn("Adoption", min_value=0, max_value=100, format="%.1f%%"),
            "testability": st.column_config.ProgressColumn("Testability", min_value=0, max_value=100, format="%.1f%%"),
            "recommended_test": st.column_config.TextColumn("Recommendation", width="medium"),
        },
    )

    st.caption(
        f"{len(use_cases)} analytics use cases are tracked against the same readiness logic: data quality, requirement coverage, adoption signal, and known defects."
    )


def render_trained_model(df: pd.DataFrame, filtered: pd.DataFrame) -> float:
    model, feature_importance, _, accuracy = train_data_product_model(df)
    feature_cols = [
        "dashboard_adoption",
        "data_quality",
        "spec_coverage",
        "missing_rate",
        "validation_pass_rate",
        "insight_confidence",
        "cycle_time_days",
        "defects_open",
    ]
    latest = (
        filtered.loc[filtered["week"] == filtered["week"].max()]
        .groupby("product", as_index=False)
        .agg(
            dashboard_adoption=("dashboard_adoption", "mean"),
            data_quality=("data_quality", "mean"),
            spec_coverage=("spec_coverage", "mean"),
            missing_rate=("missing_rate", "mean"),
            validation_pass_rate=("validation_pass_rate", "mean"),
            insight_confidence=("insight_confidence", "mean"),
            cycle_time_days=("cycle_time_days", "mean"),
            defects_open=("defects_open", "sum"),
        )
    )
    probabilities = model.predict_proba(latest[feature_cols])
    class_labels = list(model.classes_)
    predictions = latest[["product", "missing_rate", "validation_pass_rate", "insight_confidence", "defects_open"]].copy()
    predictions["predicted_status"] = model.predict(latest[feature_cols])
    predictions["model_confidence"] = probabilities.max(axis=1) * 100
    predictions["fix_first_probability"] = (
        probabilities[:, class_labels.index("Fix First")] * 100 if "Fix First" in class_labels else 0
    )
    predictions["recommended_action"] = np.select(
        [
            predictions["predicted_status"].eq("Fix First"),
            predictions["predicted_status"].eq("Watch"),
        ],
        [
            "Remediate before rollout",
            "Monitor before scaling",
        ],
        default="Ready to scale",
    )

    highest_risk = predictions.sort_values("fix_first_probability", ascending=False).iloc[0]
    st.subheader("Trained Risk Classifier")
    st.markdown(
        f"""
        <div class="story-card" style="border-left-color: {PALETTE["red"]};">
            <div class="story-title">Model insight</div>
            <div class="story-body">
                A Random Forest classifier was trained on historical product signals to classify each product as
                <b>Ready</b>, <b>Watch</b>, or <b>Fix First</b>. It flags <b>{highest_risk["product"]}</b>
                as the highest-risk product with a {highest_risk["fix_first_probability"]:.1f}% Fix First probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.dataframe(
            predictions.sort_values("fix_first_probability", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={
                "product": st.column_config.TextColumn("Data Product"),
                "missing_rate": st.column_config.ProgressColumn("Missing Rate", min_value=0, max_value=30, format="%.1f%%"),
                "validation_pass_rate": st.column_config.ProgressColumn("Validation Pass", min_value=0, max_value=100, format="%.1f%%"),
                "insight_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                "defects_open": st.column_config.NumberColumn("Open Defects", format="%d"),
                "predicted_status": st.column_config.TextColumn("Predicted Status"),
                "model_confidence": st.column_config.ProgressColumn("Model Confidence", min_value=0, max_value=100, format="%.1f%%"),
                "fix_first_probability": st.column_config.ProgressColumn("Fix First Probability", min_value=0, max_value=100, format="%.1f%%"),
                "recommended_action": st.column_config.TextColumn("Recommended Action", width="medium"),
            },
        )
    with c2:
        st.markdown(f"**Holdout accuracy:** {accuracy:.1%}")
        st.markdown("**Model Feature Importance**")
        st.bar_chart(feature_importance.set_index("Feature"), color=PALETTE["blue"], height=305)

    return float(predictions["model_confidence"].mean())


def render_insight_summary(filtered: pd.DataFrame) -> None:
    latest = filtered["week"].max()
    current = filtered.loc[filtered["week"] == latest]
    earliest = filtered.loc[filtered["week"] == filtered["week"].min()]
    adoption_delta = current["dashboard_adoption"].mean() - earliest["dashboard_adoption"].mean()
    defects_delta = pct_change(current["defects_open"].sum(), earliest["defects_open"].sum())
    cycle_delta = current["cycle_time_days"].mean() - earliest["cycle_time_days"].mean()

    st.subheader("Insight Narrative")
    st.markdown(
        f"""
        <div class="section-card">
            Adoption has moved <span class="status-good">{fmt_delta(adoption_delta)}</span> across the selected reporting window,
            while open defects have shifted <span class="status-watch">{fmt_delta(defects_delta, "%")}</span>.
            Average cycle time is now <span class="status-good">{current["cycle_time_days"].mean():.1f} days</span>,
            a {abs(cycle_delta):.1f} day movement from the starting week. The next best action is to pair stakeholder enablement
            with targeted quality remediation so business teams can use the dashboards with confidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    df, issues, use_cases, contracts = build_dataset()
    state = render_sidebar(df)
    filtered = filter_data(df, state)

    if filtered.empty:
        st.warning("No data matched the selected filters. Broaden the controls to restore the dashboard.")
        return

    render_header(filtered, df)
    render_story_summary(filtered)
    st.divider()

    tab_framing, tab_overview, tab_quality, tab_analytics = st.tabs(
        ["Decision canvas", "Performance view", "Data health", "Analytics pipeline"]
    )

    with tab_framing:
        render_problem_canvas(filtered)
        render_signal_decomposition(filtered)
        render_insight_summary(filtered)

    with tab_overview:
        render_trends(filtered)
        render_driver_matrix(filtered)

    with tab_quality:
        render_contracts(contracts, state.products)
        render_validation_lab(filtered)
        render_issues(issues, state.products, state.divisions)
        render_product_evidence_health(filtered)

    with tab_analytics:
        render_use_cases(use_cases)
        model_confidence = render_trained_model(df, filtered)
        render_model_scorecard(filtered, use_cases)
        st.subheader("Scenario Modelling Inputs")
        uplift = st.slider("Adoption uplift assumption", min_value=0, max_value=20, value=7, step=1)
        quality_gain = st.slider("Quality remediation gain", min_value=0, max_value=12, value=4, step=1)
        latest = filtered.loc[filtered["week"] == filtered["week"].max()]
        projected_decisions = int(latest["decisions_supported"].sum() * (1 + uplift / 100))
        projected_quality = min(99.0, latest["data_quality"].mean() + quality_gain)
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected decisions", f"{projected_decisions:,}", f"{uplift}% adoption scenario")
        c2.metric("Projected quality", fmt_percent(projected_quality), f"{quality_gain} pt remediation")
        c3.metric("Model confidence", fmt_percent(model_confidence), "Random Forest risk classifier")


if __name__ == "__main__":
    main()
