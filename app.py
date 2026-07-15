"""
HR Intelligence Suite -- a modern, dark-themed Streamlit ML dashboard.

Run with:  streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="HR Intelligence Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SALARY_DATA_PATH = os.path.join(BASE_DIR, "Salary.csv")
SALARY_SCALER_PATH = os.path.join(BASE_DIR, "salary_scaler.pkl")
ATTRITION_DATA_PATH = os.path.join(BASE_DIR, "Attrition.csv")
ATTRITION_MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
ATTRITION_OVERTIME_ENCODER_PATH = os.path.join(BASE_DIR, "overTime_encoder.pkl")
ATTRITION_BUSINESS_ENCODER_PATH = os.path.join(BASE_DIR, "business_encoder.pkl")

DEPARTMENTS = [
    "Sales & Marketing", "Operations", "Technology", "Analytics",
    "R&D", "Procurement", "Finance", "HR", "Legal",
]
DEPT_DUMMY_COLS = [
    "department_Finance", "department_HR", "department_Legal",
    "department_Operations", "department_Procurement", "department_R&D",
    "department_Sales & Marketing", "department_Technology",
]

EDUCATION_MAP = {"Master's & above": 3, "Bachelor's": 2, "Below Secondary": 1}
CHANNEL_MAP = {"sourcing": 3, "referred": 2, "other": 1}

# NOTE: verify this matches the exact numeric codes your salary notebook's
# encoder used for "Education Level" (e.g. LabelEncoder.classes_ order).
# This is a reasonable default (ordinal, low -> high) -- edit if different.
SALARY_EDUCATION_MAP = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
}


@st.cache_data
def load_salary_metadata():
    data = pd.read_csv(SALARY_DATA_PATH)
    gender_categories = sorted(data["Gender"].dropna().unique().tolist())
    country_categories = sorted(data["Country"].dropna().unique().tolist())
    job_title_categories = sorted(data["Job Title"].dropna().unique().tolist())
    return gender_categories, country_categories, job_title_categories


SALARY_GENDERS, SALARY_COUNTRIES, SALARY_JOB_TITLES = load_salary_metadata()
SALARY_FEATURE_ORDER = [
    "Age",
    "Education Level",
    "Years of Experience",
    "Senior",
] + [f"Gender_{gender}" for gender in SALARY_GENDERS[1:]] + [
    f"Country_{country}" for country in SALARY_COUNTRIES[1:]
] + [
    f"Job Title_{job_title}" for job_title in SALARY_JOB_TITLES[1:]
]


@st.cache_data
def load_attrition_metadata():
    data = pd.read_csv(ATTRITION_DATA_PATH)
    business_travel_options = sorted(data["BusinessTravel"].dropna().unique().tolist())
    overtime_options = sorted(data["OverTime"].dropna().unique().tolist())
    return business_travel_options, overtime_options


ATTRITION_BUSINESS_TRAVEL_OPTIONS, ATTRITION_OVERTIME_OPTIONS = load_attrition_metadata()
ATTRITION_BUSINESS_TRAVEL_LABELS = {
    0: "Non-Travel",
    1: "Travel_Frequently",
    2: "Travel_Rarely",
}
ATTRITION_OVERTIME_LABELS = {
    0: "No",
    1: "Yes",
}
ATTRITION_FEATURE_ORDER = [
    "Age",
    "OverTime",
    "BusinessTravel",
    "TotalWorkingYears",
    "MonthlyIncome",
    "StockOptionLevel",
    "Total_Satisfaction",
    "Income_to_Experience",
    "Job_Hopper_Index",
]

# =============================================================================
# GLOBAL CSS -- dark theme, rounded cards, modern type
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at 10% 0%, #14192b 0%, #0b0e17 45%, #0a0c14 100%);
    color: #e7e9f3;
}

/* Hide default Streamlit chrome, keep the sidebar expanded on desktop, and only show the toggle on small screens */
#MainMenu, footer {visibility: hidden;}



div[data-testid="collapsedControl"] {
    display: none;
}

@media (min-width: 901px) {
    section[data-testid="stSidebar"] {
        display: block !important;
        transform: none !important;
        width: 18rem !important;
        min-width: 18rem !important;
    }
    header {
    visibility: hidden;
}
}

@media (max-width: 900px) {
    div[data-testid="collapsedControl"] {
        display: flex;
    }
    header {
    visibility: visible;
}

    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.9rem;
    }

    .stApp {
        padding-top: 0.25rem;
    }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12162a 0%, #0d1120 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* Generic card */
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-bottom: 1.1rem;
}
.glass-card:hover {
    border: 1px solid rgba(124,131,253,0.45);
    transition: all 0.25s ease-in-out;
}

/* Hero */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c83fd 0%, #63d9c9 55%, #ffd166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #9aa1c3;
    font-size: 1.05rem;
    max-width: 720px;
    line-height: 1.55rem;
}

/* Model cards */
.model-card {
    background: linear-gradient(160deg, rgba(124,131,253,0.10), rgba(99,217,201,0.04));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.6rem;
    height: 100%;
}
.model-card h3 { margin-bottom: 0.35rem; font-weight: 700; }
.model-card p { color: #a4abcb; font-size: 0.92rem; line-height: 1.4rem; }
.model-icon {
    font-size: 2.1rem;
    margin-bottom: 0.6rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    background: rgba(124,131,253,0.18);
    color: #b7bbff;
    border: 1px solid rgba(124,131,253,0.35);
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.20), rgba(52,211,153,0.05));
    border: 1px solid rgba(52,211,153,0.45);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(248,113,113,0.20), rgba(248,113,113,0.05));
    border: 1px solid rgba(248,113,113,0.45);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-neutral {
    background: linear-gradient(135deg, rgba(255,209,102,0.20), rgba(255,209,102,0.05));
    border: 1px solid rgba(255,209,102,0.45);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-emoji { font-size: 3rem; }
.result-text { font-size: 1.7rem; font-weight: 800; margin-top: 0.3rem; }

/* Section title */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 1.4rem 0 0.9rem 0;
    color: #f1f2fb;
}
.section-title span { color: #7c83fd; }

/* Team card */
.team-card {
    text-align: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.2rem 0.8rem;
}
.team-avatar {
    width: 62px; height: 62px; border-radius: 50%;
    background: linear-gradient(135deg, #7c83fd, #63d9c9);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.2rem; color: #0b0e17;
    margin: 0 auto 0.6rem auto;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7291;
    font-size: 0.85rem;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 2rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #7c83fd, #6a6ff5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 6px 18px rgba(124,131,253,0.35);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(124,131,253,0.5);
}

/* Inputs */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stToggle"] label,
div[data-testid="stCheckbox"] label {
    color: #f4f7ff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    margin-bottom: 0.35rem !important;
}
div[data-baseweb="select"] > div:hover,
.stNumberInput input:hover,
.stTextInput input:hover {
    border-color: rgba(99,217,201,0.55) !important;
}

div[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: rgba(124,131,253,0.85) !important;
    box-shadow: 0 0 0 2px rgba(124,131,253,0.18) !important;
}

div[data-baseweb="select"] span {
    color: #f5f7ff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 0.2rem;
    padding-bottom: 0.5rem;
}

.stSlider [data-baseweb="slider"] * {
    font-size: 0.98rem !important;
}

/* Metric-style small stat */
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.stat-pill .num { font-size: 1.4rem; font-weight: 800; color: #f1f2fb; }
.stat-pill .lbl { font-size: 0.78rem; color: #9aa1c3; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
@st.cache_resource
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_joblib_artifact(path: str):
    if not os.path.exists(path):
        return None
    import joblib

    return joblib.load(path)


def department_one_hot(selected_dept: str) -> dict:
    row = {col: 0 for col in DEPT_DUMMY_COLS}
    col_name = f"department_{selected_dept}"
    if col_name in row:
        row[col_name] = 1
    return row  # Analytics -> stays all zeros (reference category)


def gauge_chart(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 34, "color": "#f1f2fb"}},
        title={"text": title, "font": {"size": 14, "color": "#9aa1c3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7291"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(255,255,255,0.04)"},
                {"range": [50, 100], "color": "rgba(255,255,255,0.08)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        font={"color": "#e7e9f3"},
    )
    return fig


def performance_chart(
    title: str,
    labels: list[str],
    values: list[float],
    color: str,
    y_label: str,
    y_range: tuple[float, float] = (0.0, 1.0),
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=color,
            text=[f"{value:.2f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#f1f2fb"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        height=600,
        margin=dict(l=15, r=15, t=55, b=35),
        font={"color": "#e7e9f3"},
        yaxis={
            "title": y_label,
            "range": [y_range[0], y_range[1]],
            "gridcolor": "rgba(255,255,255,0.08)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
        },
        xaxis={
            "tickfont": {"size": 12},
        },
    )
    return fig


def one_hot_selected_value(selected_value: str, categories: list[str], prefix: str) -> dict:
    encoded = {f"{prefix}_{value}": 0 for value in categories[1:]}
    if selected_value in categories[1:]:
        encoded[f"{prefix}_{selected_value}"] = 1
    return encoded


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 0.4rem 0 1.2rem 0;'>"
        "<span style='font-size:2rem;'>🧠</span>"
        "<h2 style='margin:0; font-weight:800;'>HR Intelligence</h2>"
        "<p style='color:#7c83fd; font-size:0.8rem; letter-spacing:0.08em;'>ML DASHBOARD SUITE</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Home", "Promotion Prediction", "Salary Prediction", "Attrition Prediction", "About"],
        icons=["house", "graph-up-arrow", "cash-coin", "door-open", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#7c83fd", "font-size": "16px"},
            "nav-link": {
                "font-size": "14.5px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "4px 0",
                "border-radius": "12px",
                "color": "#c7cbe6",
                "padding": "10px 14px",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, rgba(124,131,253,0.28), rgba(99,217,201,0.10))",
                "color": "#ffffff",
                "font-weight": "700",
            },
        },
    )

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='glass-card' style='padding:1rem 1.1rem;'>"
        "<p style='font-size:0.78rem; color:#9aa1c3; margin:0;'>Model Status</p>"
        "<p style='font-size:0.85rem; margin:0.3rem 0 0 0;'>"
        "✅ Promotion &nbsp;&nbsp;✅ Salary &nbsp;&nbsp;✅ Attrition</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# HOME PAGE
# =============================================================================
if selected == "Home":
    st.markdown("<div class='hero-title'>HR Intelligence Suite</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>A unified machine learning workspace for people-analytics teams — "
        "predict promotion likelihood, estimate fair salary bands, and flag attrition risk, "
        "all from one clean, professional dashboard.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in zip(
        [c1, c2, c3, c4],
        ["3", "16+", "300", "~99%"],
        ["ML Models", "Input Features", "Trees / Forest", "Uptime"],
    ):
        col.markdown(
            f"<div class='stat-pill'><div class='num'>{num}</div>"
            f"<div class='lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Our <span>Models</span></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            "<div class='model-card'><div class='model-icon'>📈</div>"
            "<span class='badge'>CLASSIFICATION</span>"
            "<h3>Promotion Prediction</h3>"
            "<p>Predicts whether an employee is likely to be promoted based on training scores, "
            "ratings, tenure, department and more — helping HR plan career growth proactively.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            "<div class='model-card'><div class='model-icon'>💰</div>"
            "<span class='badge'>REGRESSION</span>"
            "<h3>Salary Prediction</h3>"
            "<p>Estimates a fair, data-driven monthly salary using role, experience, education "
            "and performance signals to support consistent, unbiased compensation decisions.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            "<div class='model-card'><div class='model-icon'>🚪</div>"
            "<span class='badge'>CLASSIFICATION</span>"
            "<h3>Attrition Prediction</h3>"
            "<p>Flags employees at risk of leaving using satisfaction, workload and tenure "
            "signals, so managers can intervene early with retention strategies.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Meet the <span>Team</span></div>", unsafe_allow_html=True)
    t1, t2, t3, t4,t5 = st.columns(5)
    team = [
        ("HK", "Habiba Khaled", "ML Engineer"),
        ("AA", "Ahmed Abozeid", "ML Engineer"),
        ("AI", "Abdallah Ibrahim", "ML Engineer"),
        ("JM", "Joudy Mahmoud", "ML Engineer"),
        ("ZH", "Zainah Hossam", "ML Engineer"),
       
    ]
    for col, (initials, name, role) in zip([t1, t2, t3, t4,t5], team):
        col.markdown(
            f"<div class='team-card'><div class='team-avatar'>{initials}</div>"
            f"<p style='font-weight:700; margin:0;'>{name}</p>"
            f"<p style='color:#9aa1c3; font-size:0.8rem; margin:0;'>{role}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer'>Built with ❤️ using Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; "
        "© 2026 HR Intelligence Suite</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# PROMOTION PREDICTION PAGE
# =============================================================================
elif selected == "Promotion Prediction":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>📈 Promotion Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Fill in the employee profile below. All inputs are automatically "
        "converted to the exact encoding the trained model expects.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        department = st.selectbox("🏢 Department", DEPARTMENTS)
        education_label = st.selectbox("🎓 Education", list(EDUCATION_MAP.keys()))
        channel_label = st.selectbox("📥 Recruitment Channel", list(CHANNEL_MAP.keys()))
        no_of_trainings = st.number_input("📚 Number of Trainings", min_value=1, max_value=15, value=2)
        age = st.slider("🎂 Age", 18, 65, 30)

    with colB:
        previous_year_rating = st.slider("⭐ Previous Year Rating", 1, 5, 3)
        length_of_service = st.slider("🗓️ Length of Service (years)", 0, 40, 5)
        avg_training_score = st.slider("🧪 Average Training Score", 30, 100, 65)
        awards_won_label = st.toggle("🏆 Awards Won?", value=False)

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict Promotion", use_container_width=True)

    if predict_clicked:
        row = {
            "education": EDUCATION_MAP[education_label],
            "recruitment_channel": CHANNEL_MAP[channel_label],
            "no_of_trainings": no_of_trainings,
            "age": age,
            "previous_year_rating": previous_year_rating,
            "length_of_service": length_of_service,
            "awards_won?": 1 if awards_won_label else 0,
            "avg_training_score": avg_training_score,
        }
        row.update(department_one_hot(department))

        feature_order = [
            "education", "recruitment_channel", "no_of_trainings", "age",
            "previous_year_rating", "length_of_service", "awards_won?",
            "avg_training_score",
        ] + DEPT_DUMMY_COLS

        input_df = pd.DataFrame([row])[feature_order]
        import joblib

        model = joblib.load("models/Promotion_Model.pkl")
        if model is None:
            st.error("⚠️ promotion_model.pkl not found in the `models/` folder. Please add your trained model.")
        else:
            prediction = model.predict(input_df)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1])
            with r1:
                if prediction == 1:
                    st.markdown(
                        "<div class='result-positive'><div class='result-emoji'>✅</div>"
                        "<div class='result-text'>Promoted</div>"
                        "<p style='color:#a4abcb;'>This employee shows strong promotion signals.</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-negative'><div class='result-emoji'>❌</div>"
                        "<div class='result-text'>Not Promoted</div>"
                        "<p style='color:#a4abcb;'>This profile currently doesn't meet promotion criteria.</p></div>",
                        unsafe_allow_html=True,
                    )
            with r2:
                if proba is not None:
                    st.plotly_chart(gauge_chart(proba, "Promotion Probability", "#7c83fd"), use_container_width=True)
                    st.progress(float(proba))

# =============================================================================
# SALARY PREDICTION PAGE
# =============================================================================
elif selected == "Salary Prediction":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>💰 Salary Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Estimate a fair monthly salary using the same fields and encoding "
        "used in the salary notebook.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        age_s = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1, key="sal_age")
        education_s_label = st.selectbox(
            "🎓 Education Level", list(SALARY_EDUCATION_MAP.keys()), index=1, key="sal_edu"
        )
        years_of_experience_s = st.number_input(
            "💼 Years of Experience", min_value=0, max_value=60, value=5, step=1, key="sal_exp"
        )
        senior_s = st.selectbox("⭐ Senior", [0, 1], index=0, key="sal_senior")
    with colB:
        gender_s = st.selectbox("🚻 Gender", SALARY_GENDERS, index=0, key="sal_gender")
        country_s = st.selectbox("🌍 Country", SALARY_COUNTRIES, index=4 if "USA" in SALARY_COUNTRIES else 0, key="sal_country")
        job_title_s = st.selectbox("🧩 Job Title", SALARY_JOB_TITLES, index=0, key="sal_title")
    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked_s = st.button("💡 Predict Salary", use_container_width=True)

    if predict_clicked_s:
        row = {
            "Age": age_s,
            "Education Level": SALARY_EDUCATION_MAP[education_s_label],
            "Years of Experience": years_of_experience_s,
            "Senior": senior_s,
        }
        row.update(one_hot_selected_value(gender_s, SALARY_GENDERS, "Gender"))
        row.update(one_hot_selected_value(country_s, SALARY_COUNTRIES, "Country"))
        row.update(one_hot_selected_value(job_title_s, SALARY_JOB_TITLES, "Job Title"))

        input_df = pd.DataFrame([row]).reindex(columns=SALARY_FEATURE_ORDER, fill_value=0)

        model = load_joblib_artifact(os.path.join(MODEL_DIR, "salary_model.pkl"))
        scaler = load_joblib_artifact(SALARY_SCALER_PATH)
        if model is None:
            st.error("⚠️ salary_model.pkl not found in the `models/` folder. Please add your trained model.")
        elif scaler is None:
            st.error("⚠️ salary_scaler.pkl not found in the project root. Please add your trained scaler.")
        else:
            scaled_input = scaler.transform(input_df)
            predicted_salary = float(model.predict(scaled_input)[0])
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='result-neutral'><div class='result-emoji'>💵</div>"
                f"<div class='result-text'>${predicted_salary:,.0f} / year</div>"
                f"<p style='color:#a4abcb;'>Estimated fair annually compensation for this profile.</p></div>",
                unsafe_allow_html=True,
            )
            low, high = predicted_salary * 0.9, predicted_salary * 1.1
            st.markdown(
                f"<p style='text-align:center; color:#9aa1c3; margin-top:0.8rem;'>"
                f"Typical range: ${low:,.0f} – ${high:,.0f}</p>",
                unsafe_allow_html=True,
            )

# =============================================================================
# ATTRITION PREDICTION PAGE
# =============================================================================
elif selected == "Attrition Prediction":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>🚪 Attrition Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub'>Predict attrition with the new 9-feature notebook pipeline using "
        "encoded travel/overtime values and derived experience metrics.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        age_a = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1, key="att_age")
        business_travel_str = st.selectbox(
            "✈️ Business Travel",
            ATTRITION_BUSINESS_TRAVEL_OPTIONS,
            index=(
                ATTRITION_BUSINESS_TRAVEL_OPTIONS.index("Travel_Rarely")
                if "Travel_Rarely" in ATTRITION_BUSINESS_TRAVEL_OPTIONS
                else 0
            ),
            key="att_travel",
        )
        total_working_years = st.number_input(
            "💼 Total Working Years", min_value=0, max_value=60, value=8, step=1, key="att_twy"
        )
        monthly_income = st.number_input(
            "💵 Monthly Income", min_value=0, max_value=200000, value=5000, step=100, key="att_income"
        )
    with colB:
        overtime_str = st.selectbox(
            "⏱️ OverTime",
            ATTRITION_OVERTIME_OPTIONS,
            index=0,
            key="att_overtime",
        )
        stock_option_level = st.selectbox("📈 Stock Option Level", [0, 1, 2, 3], index=0, key="att_stock")
        total_satisfaction = st.slider("😊 Total Satisfaction", 1.0, 4.0, 2.5, 0.1, key="att_sat")
        num_companies_worked = st.number_input(
            "🏭 Number of Companies Worked", min_value=0, max_value=20, value=2, step=1, key="att_companies"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Business Travel and OverTime are encoded using the saved notebook label encoders before prediction.")

    predict_clicked_a = st.button("🔍 Predict Attrition Risk", use_container_width=True)

    if predict_clicked_a:
        row = {
            "Age": age_a,
            "OverTime": overtime_str,
            "BusinessTravel": business_travel_str,
            "TotalWorkingYears": total_working_years,
            "MonthlyIncome": monthly_income,
            "StockOptionLevel": stock_option_level,
            "Total_Satisfaction": total_satisfaction,
            "Income_to_Experience": monthly_income / (total_working_years + 1),
            "Job_Hopper_Index": num_companies_worked / (total_working_years + 1),
        }
        input_df = pd.DataFrame([row])[ATTRITION_FEATURE_ORDER]

        model = load_joblib_artifact(ATTRITION_MODEL_PATH)
        overtime_encoder = load_joblib_artifact(ATTRITION_OVERTIME_ENCODER_PATH)
        business_travel_encoder = load_joblib_artifact(ATTRITION_BUSINESS_ENCODER_PATH)
        if model is None:
            st.error("⚠️ attrition_model.pkl not found in the `models/` folder. Please add the new attrition model.")
        elif overtime_encoder is None or business_travel_encoder is None:
            st.error("⚠️ attrition encoder files not found in the project root. Please add both encoder pickles.")
        else:
            input_df["OverTime"] = overtime_encoder.transform(input_df['OverTime'])
            input_df["BusinessTravel"] = business_travel_encoder.transform(input_df['BusinessTravel'])
            prediction = model.predict(input_df)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 1])
            with r1:
                if prediction == 1:
                    st.markdown(
                        "<div class='result-negative'><div class='result-emoji'>🚨</div>"
                        "<div class='result-text'>High Attrition Risk</div>"
                        "<p style='color:#a4abcb;'>Consider a retention conversation soon.</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-positive'><div class='result-emoji'>✅</div>"
                        "<div class='result-text'>Likely to Stay</div>"
                        "<p style='color:#a4abcb;'>Low attrition risk detected for this profile.</p></div>",
                        unsafe_allow_html=True,
                    )
            with r2:
                if proba is not None:
                    st.plotly_chart(gauge_chart(proba, "Attrition Risk", "#f87171"), use_container_width=True)
                    st.progress(float(proba))

# =============================================================================
# ABOUT PAGE
# =============================================================================
elif selected == "About":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>ℹ️ About This Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-sub' style='max-width:100%;'>"
        "The <b>HR Intelligence Suite</b> is a machine-learning powered dashboard built to "
        "demonstrate end-to-end, production-style ML deployment for people-analytics use cases. "
        "It bundles three independent models — promotion, salary, and attrition prediction — "
        "behind one clean, unified interface, with model benchmark charts for quick comparison.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model <span>Benchmarks</span></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            performance_chart(
                "Salary Prediction R²",
                ["Linear Regression", "SVR", "Random Forest"],
                [0.83, 0.79, 0.95],
                ["#7c83fd", "#63d9c9", "#ffd166"],
                "R² Score",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            performance_chart(
                "Promotion Prediction Accuracy",
                ["Naive Bayes", "Logistic Regression", "Random Forest"],
                [0.63, 0.79, 0.89],
                ["#f87171", "#7c83fd", "#63d9c9"],
                "Accuracy",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(
            performance_chart(
                "Attrition Prediction Accuracy",
                ["Random Forest", "Logistic Regression", "SVM", "KNN"],
                [0.850340, 0.721088, 0.789116, 0.768707],
                ["#63d9c9", "#7c83fd", "#ffd166", "#f87171"],
                "Accuracy",
                (0.0, 1.0),
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            "<div class='glass-card'><h3>🧰 Tech Stack</h3>"
            "<p style='color:#a4abcb;'>Streamlit · scikit-learn · Plotly · Pandas · NumPy · "
            "streamlit-option-menu</p></div>",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            "<div class='glass-card'><h3>📦 How Predictions Work</h3>"
            "<p style='color:#a4abcb;'>User-friendly inputs are converted into the exact encoded "
            "feature format each trained model expects, in the exact column order used during "
            "training, before being passed to the model for inference.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer'>Built with ❤️ using Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; "
        "© 2026 HR Intelligence Suite</div>",
        unsafe_allow_html=True,
    )