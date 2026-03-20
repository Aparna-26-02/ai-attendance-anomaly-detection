
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Attendance Monitor", layout="wide")

# ---------------- CUSTOM CSS STYLE ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    background-attachment: fixed;
}

/* Main title */
.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 48px;
    font-weight: 700;
    color: white;
    text-align: center;
    padding: 40px 30px;
    border-radius: 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    margin-bottom: 40px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
}

/* Section headers */
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 28px;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 35px;
    margin-bottom: 20px;
    padding-left: 5px;
    border-left: 5px solid #667eea;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 30px 20px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    text-align: center;
    border-top: 4px solid #667eea;
}

.metric-card h3 {
    color: #7c8fa3;
    font-size: 14px;
}

.metric-card h2 {
    color: #2c3e50;
    font-size: 36px;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 15px;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px 28px;
    border: none;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
/* Professional Reset Button (White Text) */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s ease;
}

/* Hover effect */
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #5a6fd8, #6a3fa0);
}

/* Click effect */
div[data-testid="stButton"] > button:active {
    transform: scale(0.96);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
'<div class="main-title"> Employee Attendance Behavior Analysis</div>',
unsafe_allow_html=True
)

# ---------------- SESSION STATE ----------------
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "mapping_confirmed" not in st.session_state:
    st.session_state.mapping_confirmed = False

if "validated" not in st.session_state:
    st.session_state.validated = False

# ---------------- RESET BUTTON (GLOBAL) ----------------
col1, col2 = st.columns([10, 1])

with col2:
    if st.button("⟲ Reset"):
        st.session_state.file_uploaded = False
        st.session_state.mapping_confirmed = False

        keys_to_remove = [
            "df",
            "emp_col",
            "login_col",
            "logout_col",
            "date_col",
            "dept_col"
        ]

        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()
# ---------------- TIME CONVERSION ----------------
def time_to_minutes(time_str):
    try:
        if pd.isna(time_str):
            return None
        parts = str(time_str).split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes
    except:
        return None


# =====================================================
# STEP 1 : UPLOAD DATASET
# =====================================================
# ---------------- HELPER FUNCTIONS ----------------
def is_time_column(series):
    try:
        sample = series.dropna().astype(str).head(10)
        for val in sample:
            if ":" not in val:
                return False
        return True
    except:
        return False


def is_date_column(series):
    try:
        sample = series.dropna().astype(str).head(20)

        # Try flexible parsing
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)

        # If most values parsed → it's a date column
        success_ratio = parsed.notna().sum() / len(sample)

        return success_ratio > 0.7   # 70% threshold
    except:
        return False

if not st.session_state.file_uploaded:

    st.markdown('<div class="section-title"> Upload Attendance Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Attendance Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV is empty. Please upload a valid dataset.")
            st.stop()

        st.session_state.df = df
        st.session_state.file_uploaded = True
        st.rerun()



# =====================================================
# STEP 2 : COLUMN SELECTION
# =====================================================

elif not st.session_state.mapping_confirmed:

    df = st.session_state.df

    # Initialize validation flag
    if "validated" not in st.session_state:
        st.session_state.validated = False

    st.markdown('<div class="section-title"> Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    columns = df.columns.tolist()

    st.markdown('<div class="section-title"> Select Required Columns</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        emp_col = st.selectbox("Employee ID Column", columns)
        login_col = st.selectbox("Login Time Column", columns)
    with col2:
        logout_col = st.selectbox("Logout Time Column", columns)
        date_col = st.selectbox("Date Column", columns)

    dept_col = st.selectbox(
        "Department Column (Optional)",
        ["None"] + columns
    )

    # ---------------- VALIDATION DISPLAY (ONLY AFTER CLICK) ----------------
    if st.session_state.validated:

        error_messages = []

        if not is_time_column(df[login_col]):
            error_messages.append("Login Time column must be in HH:MM format")

        if not is_time_column(df[logout_col]):
            error_messages.append("Logout Time column must be in HH:MM format")

        if not is_date_column(df[date_col]):
            error_messages.append("Date column must be a valid date format")

        if error_messages:
            st.error("❌ Invalid column selection:\n\n" + "\n".join(error_messages))

    # ---------------- CONFIRM BUTTON ----------------
    if st.button("Confirm Column Mapping", use_container_width=True):

        st.session_state.validated = True  # trigger validation

        error_messages = []

        if not is_time_column(df[login_col]):
            error_messages.append("Login Time column must be in HH:MM format")

        if not is_time_column(df[logout_col]):
            error_messages.append("Logout Time column must be in HH:MM format")

        if not is_date_column(df[date_col]):
            error_messages.append("Date column must be a valid date format")

        if error_messages:
            st.warning("⚠️ Please fix the errors before proceeding.")
        else:
            st.session_state.emp_col = emp_col
            st.session_state.login_col = login_col
            st.session_state.logout_col = logout_col
            st.session_state.date_col = date_col
            st.session_state.dept_col = dept_col

            st.session_state.mapping_confirmed = True
            st.session_state.validated = False  # reset state
            st.rerun()
# =====================================================
# STEP 3 : PREPROCESS + MODEL
# =====================================================

else:

    df = st.session_state.df.copy()

    emp_col = st.session_state.emp_col
    login_col = st.session_state.login_col
    logout_col = st.session_state.logout_col
    date_col = st.session_state.date_col
    dept_col = st.session_state.dept_col

    df = df.rename(columns={
        emp_col: "employee_id",
        login_col: "login_time",
        logout_col: "logout_time",
        date_col: "date"
    }).copy()

    df = df.dropna(subset=["login_time", "logout_time"])

    df["login_min"] = df["login_time"].apply(time_to_minutes)
    df["logout_min"] = df["logout_time"].apply(time_to_minutes)

    df = df.dropna(subset=["login_min", "logout_min"])

    df["total_work_min"] = (df["logout_min"] - df["login_min"] + 1440) % 1440
    df["work_hours"] = df["total_work_min"] / 60

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    df["day_of_week"] = df["date_dt"].dt.day_name()

    df["login_deviation"] = df["login_min"] - 540

    if dept_col == "None":
        df["department"] = "General"
    else:
        df["department"] = df[dept_col]

    df["day_enc"] = LabelEncoder().fit_transform(df["day_of_week"].astype(str))
    df["dept_enc"] = LabelEncoder().fit_transform(df["department"].astype(str))

    st.markdown('<div class="section-title"> Processed Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    contamination = st.slider(" Anomaly Sensitivity (%)", 1, 20, 7) / 100

    features = [
        "login_min",
        "logout_min",
        "total_work_min",
        "login_deviation",
        "day_enc",
        "dept_enc"
    ]

    if st.button(" Run AI Attendance Analysis", use_container_width=True):

        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])

        iso = IsolationForest(contamination=contamination, random_state=42)
        lof = LocalOutlierFactor(contamination=contamination)

        df["iso"] = iso.fit_predict(X)
        df["lof"] = lof.fit_predict(X)

        def status(row):
            if row["iso"] == -1 and row["lof"] == -1:
                return "High Anomaly"
            elif row["iso"] == -1 or row["lof"] == -1:
                return "Unusual"
            else:
                return "Normal"

        df["status"] = df.apply(status, axis=1)

        def reason(row):
            if row["work_hours"] < 6:
                return "Short Work Duration"
            if row["login_deviation"] > 60:
                return "Very Late Login"
            if row["login_deviation"] < -90:
                return "Unusually Early Login"
            return "Behavior Pattern Deviation"

        df["reason"] = df.apply(reason, axis=1)

        st.markdown('<div class="section-title"> Detection Summary</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.markdown(f'<div class="metric-card"><h3>Total Records</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><h3>Normal Records</h3><h2>{(df["status"] == "Normal").sum()}</h2></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><h3>Anomalies</h3><h2>{(df["status"] != "Normal").sum()}</h2></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title"> Work Duration Distribution</div>', unsafe_allow_html=True)

        fig1 = px.histogram(df, x="work_hours", nbins=30, color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<div class="section-title"> Login Time vs Work Duration</div>', unsafe_allow_html=True)

        fig2 = px.scatter(
            df,
            x="login_min",
            y="work_hours",
            color="status",
            color_discrete_map={
                "Normal": "#2ecc71",
                "Unusual": "#f39c12",
                "High Anomaly": "#e74c3c"
            },
            hover_data=["employee_id"]
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title"> Anomalies by Employee</div>', unsafe_allow_html=True)

        emp_anomaly = df[df["status"] != "Normal"].groupby("employee_id").size().reset_index(name="count")

        fig3 = px.bar(emp_anomaly, x="employee_id", y="count", color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-title"> Anomalies by Department</div>', unsafe_allow_html=True)

        dept_anomaly = df[df["status"] != "Normal"].groupby("department").size().reset_index(name="count")

        fig4 = px.bar(dept_anomaly, x="department", y="count", color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-title"> Attendance Work Hours Trend</div>', unsafe_allow_html=True)

        trend = df.groupby("date_dt")["work_hours"].mean().reset_index()

        fig5 = px.line(trend, x="date_dt", y="work_hours", color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown('<div class="section-title"> Anomaly Records</div>', unsafe_allow_html=True)

        st.dataframe(
            df[df["status"] != "Normal"][[
                "employee_id",
                "date",
                "login_time",
                "logout_time",
                "work_hours",
                "status",
                "reason"
            ]],
            use_container_width=True
        )
        # Prepare anomaly data
        anomaly_df = df[df["status"] != "Normal"][[
            "employee_id",
            "date",
            "login_time",
            "logout_time",
            "work_hours",
            "status",
            "reason"
        ]]

# Convert to CSV
        csv = anomaly_df.to_csv(index=False).encode("utf-8")

# Download button
        st.download_button(
            label=" Download Anomaly Records as CSV",
            data=csv,
            file_name="anomaly_records.csv",
            mime="text/csv",
            use_container_width=True
        )
